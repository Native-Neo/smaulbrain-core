"""Focused tests for the experimental Dust activation-perturbation trainer.

These tests deliberately use very small sequences and populations so the whole
file stays in the normal CPU test budget.  The mathematical checks compare the
stochastic estimator with exact autograd on the *same* batch; they do not claim
that the two algorithms produce identical gradients (Dust intentionally uses
limited cross-token credit).
"""
import argparse
import os
import signal

import numpy as np
import pytest

from smaulbrain import checkpoint as ckpt
from smaulbrain import cli
from smaulbrain import config as cfg
from smaulbrain import data as D
from smaulbrain import dust
from smaulbrain import growth as G
from smaulbrain import model as M
from smaulbrain import npgrad as ng
from smaulbrain import train_loop as TL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model(preset="dense-tiny", seed=0, **kwargs):
    spec = cfg.PRESETS[preset]
    n_exp = spec.n_experts_init if spec.kind == "moe" else 0
    mtp_depth = kwargs.pop("mtp_depth", 0)
    params = M.build_params(
        spec, np.random.default_rng(seed), n_exp, mtp_depth=mtp_depth)
    return M.Model(spec, params, mtp_depth=mtp_depth, **kwargs), spec


def _tokens(seed=1, length=17):
    return np.random.default_rng(seed).integers(0, 256, length).astype(np.int64)


def _small_config(**overrides):
    values = dict(
        writer_draws=1, hidden_draws=1, hub_draws=1, embed_draws=1,
        head_draws=64, local_draws=1, chunk=4, sigma=0.05,
        hub_sigma=0.05, hub_sigma_deep=0.05, local_sigma=0.05,
        head_sigma=0.05, gamma=0.5, seed=123, lr=0.05,
    )
    values.update(overrides)
    return dust.DustConfig(**values)


def _args(tmp_path, extra=(), preset="dense-tiny", seq_len=16, trainer="dust"):
    ap = argparse.ArgumentParser()
    cli.add_common_args(ap, need_train=True)
    argv = [
        "--preset", preset, "--checkpoint", str(tmp_path / "run"),
        "--seq-len", str(seq_len), "--mtp-depth", "0",
        "--growth-every-steps", "0", "--trainer", trainer,
        "--seed", "7",
    ]
    argv.extend(extra)
    args = ap.parse_args(argv)
    cli.validate_args(args)
    return args


def _trainer(tmp_path, extra=(), preset="dense-tiny", seq_len=16,
             trainer="dust", data=None):
    args = _args(tmp_path, extra, preset, seq_len, trainer)
    if data is None:
        data = D.synth_bytes(4096, seed=3)
    return TL.Trainer(cfg.PRESETS[preset], args, D.ByteStream(data=data))


def _capture_y_grads(model, tokens, layer=0):
    """Exact dL/dY for one attention block, using the public autograd path."""
    captured = []
    original = model._attn

    def wrapped(*a, **kw):
        out = original(*a, **kw)
        captured.append(out[0])
        return out

    model._attn = wrapped
    try:
        logits, _ = model.forward(tokens[:-1], train=True)
        loss = ng.cross_entropy(logits, tokens[1:])
        loss.backward()
    finally:
        model._attn = original
    out_grad = captured[layer].grad
    return (np.asarray(out_grad) @ model.p[f"L{layer}.attn.Wo"].data.T).copy()


def _exact_logit_grad(clean):
    logits = clean.logits.astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    prob = np.exp(shifted)
    prob /= prob.sum(axis=-1, keepdims=True)
    prob[np.arange(clean.T), clean.targets] -= 1.0
    return prob


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ---------------------------------------------------------------------------
# Forward parity, codecs, and numerical stability
# ---------------------------------------------------------------------------
def test_dust_forward_matches_reference_and_is_non_contiguous_safe():
    model, spec = _model()
    toks = _tokens(2, 20)
    reference, _ = model.forward(toks[:-1], train=False)
    clean = dust.DustForward(model).capture(toks)
    np.testing.assert_allclose(clean.logits, reference.data, rtol=0, atol=2e-6)
    np.testing.assert_allclose(
        clean.loss,
        ng.cross_entropy(ng.Tensor(clean.logits), clean.targets,
                         reduction="none").data,
        rtol=0, atol=2e-6)

    # Every other token is a non-contiguous view; the forward must copy/gather
    # deterministically rather than depending on memory layout.
    strided = np.arange(0, 2 * len(toks), 2, dtype=np.int64)
    clean2 = dust.DustForward(model).capture(strided)
    contiguous = dust.DustForward(model).capture(np.ascontiguousarray(strided))
    np.testing.assert_array_equal(clean2.logits, contiguous.logits)


def test_per_token_cross_entropy_handles_noncontiguous_and_single_token():
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((16, 19)).astype(np.float32)
    base = np.zeros((16, 38), dtype=np.float32)
    base[:, ::2] = logits
    view = base[:, ::2]
    assert not view.flags.c_contiguous
    targets = rng.integers(0, 19, 16)
    got = dust.per_token_cross_entropy(view, targets)
    ref = ng.cross_entropy(ng.Tensor(logits), targets, reduction="none").data
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)

    one = dust.per_token_cross_entropy(logits[:1], targets[:1])
    assert one.shape == (1,)
    assert np.isfinite(one).all()


def test_future_credit_is_stable_for_small_and_unit_gamma():
    rng = np.random.default_rng(4)
    score = rng.standard_normal((3, 64)).astype(np.float32)
    for gamma in (0.0, 0.3, 0.5, 0.9, 0.98, 1.0):
        got = dust.future_credit(score, gamma)
        reference = np.empty_like(score)
        carry = np.zeros(score.shape[0], dtype=np.float64)
        for t in range(score.shape[1] - 1, -1, -1):
            carry = score[:, t] + gamma * carry
            reference[:, t] = carry
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, reference, rtol=0, atol=2e-5)


def test_dust_config_rejects_invalid_seed():
    with pytest.raises(ValueError, match="seed"):
        _small_config(seed=-1).validate(259)


def test_clean_forward_nan_is_rejected():
    model, _ = _model()
    model.p["embed"].data[0, 0] = np.nan
    engine = dust.DustEngine(model, _small_config())
    toks = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    with pytest.raises(FloatingPointError):
        engine.loss_and_gradients(toks)


# ---------------------------------------------------------------------------
# Estimator mathematics
# ---------------------------------------------------------------------------
def test_direct_estimator_matches_exact_gradient_on_one_token():
    # With T=1 there is no cross-token credit ambiguity: the direct estimator
    # must recover dL/dY up to population noise.
    model, _ = _model()
    toks = _tokens(6, 2)  # one predictor token
    exact = _capture_y_grads(model, toks, layer=0)
    clean = dust.DustForward(model).capture(toks)
    engine = dust.DustEngine(model, _small_config())
    stats = {"block_evals": 0, "local_block_evals": 0, "head_column_evals": 0,
             "draws": 0, "local_draws": 0, "head_draws": 0,
             "scalar_forwards": 0, "draw_batches": 0, "temp_bytes": 0}
    key = "L0.attn.hub"
    err = engine._direct_group(
        clean.x_in[0], 0, [key], {key: engine.d}, 512, 0.05,
        np.random.default_rng(7), clean, stats)[key]
    assert _cos(err, exact) > 0.8
    ratio = np.linalg.norm(err) / (np.linalg.norm(exact) + 1e-30)
    assert 0.6 < ratio < 1.5


_STATS = {"block_evals": 0, "local_block_evals": 0, "head_column_evals": 0,
          "draws": 0, "local_draws": 0, "head_draws": 0,
          "scalar_forwards": 0, "draw_batches": 0, "temp_bytes": 0}


def test_batch_centered_reward_correction_restores_linear_scale():
    # The documented estimator is -(1/(K sigma)) sum_i r_i a_i.  Removing the
    # per-batch reward mean shrinks E[c_i a_i] by (n - 1)/n; the correction
    # multiplies the centered batch by n/(n - 1) exactly, so the scale law is
    # independent of the draw batch size.  Pure linear rewards
    # r_i = -sigma * <g, a_i> + mu make the analytic corrigendum exact: the
    # corrected estimator returns g, the pre-fix batch-centred-only variant
    # returns ((n - 1) / n) * g.  mu is a constant mean-reward term (dust
    # rewards have one from second-order loss growth); its scale is ~ the
    # per-draw signal so it does not dominate the Monte-Carlo window, and
    # batch-centering must strip it in both variants.
    rng = np.random.default_rng(500)
    d, K, sigma = 8, 512, 0.05
    g = rng.standard_normal(d)
    mu = 0.005
    reps = 24
    for n in (1, 2, 4, 8):
        corrected_sum = np.zeros((1, d))
        bugged_sum = np.zeros((1, d))
        for _ in range(reps):
            for _ in range(K // n):
                a = rng.standard_normal((n, 1, d))
                r = -(sigma * (a @ g)) + mu
                if n > 1:
                    rb = r - r.mean(axis=0, keepdims=True)
                    rb *= n / (n - 1)
                    corrected_sum -= np.einsum("nt,ntd->td", rb, a) / sigma
                    bugged_sum -= np.einsum(
                        "nt,ntd->td", r - r.mean(axis=0, keepdims=True), a) / sigma
                else:
                    corrected_sum -= np.einsum("nt,ntd->td", r, a) / sigma
        corrected = corrected_sum / (reps * K)
        np.testing.assert_allclose(corrected, g[None, :], rtol=0.10, atol=0.10)
        if n > 1:
            bugged = bugged_sum / (reps * K)
            np.testing.assert_allclose(
                bugged, ((n - 1) / n) * g[None, :], rtol=0.10, atol=0.10)


def test_direct_estimator_scale_is_chunk_invariant():
    # The draw batch size is a performance knob.  After the n/(n - 1)
    # correction the estimator's expected scale no longer depends on it:
    # at T=1 there is no cross-token credit ambiguity, so the mean norm
    # ratio against the exact autograd gradient is ~1 for every chunk.
    model, _ = _model()
    toks = _tokens(6, 2)  # one predictor token
    exact = _capture_y_grads(model, toks, layer=0)
    exact_norm = np.linalg.norm(exact)
    clean = dust.DustForward(model).capture(toks)
    engine = dust.DustEngine(model, _small_config())
    key = "L0.attn.hub"
    per_chunk = {}
    for chunk in (1, 2, 4, 64):
        ratios = []
        for s in range(4):
            stats = dict(_STATS)
            cfgv = _small_config(chunk=chunk)
            engine_c = dust.DustEngine(model, cfgv)
            err = engine_c._direct_group(
                clean.x_in[0], 0, [key], {key: engine_c.d}, 512, 0.05,
                np.random.default_rng(s), clean, stats)[key]
            ratios.append(np.linalg.norm(err) / exact_norm)
        per_chunk[chunk] = float(np.mean(ratios))
    span = max(per_chunk.values()) - min(per_chunk.values())
    assert span < 0.2, per_chunk
    for chunk, mean_ratio in per_chunk.items():
        assert 0.85 < mean_ratio < 1.15, (chunk, mean_ratio, per_chunk)


def test_batched_direct_groups_match_separate_forwards():
    # The fast path evaluates hub/writer/hidden groups as one batch with zeros
    # outside each group's row.  It must reproduce separate forwards exactly
    # for the same RNG stream and draw count.
    model, _ = _model()
    toks = _tokens(10, 17)
    cfgv = _small_config(writer_draws=2, hub_draws=2, hidden_draws=2,
                         embed_draws=1, head_draws=16, local_draws=0,
                         chunk=4)
    engine = dust.DustEngine(model, cfgv)
    clean = dust.DustForward(model).capture(toks)
    keys = [f"L0.{name}" for name in
            ("attn.hub", "attn.out", "ffn.out", "ffn.a", "ffn.b")]
    hub, wo, wc, wa, wb = keys
    groups = [
        {"sites": [hub], "widths": {hub: engine.d},
         "draws": 2, "sigma": cfgv.hub_sigma},
        {"sites": [wo, wc],
         "widths": {wo: engine.d, wc: engine.d},
         "draws": 2, "sigma": cfgv.sigma},
        {"sites": [wa, wb],
         "widths": {wa: model.spec.ffn_hidden, wb: model.spec.ffn_hidden},
         "draws": 2, "sigma": cfgv.sigma},
    ]
    stats_a = {"block_evals": 0, "local_block_evals": 0,
               "head_column_evals": 0, "draws": 0, "local_draws": 0,
               "head_draws": 0, "scalar_forwards": 0, "draw_batches": 0,
               "temp_bytes": 0}
    stats_b = dict(stats_a)
    together = engine._direct_groups(
        clean.x_in[0], 0, groups,
        np.random.default_rng(np.random.SeedSequence([17, 0])),
        clean, stats_a)
    separate = {}
    rng = np.random.default_rng(np.random.SeedSequence([17, 0]))
    for group in groups:
        separate.update(engine._direct_group(
            clean.x_in[0], 0, group["sites"], group["widths"],
            group["draws"], group["sigma"], rng, clean, stats_b))
    for key in keys:
        np.testing.assert_array_equal(together[key], separate[key])
    assert stats_a["block_evals"] == stats_b["block_evals"]
    assert stats_a["draws"] == stats_b["draws"]


def test_parallel_head_matches_serial(monkeypatch):
    # The concurrent head pass must be deterministic and byte-identical to the
    # serial path; only wall time should differ.
    model, _ = _model()
    engine = dust.DustEngine(model, _small_config(
        writer_draws=5, hidden_draws=5, hub_draws=5, embed_draws=5,
        head_draws=256, local_draws=0, chunk=2))
    toks = _tokens(11, 33)
    monkeypatch.setattr(dust.os, "cpu_count", lambda: 1)
    loss_a, grads_a, _ = engine.loss_and_gradients(toks, step=4)
    monkeypatch.setattr(dust.os, "cpu_count", lambda: 4)
    loss_b, grads_b, _ = engine.loss_and_gradients(toks, step=4)
    assert loss_a == loss_b
    for name in grads_a:
        np.testing.assert_array_equal(grads_a[name], grads_b[name])


def test_local_attention_estimator_matches_exact_target():
    # Feed the *exact* per-token dL/dY target so this isolates the local
    # Jacobian estimate and the future-credit rule from hub noise.
    model, _ = _model()
    toks = _tokens(8, 9)
    exact_mean = _capture_y_grads(model, toks, layer=0)
    T = len(toks) - 1
    exact_per_token = exact_mean * T
    _, bp_grads, _ = model.loss_and_backward(toks)
    clean = dust.DustForward(model).capture(toks)
    engine = dust.DustEngine(model, _small_config(gamma=0.5))
    stats = {"block_evals": 0, "local_block_evals": 0, "head_column_evals": 0,
             "draws": 0, "local_draws": 0, "head_draws": 0,
             "scalar_forwards": 0, "draw_batches": 0, "temp_bytes": 0}

    rng = np.random.default_rng(11)
    err_r = engine._local_error_block(
        "r", 0, 128, 0.05, exact_per_token, rng, clean, stats)
    got_r = clean.h[0].T @ err_r / T
    assert _cos(got_r, bp_grads["L0.attn.Wr"]) > 0.85
    ratio_r = np.linalg.norm(got_r) / (np.linalg.norm(bp_grads["L0.attn.Wr"]) + 1e-30)
    assert 0.6 < ratio_r < 1.5

    err_k = engine._local_error_block(
        "k", 0, 256, 0.05, exact_per_token, rng, clean, stats)
    got_k = clean.h[0].T @ err_k / T
    assert _cos(got_k, bp_grads["L0.attn.Wk"]) > 0.7
    ratio_k = np.linalg.norm(got_k) / (np.linalg.norm(bp_grads["L0.attn.Wk"]) + 1e-30)
    assert 0.5 < ratio_k < 2.0


def test_head_slab_estimator_matches_exact_logit_gradient():
    model, _ = _model()
    toks = _tokens(9, 9)
    clean = dust.DustForward(model).capture(toks)
    exact = _exact_logit_grad(clean)
    engine = dust.DustEngine(model, _small_config())
    stats = {"block_evals": 0, "local_block_evals": 0, "head_column_evals": 0,
             "draws": 0, "local_draws": 0, "head_draws": 0,
             "scalar_forwards": 0, "draw_batches": 0, "temp_bytes": 0}
    for slab in (7, 259):
        engine.config.head_slab = slab
        engine.config.head_draws = 2048
        err = engine._head_errors(np.random.default_rng(13), clean, stats)
        assert _cos(err, exact) > 0.8, slab
        ratio = np.linalg.norm(err) / (np.linalg.norm(exact) + 1e-30)
        assert 0.6 < ratio < 1.5, slab


# ---------------------------------------------------------------------------
# Engine integration, reproducibility, and safe failure
# ---------------------------------------------------------------------------
def test_dust_never_invokes_autograd_or_backward(monkeypatch):
    model, _ = _model()
    engine = dust.DustEngine(model, _small_config(local_draws=0))

    def boom(*a, **k):
        raise AssertionError("Dust must not invoke autograd backward")

    monkeypatch.setattr(ng.Tensor, "backward", boom)
    monkeypatch.setattr(M.Model, "loss_and_backward", boom)
    loss, grads, info = engine.loss_and_gradients(_tokens(10, 13), step=0)
    assert np.isfinite(loss)
    assert len(grads) == 61  # 60 matrix sites + tied embedding
    assert all(np.isfinite(g).all() for g in grads.values())
    assert info["algorithm"] == "dust"


def test_reproducible_by_seed_and_step():
    model, _ = _model()
    engine = dust.DustEngine(model, _small_config())
    toks = _tokens(11, 13)
    _, a, _ = engine.loss_and_gradients(toks, step=9)
    _, b, _ = engine.loss_and_gradients(toks, step=9)
    assert a.keys() == b.keys()
    assert all(np.array_equal(a[k], b[k]) for k in a)
    _, c, _ = engine.loss_and_gradients(toks, step=10)
    assert any(not np.array_equal(a[k], c[k]) for k in a)


def test_default_scalar_policy_freezes_nonmatrix_and_local_zero_freezes_attention():
    model, _ = _model()
    engine = dust.DustEngine(model, _small_config(local_draws=1))
    _, grads, info = engine.loss_and_gradients(_tokens(12, 9), step=0)
    assert len(info["dust"]["frozen_params"]) == 61
    assert "L0.attn.Wr" in grads and "L0.attn.Wk" in grads
    assert "L0.attn.w" not in grads and "final_norm" not in grads

    engine0 = dust.DustEngine(model, _small_config(local_draws=0))
    _, grads0, info0 = engine0.loss_and_gradients(_tokens(12, 9), step=0)
    assert "L0.attn.Wr" not in grads0
    assert "L0.attn.Wk" not in grads0
    assert "L0.attn.Wv" not in grads0
    assert len(info0["dust"]["frozen_params"]) == 61 + 45


def test_scalar_es_is_restored_on_failure_and_returns_gradients():
    model, _ = _model()
    engine = dust.DustEngine(model, _small_config(scalar_draws=1))
    names = [n for n in model.p if n not in engine._matrix_sites]
    before = {n: model.p[n].data.copy() for n in names}
    stats = {"scalar_forwards": 0}

    original = engine.fw.loss_only

    def explode(_tokens):
        raise RuntimeError("synthetic scalar-forward failure")

    engine.fw.loss_only = explode
    try:
        with pytest.raises(RuntimeError):
            engine._scalar_gradients(
                _tokens(13, 9), np.random.default_rng(0), None, stats)
    finally:
        engine.fw.loss_only = original
    for name in names:
        np.testing.assert_array_equal(model.p[name].data, before[name])

    grads = engine._scalar_gradients(
        _tokens(13, 9), np.random.default_rng(1), None, stats)
    assert set(grads) == set(names)
    assert all(np.isfinite(g).all() for g in grads.values())
    assert stats["scalar_forwards"] == 2


def test_nonfinite_perturbation_is_rejected(monkeypatch):
    model, _ = _model()
    engine = dust.DustEngine(model, _small_config(local_draws=0))
    original = engine.fw.run_blocks

    def bad(*a, **k):
        logits, h, x = original(*a, **k)
        return np.full_like(logits, np.nan), h, x

    monkeypatch.setattr(engine.fw, "run_blocks", bad)
    with pytest.raises(FloatingPointError):
        engine.loss_and_gradients(_tokens(14, 9), step=0)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def test_dust_sgd_momentum_state_and_rejects_nonfinite_gradient():
    params = {"w": np.ones((2, 3), dtype=np.float32)}
    opt = dust.DustSGD(lr=0.1, momentum=0.9)
    opt.register(params)
    opt.step(params, {"w": np.ones_like(params["w"])})
    np.testing.assert_allclose(params["w"], np.full((2, 3), 0.9, np.float32),
                               rtol=1e-6)
    opt.step(params, {"w": np.zeros_like(params["w"])})
    np.testing.assert_allclose(params["w"], np.full((2, 3), 0.81, np.float32),
                               rtol=1e-6)

    state = opt.state_dict()
    params2 = {"w": np.full((2, 3), 0.81, np.float32)}
    opt2 = dust.DustSGD(lr=0.1, momentum=0.9)
    opt2.register(params2)
    opt2.load_state_dict(state)
    opt.step(params, {"w": np.zeros_like(params["w"])})
    opt2.step(params2, {"w": np.zeros_like(params2["w"])})
    np.testing.assert_allclose(params2["w"], params["w"], rtol=1e-6)

    before = params["w"].copy()
    with pytest.raises(FloatingPointError):
        opt.step(params, {"w": np.full((2, 3), np.nan, np.float32)})
    np.testing.assert_array_equal(params["w"], before)
    with pytest.raises(ValueError):
        dust.DustSGD(lr=0.0)
    with pytest.raises(ValueError):
        dust.DustSGD(momentum=1.0)


# ---------------------------------------------------------------------------
# Trainer integration: learning, checkpoints, resume, SIGINT, validation
# ---------------------------------------------------------------------------
def test_dust_training_updates_params_and_improves_heldout_loss(tmp_path):
    pattern = bytes([65, 66, 67, 68]) * 512
    heldout = np.frombuffer(bytes([65, 66, 67, 68]) * 64, dtype=np.uint8).astype(np.int64)
    tr = _trainer(
        tmp_path,
        extra=[
            "--dust-writer-draws", "2", "--dust-hidden-draws", "2",
            "--dust-hub-draws", "2", "--dust-embed-draws", "2",
            "--dust-local-draws", "2", "--dust-head-draws", "256",
            "--dust-chunk", "4", "--dust-lr", "0.1", "--dust-grad-clip", "0",
        ],
        seq_len=16, data=pattern)
    before = {k: v.data.copy() for k, v in tr.model.p.items()}
    initial = G.eval_loss_on_tokens(tr.model, heldout)
    for step in range(1, 7):
        tr.train(max_steps=step, log_every=0, ckpt_every=0, grad_clip=100.0)
    final = G.eval_loss_on_tokens(tr.model, heldout)
    assert initial > 4.0
    assert final < 2.0
    assert final < initial - 2.0
    assert any(not np.array_equal(before[k], v.data) for k, v in tr.model.p.items())
    assert np.isfinite(tr.last_metrics["loss"])
    assert tr.algorithm == "dust" and tr.opt.n_updates == 6


def test_dust_manifest_and_algorithm_mismatch(tmp_path):
    # A backprop-authored checkpoint must not silently load under Dust.
    bp = _trainer(tmp_path / "bp", trainer="backprop", seq_len=8)
    bp.save()
    man = ckpt.load_checkpoint(str(tmp_path / "bp" / "run"))["manifest"]
    assert man["config"]["algorithm"] == "backprop"
    dust_tr = _trainer(tmp_path / "dust", seq_len=8)
    with pytest.raises(ValueError, match="trained with 'backprop'"):
        dust_tr.load(tmp_path / "bp" / "run")

    # And a Dust checkpoint must not load under backprop.
    dust_tr.save()
    man = ckpt.load_checkpoint(str(tmp_path / "dust" / "run"))["manifest"]
    assert man["config"]["algorithm"] == "dust"
    assert man["config"]["dust"]["gamma"] == pytest.approx(0.5)
    assert man["optimizer"]["class"] == "DustSGD"
    bp2 = _trainer(tmp_path / "bp2", trainer="backprop", seq_len=8)
    with pytest.raises(ValueError, match="trained with 'dust'"):
        bp2.load(tmp_path / "dust" / "run")


def test_dust_checkpoint_resume_is_bitwise_exact(tmp_path):
    import shutil

    opts = [
        "--dust-writer-draws", "1", "--dust-hidden-draws", "1",
        "--dust-hub-draws", "1", "--dust-embed-draws", "1",
        "--dust-local-draws", "1", "--dust-head-draws", "64",
        "--dust-chunk", "4", "--dust-lr", "0.05", "--dust-grad-clip", "0",
    ]
    data = D.synth_bytes(2048, seed=3)
    first = _trainer(tmp_path / "a", extra=opts, seq_len=8, data=data)
    first.train(max_steps=2, log_every=0, ckpt_every=2, grad_clip=100.0)
    snapshot = tmp_path / "snapshot"
    shutil.copytree(tmp_path / "a" / "run", snapshot)
    params2 = {k: v.data.copy() for k, v in first.model.p.items()}
    momentum2 = {k: v["m"].copy() for k, v in first.opt.state.items()}
    first.train(max_steps=4, log_every=0, ckpt_every=0, grad_clip=100.0)
    params4 = {k: v.data.copy() for k, v in first.model.p.items()}
    momentum4 = {k: v["m"].copy() for k, v in first.opt.state.items()}

    resumed = _trainer(tmp_path / "b", extra=opts, seq_len=8, data=data)
    resumed.load(snapshot)
    assert resumed.sched.global_step == 2
    for name, value in params2.items():
        np.testing.assert_array_equal(resumed.model.p[name].data, value)
    for name, value in momentum2.items():
        np.testing.assert_array_equal(resumed.opt.state[name]["m"], value)
    resumed.train(max_steps=4, log_every=0, ckpt_every=0, grad_clip=100.0)
    assert resumed.sched.global_step == 4
    for name, value in params4.items():
        np.testing.assert_array_equal(resumed.model.p[name].data, value)
    for name, value in momentum4.items():
        np.testing.assert_array_equal(resumed.opt.state[name]["m"], value)


def test_dust_resume_restores_optimizer_hyperparameters(tmp_path):
    base = [
        "--dust-writer-draws", "1", "--dust-hidden-draws", "1",
        "--dust-hub-draws", "1", "--dust-embed-draws", "1",
        "--dust-local-draws", "1", "--dust-head-draws", "64",
        "--dust-chunk", "4", "--dust-grad-clip", "0",
    ]
    first = _trainer(
        tmp_path / "a", extra=base + ["--dust-lr", "0.05",
                                      "--dust-momentum", "0.9"],
        seq_len=8)
    first.train(max_steps=1, log_every=0, ckpt_every=1, grad_clip=100.0)
    changed = _trainer(
        tmp_path / "b", extra=base + ["--dust-lr", "0.02",
                                      "--dust-momentum", "0.8"],
        seq_len=8)
    changed.load(tmp_path / "a" / "run")
    assert changed.opt.lr == pytest.approx(0.05)
    assert changed.opt.momentum == pytest.approx(0.9)
    assert changed.dust_config.lr == pytest.approx(0.05)
    assert changed.dust_config.momentum == pytest.approx(0.9)
    assert changed.sched.global_step == 1


def test_checkpoint_carries_the_optimizer_update_counter(tmp_path):
    tr = _trainer(tmp_path, trainer="backprop", seq_len=8)
    tr.train(max_steps=3, log_every=0, ckpt_every=0, grad_clip=100.0)
    assert tr.opt.n_updates == 3
    tr.save()
    man = ckpt.load_checkpoint(str(tmp_path / "run"))["manifest"]
    assert man["optimizer"]["n_updates"] == 3

    resumed = _trainer(tmp_path / "resume", trainer="backprop", seq_len=8)
    resumed.load(tmp_path / "run")
    assert resumed.opt.n_updates == 3


def test_dust_ctrl_c_saves_at_committed_step(tmp_path, monkeypatch):
    opts = [
        "--dust-writer-draws", "1", "--dust-hidden-draws", "1",
        "--dust-hub-draws", "1", "--dust-embed-draws", "1",
        "--dust-local-draws", "1", "--dust-head-draws", "64",
        "--dust-chunk", "4", "--dust-lr", "0.05", "--dust-grad-clip", "0",
    ]
    tr = _trainer(tmp_path, extra=opts, seq_len=8)
    original = tr.dust.loss_and_gradients
    calls = {"n": 0}

    def interrupted(tokens, step=0):
        out = original(tokens, step=step)
        calls["n"] += 1
        if calls["n"] == 1:
            os.kill(os.getpid(), signal.SIGINT)
        return out

    monkeypatch.setattr(tr.dust, "loss_and_gradients", interrupted)
    with pytest.raises(KeyboardInterrupt):
        tr.train(max_steps=3, log_every=0, ckpt_every=0, grad_clip=100.0)
    man = ckpt.load_checkpoint(str(tmp_path / "run"))["manifest"]
    assert man["scheduler"]["global_step"] == 1
    assert man["config"]["algorithm"] == "dust"

    resumed = _trainer(tmp_path / "resume", extra=opts, seq_len=8)
    resumed.load(tmp_path / "run")
    assert resumed.sched.global_step == 1
    resumed.train(max_steps=3, log_every=0, ckpt_every=0, grad_clip=100.0)
    assert resumed.sched.global_step == 3


def test_dust_cli_defaults_are_wired(tmp_path):
    args = _args(tmp_path)
    dc = dust.config_from_args(args)
    assert dc.writer_draws == 2
    assert dc.local_draws == 2
    assert dc.head_draws == 1024
    assert dc.sigma == pytest.approx(0.05)
    assert dc.hub_sigma == pytest.approx(0.05)
    assert dc.hub_sigma_deep == pytest.approx(0.05)
    assert dc.gamma == pytest.approx(0.5)
    assert dc.scalar_draws == 0
    assert dc.seed == args.seed + 0xD057
    assert dc.grad_clip == 0.0


@pytest.mark.parametrize("extra", [
    ["--preset", "moe-tiny", "--mtp-depth", "0"],
    ["--mtp-depth", "1"],
    ["--forward-quant", "fp8"],
    ["--backward-mode", "fp8"],
    ["--activation-storage", "bf16"],
    ["--gradient-storage", "int8"],
    ["--dust-sigma", "0"],
    ["--dust-gamma", "1.5"],
    ["--dust-writer-draws", "0"],
    ["--dust-local-draws", "-1"],
    ["--dust-head-slab", "0"],
    ["--dust-lr", "0"],
    ["--dust-grad-clip", "-1"],
    ["--dust-seed", "-1"],
    ["--grad-clip", "0.5"],
])
def test_dust_cli_rejects_unsupported_or_invalid_configuration(tmp_path, extra):
    with pytest.raises((ValueError, SystemExit)):
        _args(tmp_path, extra=extra)


def test_dust_engine_rejects_moe_mtp_and_quantized_models():
    moe_spec = cfg.PRESETS["moe-tiny"]
    moe_params = M.build_params(
        moe_spec, np.random.default_rng(0), moe_spec.n_experts_init)
    moe = M.Model(moe_spec, moe_params)
    with pytest.raises(ValueError, match="dense models only"):
        dust.DustEngine(moe, _small_config())

    mtp, _ = _model(mtp_depth=1)
    with pytest.raises(ValueError, match="MTP"):
        dust.DustEngine(mtp, _small_config())

    quantized, _ = _model(forward_quant="fp8", backward_mode="fp8")
    with pytest.raises(ValueError, match="fp32"):
        dust.DustEngine(quantized, _small_config())


def test_backprop_storage_paths_still_work_with_dust_module_present():
    # The Dust work must not regress the existing BF16 storage / FP8 backward
    # path.  The full suite has deeper coverage; this is a small guard.
    spec = cfg.PRESETS["dense-tiny"]
    params = M.build_params(spec, np.random.default_rng(15))
    model = M.Model(spec, params, activation_storage="bf16",
                    gradient_storage="bf16")
    toks = _tokens(16, 9)
    loss, grads, _ = model.loss_and_backward(toks)
    assert np.isfinite(loss)
    assert grads and all(np.isfinite(g).all() for g in grads.values())

    fp8 = M.Model(spec, M.build_params(spec, np.random.default_rng(16)),
                  forward_quant="fp8", backward_mode="fp8",
                  activation_storage="bf16", gradient_storage="bf16")
    loss, grads, _ = fp8.loss_and_backward(toks)
    assert np.isfinite(loss)
    assert grads and all(np.isfinite(g).all() for g in grads.values())
