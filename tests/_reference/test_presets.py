"""Preset/config arithmetic + model core tests (critic-math audit).

Covers:
  * build_params / count_params_dict / expert_ids vs the config formulas and
    the EXPECTED_* tables (the arithmetic referenced by config.py);
  * Model._assert_counts validation;
  * Model._rmsnorm and Model._attn vs math_ref (fixed weights);
  * chunked forward equivalence: forward([a,b,c]) vs step_token, and two
    continuing calls vs one long forward (dense; MoE without drops);
  * MTP row alignment: depth k targets tok[i+k+1] from hidden[i] and
    embed(tok[i+k]), including T < depth;
  * loss_and_backward gradient sanity vs an independent autograd forward and
    a central finite-difference probe on final_norm.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from smaulbrain import config as cfg
from smaulbrain import math_ref
from smaulbrain import model as M
from smaulbrain.smaulopt import SmaulOPT

EPS = 1e-6


# --------------------------------------------------------------------------
# claim 7: preset arithmetic
# --------------------------------------------------------------------------
def test_dense_param_formulas_match_expected_tables():
    for name, expected in cfg.EXPECTED_DENSE_TOTAL.items():
        spec = cfg.PRESETS[name]
        assert cfg.dense_param_total(spec) == expected
        # formula: V*d + d + L*(4*d^2 + 4*d + 3*d*h)
        d, h, L, V = spec.d_model, spec.ffn_hidden, spec.n_layers, spec.vocab_size
        assert cfg.dense_param_total(spec) == V * d + d + L * (4 * d * d + 4 * d + 3 * d * h)


def test_expert_and_shared_formulas_match_expected_tables():
    for name, expected in cfg.EXPECTED_EXPERT_PARAMS.items():
        spec = cfg.PRESETS[name]
        assert cfg.expert_param_count(spec) == expected == 2 * spec.d_model * spec.expert_hidden
        d, L, V = spec.d_model, spec.n_layers, spec.vocab_size
        assert cfg.shared_param_total(spec) == V * d + d + L * (4 * d * d + 4 * d) + d * spec.n_experts_init
        assert cfg.whole_model_total(spec) == cfg.shared_param_total(spec) + \
            spec.n_experts_init * cfg.expert_param_count(spec)


def test_mtp_param_formula_and_arch_report():
    spec = cfg.PRESETS["dense-tiny"]
    d = spec.d_model
    assert cfg.mtp_param_count(spec, 0) == 0
    assert cfg.mtp_param_count(spec, 3) == 3 * (6 * d * d + 6 * d)
    rep = cfg.arch_report(spec, 0, mtp_depth=2)
    assert rep["mtp_params"] == 2 * (6 * d * d + 6 * d)
    assert rep["total_params"] == cfg.dense_param_total(spec) + rep["mtp_params"]
    rep_moe = cfg.arch_report(cfg.PRESETS["moe-tiny"], 32, mtp_depth=1)
    assert rep_moe["total_params"] == cfg.shared_param_total(cfg.PRESETS["moe-tiny"]) + \
        32 * cfg.expert_param_count(cfg.PRESETS["moe-tiny"]) + rep_moe["mtp_params"]


def test_build_params_counts_dense_and_moe():
    spec = cfg.PRESETS["dense-tiny"]
    p = M.build_params(spec, generator=torch.Generator().manual_seed(0))
    assert M.count_params_dict(p) == cfg.EXPECTED_DENSE_TOTAL["dense-tiny"]
    # model construction re-validates the exact count
    m = M.Model(spec, {k: v.clone() for k, v in p.items()})
    assert M.count_params_dict(m.p) == cfg.EXPECTED_DENSE_TOTAL["dense-tiny"]
    assert len(m.p) == 2 + 11 * spec.n_layers

    p3 = M.build_params(spec, generator=0, mtp_depth=3)
    assert M.count_params_dict(p3) == cfg.dense_param_total(spec) + cfg.mtp_param_count(spec, 3)
    m3 = M.Model(spec, p3, mtp_depth=3)
    assert M.count_params_dict(m3.p) == cfg.EXPECTED_DENSE_TOTAL["dense-tiny"] + cfg.mtp_param_count(spec, 3)
    m3._assert_mtp_counts()

    ms = cfg.PRESETS["moe-tiny"]
    pm = M.build_params(ms, generator=0)
    assert M.expert_ids(pm) == list(range(ms.n_experts_init))
    shared = {k: v for k, v in pm.items() if not k.startswith("expert.")}
    assert M.count_params_dict(shared) == cfg.shared_param_total(ms)
    for eid in M.expert_ids(pm):
        n = pm[f"expert.{eid}.Win"].numel() + pm[f"expert.{eid}.Wout"].numel()
        assert n == cfg.EXPECTED_EXPERT_PARAMS["moe-tiny"]
    assert M.count_params_dict(pm) == cfg.whole_model_total(ms)
    # build_params honours an explicit n_experts override
    pm48 = M.build_params(ms, generator=0, n_experts=48)
    assert M.expert_ids(pm48) == list(range(48))
    assert M.count_params_dict(pm48) == cfg.whole_model_total(ms, n_experts=48)


def test_moe_counts_track_router_width_across_growth_and_prune():
    """Count formulas must agree with build_params for any expert count: growth
    resizes the router to match the expert registry (validate_candidate), so
    the totals move by d*(n - n_experts_init) on top of n*expert_params."""
    spec = cfg.PRESETS["moe-tiny"]
    d = spec.d_model
    for n in (26, 30, 38, 48):
        p = M.build_params(spec, generator=0, n_experts=n)
        built = M.count_params_dict(p)
        assert p["router"].shape == (d, n)
        assert built == cfg.whole_model_total(spec, n_experts=n), n
        rep = cfg.arch_report(spec, n)
        assert rep["router_params"] == d * n
        assert rep["total_params"] == built, n
        assert M.Model(spec, p).report()["total_params"] == built, n
    # at the init expert count the router-delta term vanishes
    assert cfg.whole_model_total(spec) == 85248 + 32 * 10240


def test_count_helpers_and_assert_counts_validation():
    assert M.count_params_dict({"a": torch.zeros(2, 3), "b": torch.zeros(4)}) == 10
    assert M.expert_ids({}) == []
    assert M.expert_ids({"expert.10.Win": torch.zeros(1), "expert.2.Wout": torch.zeros(1)}) == [2, 10]
    spec = cfg.PRESETS["dense-tiny"]
    p = M.build_params(spec, generator=0)
    p["extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        M.Model(spec, p)  # param count drift is caught at construction
    p2 = M.build_params(spec, generator=0, mtp_depth=1)
    with pytest.raises(ValueError):
        M.Model(spec, p2, mtp_depth=0)  # MTP params without mtp_depth


# --------------------------------------------------------------------------
# claim 8: single-layer attention / rmsnorm vs math_ref
# --------------------------------------------------------------------------
@pytest.mark.parametrize("B", [1, 4])
def test_model_rmsnorm_matches_math_ref(B):
    torch.manual_seed(0)
    d = 32
    X = torch.randn(B, d) if B > 1 else torch.randn(d)
    gain = torch.randn(d)
    y = M.Model._rmsnorm(X, gain)
    rows = X.tolist() if B > 1 else [X.tolist()]
    got = y.tolist() if B > 1 else [y.tolist()]
    for b in range(B):
        ref = math_ref.rmsnorm_ref(rows[b], gain.tolist())
        assert max(abs(got[b][i] - ref[i]) for i in range(d)) <= 3e-6


@pytest.mark.parametrize("T", [1, 3, 6])
def test_single_layer_attn_matches_math_ref(T):
    torch.manual_seed(1)
    spec = cfg.PRESETS["dense-tiny"]
    model = M.Model(spec, M.build_params(spec, generator=torch.Generator().manual_seed(3)))
    p = model.p
    X = torch.randn(T, spec.d_model) * 0.3
    out, state = M.Model._attn(model, X, "L0.", None)
    Yr, sr = math_ref.recurrent_forward_ref(
        X.tolist(), p["L0.attn.Wr"].tolist(), p["L0.attn.Wk"].tolist(),
        p["L0.attn.Wv"].tolist(), p["L0.attn.Wo"].tolist(),
        p["L0.attn.w"].tolist(), p["L0.attn.u"].tolist())
    assert max(abs(out[i, j].item() - Yr[i][j]) for i in range(T)
               for j in range(spec.d_model)) <= 3e-5
    assert max(abs(state[j].item() - sr[j]) for j in range(spec.d_model)) <= 3e-5


def test_attn_state_continuation_matches_one_pass():
    torch.manual_seed(2)
    spec = cfg.PRESETS["dense-tiny"]
    model = M.Model(spec, M.build_params(spec, generator=0))
    X = torch.randn(6, spec.d_model)
    full, s_full = M.Model._attn(model, X, "L0.", None)
    a, sa = M.Model._attn(model, X[:2], "L0.", None)
    b, sb = M.Model._attn(model, X[2:], "L0.", sa)
    assert torch.equal(torch.cat([a, b]), full)
    assert torch.equal(sb, s_full)


# --------------------------------------------------------------------------
# claim 11: chunked / step-wise forward equivalence
# --------------------------------------------------------------------------
def _dense_model(depth=0, device="cpu"):
    spec = cfg.PRESETS["dense-tiny"]
    p = M.build_params(spec, generator=torch.Generator().manual_seed(4), mtp_depth=depth)
    return spec, M.Model(spec, p, mtp_depth=depth, device=device)


def test_forward_vs_step_token_dense():
    spec, model = _dense_model()
    toks = [11, 22, 33, 44]
    logits, states = model.forward(toks)
    st = model.init_states()
    seq = torch.stack([model.step_token(t, st) for t in toks])
    assert (seq - logits).abs().max().item() <= 1e-5
    assert max((a - b).abs().max().item() for a, b in zip(states, st)) == 0.0


def test_forward_continuing_states_dense():
    spec, model = _dense_model()
    toks = [3, 9, 27, 81]
    one, s1 = model.forward(toks)
    a, s2 = model.forward(toks[:2])
    b, s2 = model.forward(toks[2:], states=s2)
    assert (torch.cat([a, b]) - one).abs().max().item() <= 1e-5
    assert max((x - y).abs().max().item() for x, y in zip(s2, s1)) == 0.0


def test_forward_vs_step_token_moe_without_drops():
    spec = cfg.PRESETS["moe-tiny"]
    p = M.build_params(spec, generator=0)
    model = M.Model(spec, p, capacity_factor=1e9)  # no capacity drops
    toks = [11, 22, 33]
    logits, states = model.forward(toks)
    st = model.init_states()
    seq = torch.stack([model.step_token(t, st) for t in toks])
    assert (seq - logits).abs().max().item() <= 1e-5
    # MoE with the default capacity_factor drops tokens in a batched pass, so
    # batched and single-token forwards legitimately differ (documented).
    model_drop = M.Model(spec, M.build_params(spec, generator=0), capacity_factor=1.25)
    lg, _ = model_drop.forward(toks)
    assert model_drop._last_aux["overflow_dropped"] > 0
    assert (lg - logits).abs().max().item() > 1e-3


def test_mtp_step_matches_single_row_module():
    """Inference (mtp_step) and training (batched _mtp_module) agree at T=1."""
    spec, model = _dense_model(depth=3)
    toks = torch.tensor([11, 22, 33, 44])
    with torch.no_grad():
        _lg, _, det = model._forward_impl(toks[:-1], return_hidden=True)
    h = det["hidden"][0]
    states = [None] * 4
    for k in range(1, 4):
        lg1, Y1, s1 = model.mtp_step(k, h, int(toks[k]), states[k])
        with torch.no_grad():
            lg2, Y2, s2 = model._mtp_module(k, h.reshape(1, -1), torch.tensor([int(toks[k])]),
                                           state=states[k])
        assert torch.equal(lg1, lg2[0])
        assert torch.equal(Y1, Y2[0])
        assert torch.equal(s1, s2)
        h = Y1


# --------------------------------------------------------------------------
# claim 10: MTP row alignment
# --------------------------------------------------------------------------
def _mtp_alignment_check(T, depth):
    spec = cfg.PRESETS["dense-tiny"]
    model = M.Model(spec, M.build_params(spec, generator=torch.Generator().manual_seed(1),
                                         mtp_depth=depth), mtp_depth=depth)
    toks = torch.randint(0, spec.vocab_size, (T,))
    with torch.no_grad():
        _, _, det = model._forward_impl(toks[:-1], return_hidden=True)
    X = det["hidden"]
    calls = []
    real = model._mtp_module

    def recording(k, H, tok_in, state=None):
        out = real(k, H, tok_in, state)
        calls.append({"k": k, "H": H.clone(), "tok_in": tok_in.clone(),
                      "logits": out[0].clone(), "Y": out[1].clone()})
        return out

    model._mtp_module = recording
    with torch.no_grad():
        losses = [float(v.item()) for v in model._mtp_losses(X, toks)]
    return toks, calls, losses


@pytest.mark.parametrize("T,depth", [(6, 3), (4, 10), (3, 1), (2, 1), (8, 4)])
def test_mtp_row_alignment(T, depth):
    toks, calls, losses = _mtp_alignment_check(T, depth)
    expected_depths = [k for k in range(1, depth + 1) if T - 1 - k > 0]
    assert [c["k"] for c in calls] == expected_depths
    for i, c in enumerate(calls):
        k = c["k"]
        L = T - 1 - k
        assert c["H"].shape[0] == L
        # tok_in is the teacher-forced "next" token of this depth
        assert c["tok_in"].tolist() == toks[k:k + L].tolist()
        # the loss is CE(logits, tok[i+k+1]) and not the neighbouring shifts
        ce = float(F.cross_entropy(c["logits"], toks[k + 1:k + 1 + L]).item())
        assert abs(ce - losses[i]) <= 1e-6
        for s in (k, k + 2):
            if s != k + 1 and s + L <= T:
                other = float(F.cross_entropy(c["logits"], toks[s:s + L]).item())
                assert abs(other - losses[i]) > 1e-7


def test_mtp_hidden_chain():
    toks, calls, losses = _mtp_alignment_check(7, 4)
    spec = cfg.PRESETS["dense-tiny"]
    model = M.Model(spec, M.build_params(spec, generator=torch.Generator().manual_seed(1),
                                         mtp_depth=4), mtp_depth=4)
    with torch.no_grad():
        _, _, det = model._forward_impl(toks[:-1], return_hidden=True)
        X = det["hidden"]
    # recompute the chain explicitly and compare to the recorded inputs
    H = X
    for c in calls:
        k, L = c["k"], c["H"].shape[0]
        assert torch.allclose(c["H"], H[:L], atol=0)
        logits, Y, s = model._mtp_module(k, H[:L], toks[k:k + L])
        H = Y


# --------------------------------------------------------------------------
# claim 9: loss_and_backward gradients
# --------------------------------------------------------------------------
def _independent_dense_forward(p, spec, tokens):
    X = p["embed"][tokens[:-1]]
    for i in range(spec.n_layers):
        px = f"L{i}."
        h = X * torch.rsqrt(X.square().mean(-1, keepdim=True) + EPS) * p[px + "n1"]
        R = h @ p[px + "attn.Wr"]
        K = h @ p[px + "attn.Wk"]
        V = h @ p[px + "attn.Wv"]
        alpha = torch.exp(-F.softplus(p[px + "attn.w"]))
        kv = K * V
        st = torch.zeros_like(alpha)
        rows = []
        for t in range(h.shape[0]):
            st = alpha * st + kv[t]
            rows.append(st)
        S = torch.stack(rows)
        Y = torch.sigmoid(R) * (S + p[px + "attn.u"] * kv)
        X = X + Y @ p[px + "attn.Wo"]
        h2 = X * torch.rsqrt(X.square().mean(-1, keepdim=True) + EPS) * p[px + "n2"]
        A = h2 @ p[px + "ffn.Wa"]
        B = h2 @ p[px + "ffn.Wb"]
        X = X + (F.silu(A) * B) @ p[px + "ffn.Wc"]
    H = X * torch.rsqrt(X.square().mean(-1, keepdim=True) + EPS) * p["final_norm"]
    return F.cross_entropy(H @ p["embed"].T, tokens[1:])


def test_loss_and_backward_matches_independent_autograd():
    spec = cfg.PRESETS["dense-tiny"]
    model = M.Model(spec, M.build_params(spec, generator=torch.Generator().manual_seed(5)))
    torch.manual_seed(0)
    toks = torch.randint(0, spec.vocab_size, (6,))
    loss, grads, aux = model.loss_and_backward(toks)
    assert set(grads) == set(model.p)  # every param has a gradient
    assert all(torch.isfinite(g).all().item() for g in grads.values())
    assert all(g.shape == model.p[k].shape for k, g in grads.items())
    assert all(v.grad is None for v in model.params)  # zero_grad left nothing

    names = list(model.p)
    ref = _independent_dense_forward(model.p, spec, toks)
    gs = torch.autograd.grad(ref, [model.p[k] for k in names], allow_unused=False)
    assert abs(float(ref.detach()) - loss) <= 1e-6
    worst = 0.0
    for k, g in zip(names, gs):
        d = (grads[k] - g).abs().max().item()
        worst = max(worst, d / max(1e-9, g.abs().max().item()))
        assert d <= 1e-5 * max(1.0, g.abs().max().item()), (k, d)
    assert worst <= 1e-5


def test_loss_and_backward_finite_difference_on_final_norm():
    spec = cfg.PRESETS["dense-tiny"]
    model = M.Model(spec, M.build_params(spec, generator=torch.Generator().manual_seed(6)))
    torch.manual_seed(1)
    toks = torch.randint(0, spec.vocab_size, (5,))
    _loss, grads, _aux = model.loss_and_backward(toks)
    base = model.p["final_norm"].detach().clone()
    eps = 1e-3
    fd = torch.zeros_like(base)
    with torch.no_grad():
        for i in range(8):
            for sign, key in ((1, "p"), (-1, "m")):
                model.p["final_norm"][i] = base[i] + sign * eps
                lg, _ = model.forward(toks[:-1], train=True)
                val = F.cross_entropy(lg, toks[1:]).item()
                if sign == 1:
                    vp = val
                else:
                    vm = val
            model.p["final_norm"][i] = base[i]
            fd[i] = (vp - vm) / (2 * eps)
    assert (grads["final_norm"][:8] - fd[:8]).abs().max().item() <= 5e-3


def test_loss_and_backward_mtp_and_validation():
    spec = cfg.PRESETS["dense-tiny"]
    model = M.Model(spec, M.build_params(spec, generator=0, mtp_depth=2),
                    mtp_depth=2, mtp_loss_weight=0.1)
    toks = torch.randint(0, spec.vocab_size, (7,))
    loss, grads, aux = model.loss_and_backward(toks)
    assert len(aux["mtp_losses"]) == 2
    assert abs(loss - (aux["loss_main"] + 0.1 * sum(aux["mtp_losses"]))) <= 1e-4
    assert set(grads) == set(model.p)
    with pytest.raises(ValueError):
        model.loss_and_backward([1])


def test_grad_coverage_across_quant_combos():
    """loss_and_backward provides finite grads for every param in every
    supported quantization combination."""
    spec = cfg.PRESETS["dense-tiny"]
    base = M.build_params(spec, generator=torch.Generator().manual_seed(7))
    toks = torch.randint(0, spec.vocab_size, (5,))
    for fq, bm in [("fp32", "fp32"), ("fp8", "fp8"), ("ternary", "fp32"),
                   ("ternary", "ternary"), ("ternary", "fp8")]:
        model = M.Model(spec, {k: v.clone() for k, v in base.items()},
                        forward_quant=fq, backward_mode=bm)
        _loss, grads, _aux = model.loss_and_backward(toks)
        assert set(grads) == set(model.p), (fq, bm, set(model.p) - set(grads))
        assert all(torch.isfinite(g).all().item() for g in grads.values()), (fq, bm)
