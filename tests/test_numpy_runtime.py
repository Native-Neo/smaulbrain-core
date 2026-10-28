import os
import shutil
import time

import numpy as np
import pytest

from smaulbrain import checkpoint as ckpt
from smaulbrain import config as cfg
from smaulbrain import inference as inf
from smaulbrain import model as model
from smaulbrain import npgrad as ng
from smaulbrain import ops
from smaulbrain import quant
from smaulbrain import safetensors_np
from smaulbrain import smaulopt
from smaulbrain import _native_opt as native_opt


def test_npgrad_broadcast_and_dtype_cast():
    x = ng.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
    w = ng.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
    y = (x.float() * w).sum()
    y.backward()
    assert x.dtype == np.float32 and y.dtype == np.float32
    np.testing.assert_allclose(x.grad, np.broadcast_to(w.data, x.shape))
    np.testing.assert_allclose(w.grad, x.data.sum(axis=0))

    z = ng.Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), requires_grad=True)
    out = ng.Tensor(np.ones((2, 3), dtype=np.float32)) * z.float()
    out.sum().backward()
    np.testing.assert_allclose(z.grad, np.ones((2, 3)))


def test_index_add_rows_many_matches_sequential_forward_backward():
    rng = np.random.default_rng(26)
    ids_a, ids_b = np.array([0, 2, 3]), np.array([1, 2, 3])
    data_a = rng.standard_normal((3, 4)).astype(np.float32)
    data_b = rng.standard_normal((3, 4)).astype(np.float32)
    upstream = rng.standard_normal((5, 4)).astype(np.float32)

    def run(batched):
        base = ng.Tensor(np.zeros((5, 4), dtype=np.float32))
        a = ng.Tensor(data_a.copy(), requires_grad=True)
        b = ng.Tensor(data_b.copy(), requires_grad=True)
        if batched:
            out = ng.index_add_rows_many(base, [(ids_a, a), (ids_b, b)])
        else:
            out = ng.index_add_rows(ng.index_add_rows(base, ids_a, a), ids_b, b)
        return out, (a, b)

    fast, fast_sources = run(True)
    ref, ref_sources = run(False)
    np.testing.assert_array_equal(fast.data, ref.data)
    fast.backward(upstream)
    ref.backward(upstream)
    for got, want in zip(fast_sources, ref_sources):
        np.testing.assert_array_equal(got.grad, want.grad)


def test_scan_matches_explicit_recurrence():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((2, 5, 4)).astype(np.float32)
    alpha = rng.uniform(0.2, 0.9, size=4).astype(np.float32)
    got = ng.scan(ng.Tensor(values), ng.Tensor(alpha)).data
    state = np.zeros((2, 4), dtype=np.float32)
    rows = []
    for t in range(values.shape[1]):
        state = alpha * state + values[:, t]
        rows.append(state.copy())
    np.testing.assert_allclose(got, np.stack(rows, axis=1), rtol=1e-5, atol=1e-6)


def test_sigmoid_inplace_matches_allocating_reference():
    rng = np.random.default_rng(20)
    values = (rng.standard_normal((41, 29)) * 3.0).astype(np.float32)
    x = ng.Tensor(values, requires_grad=True)
    y = ng.sigmoid(x)
    y.backward(np.ones_like(y.data))

    x64 = values.astype(np.float64)
    want = (0.5 * (1.0 + np.tanh(0.5 * x64))).astype(np.float32)
    np.testing.assert_array_equal(y.data, want)
    np.testing.assert_array_equal(x.grad, want * (1.0 - want))


def test_fused_rmsnorm_matches_explicit_graph():
    rng = np.random.default_rng(21)
    gain_values = rng.standard_normal(32).astype(np.float32)
    for shape in ((33, 32), (3, 33, 32)):
        values = (rng.standard_normal(shape) * 2.0).astype(np.float32)
        upstream = (rng.standard_normal(shape) * 0.5).astype(np.float32)

        x_graph = ng.Tensor(values.copy(), requires_grad=True)
        g_graph = ng.Tensor(gain_values.copy(), requires_grad=True)
        y_graph = (x_graph * ng.rsqrt(
            x_graph.square().mean(dim=-1, keepdim=True) + ops.EPS) * g_graph)
        y_graph.backward(upstream)

        x_fused = ng.Tensor(values.copy(), requires_grad=True)
        g_fused = ng.Tensor(gain_values.copy(), requires_grad=True)
        y_fused = ng.rmsnorm(x_fused, g_fused, ops.EPS)
        y_fused.backward(upstream)

        np.testing.assert_array_equal(y_fused.data, y_graph.data)
        np.testing.assert_array_equal(x_fused.grad, x_graph.grad)
        np.testing.assert_array_equal(g_fused.grad, g_graph.grad)


def test_native_scan_fast_path_matches_closed_form(monkeypatch):
    """The 2-D training fast path must agree with the chunked reference."""
    if not quant.prepare_ternary_backend():
        pytest.skip("native C++ ternary backend is unavailable")
    rng = np.random.default_rng(19)
    values = rng.standard_normal((257, 32)).astype(np.float32)
    alpha = rng.uniform(0.2, 0.9, size=32).astype(np.float32)
    upstream = rng.standard_normal(values.shape).astype(np.float32)

    x_fast = ng.Tensor(values, requires_grad=True)
    a_fast = ng.Tensor(alpha, requires_grad=True)
    y_fast = ng.scan(x_fast, a_fast)
    y_fast.backward(upstream)

    native_lib = ng._native_scan._NATIVE_LIB
    monkeypatch.setattr(ng._native_scan, "_NATIVE_LIB", None)
    x_ref = ng.Tensor(values, requires_grad=True)
    a_ref = ng.Tensor(alpha, requires_grad=True)
    y_ref = ng.scan(x_ref, a_ref)
    y_ref.backward(upstream)
    monkeypatch.setattr(ng._native_scan, "_NATIVE_LIB", native_lib)

    np.testing.assert_allclose(y_fast.data, y_ref.data, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(x_fast.grad, x_ref.grad, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(a_fast.grad, a_ref.grad, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("shape,batch", [
    ((32, 32), 512),
    ((32, 128), 512),
    ((128, 32), 512),
    ((7, 33), 65),
])
def test_tiled_packed_parallel_blocks_match_reference(shape, batch):
    """Batch/row work partitioning must not change packed matmul results."""
    if not quant.prepare_ternary_backend():
        pytest.skip("native C++ ternary backend is unavailable")
    din, dout = shape
    rng = np.random.default_rng(din * 1000 + dout + batch)
    W = (rng.standard_normal(shape) * 0.3).astype(np.float32)
    packed, scale, packed_shape, _ = quant.pack_ternary_tiled(W)
    dense = quant.unpack_ternary(packed, packed_shape, scale)
    X = rng.standard_normal((batch, din)).astype(np.float32)
    dY = rng.standard_normal((batch, dout)).astype(np.float32)
    np.testing.assert_allclose(
        quant.packed_matmat(X, packed, packed_shape, scale),
        X @ dense, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        quant.packed_matmat_transpose(dY, packed, packed_shape, scale),
        dY @ dense.T, rtol=2e-5, atol=2e-5)


def test_pack_ternary_tiled_many_matches_individual():
    if not quant.prepare_ternary_backend():
        pytest.skip("native C++ ternary backend is unavailable")
    rng = np.random.default_rng(91)
    shapes = [(32, 32), (32, 128), (128, 32), (7, 33)]
    weights = [(rng.standard_normal(shape) * 0.25).astype(np.float32)
               for shape in shapes]
    single = [quant.pack_ternary_tiled(W) for W in weights]
    batched = quant.pack_ternary_tiled_many(weights)
    assert len(batched) == len(single)
    for (packed_a, scale_a, shape_a, th_a), (packed_b, scale_b, shape_b, th_b) in zip(single, batched):
        assert shape_a == shape_b
        assert scale_a == scale_b
        assert th_a == th_b
        np.testing.assert_array_equal(quant.unpack_codes(packed_a, shape_a),
                                      quant.unpack_codes(packed_b, shape_b))
    # The batch shares one allocation but each matrix owns a disjoint slice.
    assert batched[0][0].buf.base is batched[1][0].buf.base
    assert not np.shares_memory(batched[0][0].buf, batched[1][0].buf)


def test_packed_cache_prefetch_repacks_invalidated_weights():
    rng = np.random.default_rng(92)
    params = {
        "L0.attn.Wr": ng.Parameter((rng.standard_normal((32, 32)) * 0.2).astype(np.float32)),
        "expert.0.Win": ng.Parameter((rng.standard_normal((32, 128)) * 0.2).astype(np.float32)),
    }
    cache = model.PackedCache()
    for key, W in params.items():
        cache.get(key, W)
    assert cache.builds == 2 and cache.stale == {}
    cache.invalidate(params.keys())
    assert cache.d == {} and set(cache.stale) == set(params)
    for W in params.values():
        W.data[...] += 0.05
    # Dynamic expert keys are left to lazy packing; static keys are batched.
    cache.prefetch(exclude_prefixes=("expert.",))
    assert cache.stale == {}
    assert "L0.attn.Wr" in cache.d and "expert.0.Win" not in cache.d
    for key, W in params.items():
        packed, scale, shape = cache.get(key, W)
        ref_packed, ref_scale, ref_shape, _ = quant.pack_ternary_tiled(W.data)
        assert shape == ref_shape and scale == ref_scale
        np.testing.assert_array_equal(quant.unpack_codes(packed, shape),
                                      quant.unpack_codes(ref_packed, ref_shape))
    assert cache.builds == 4
    assert cache.stale == {}


def test_safetensors_and_quant_roundtrip(tmp_path):
    arrays = {
        "f32": np.arange(6, dtype=np.float32).reshape(2, 3),
        "bool": np.array([True, False, True]),
        "u8": np.arange(4, dtype=np.uint8),
    }
    path = tmp_path / "t.safetensors"
    safetensors_np.save_file(arrays, path, metadata={"kind": "test"})
    loaded, metadata = safetensors_np.load_file(path)
    assert metadata["kind"] == "test"
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded[key], value)

    rng = np.random.default_rng(1)
    W = rng.standard_normal((8, 6)).astype(np.float32)
    packed, scale, shape, _ = quant.pack_ternary(W)
    dense = quant.unpack_ternary(packed, shape, scale)
    assert dense.shape == W.shape
    x = rng.standard_normal(8).astype(np.float32)
    y = quant.packed_matmat(x, packed, shape, scale)
    np.testing.assert_allclose(y, x @ dense, rtol=1e-5, atol=1e-5)
    q, qscale = quant.fp8_e4m3_quantize(W)
    assert q.shape == W.shape and np.isfinite(qscale)


def test_safetensors_zero_dim_roundtrip(tmp_path):
    path = tmp_path / "scalars.safetensors"
    arrays = {
        "scale": np.asarray(2.25, dtype=np.float32),
        "count": np.zeros((), dtype=np.int64),
    }
    safetensors_np.save_file(arrays, path, metadata={"kind": "test"})
    loaded, metadata = safetensors_np.load_file(path)
    assert metadata.get("kind") == "test"
    assert "__np_zero_dim__" not in metadata
    for name, value in arrays.items():
        assert loaded[name].shape == ()
        np.testing.assert_array_equal(loaded[name], value)

    # Genuine (1,)-shaped tensors are not confused with the scalar marker and
    # stay 1-dimensional on the roundtrip.
    plain = tmp_path / "plain.safetensors"
    safetensors_np.save_file({"plain": np.zeros((1,), np.float32)}, plain)
    assert safetensors_np.load_file(plain)[0]["plain"].shape == (1,)


def test_batched_native_optimizer_matches_single_matrix_path(monkeypatch):
    rng = np.random.default_rng(23)
    shapes = [(32, 32), (33, 64), (128, 32), (32,)]
    base = {f"w{i}": rng.standard_normal(shape).astype(np.float32)
            for i, shape in enumerate(shapes)}
    grads = {name: rng.standard_normal(value.shape).astype(np.float32)
             for name, value in base.items()}

    def make():
        params = {name: value.copy() for name, value in base.items()}
        opt = smaulopt.SmaulOPT(lr=3e-4, weight_decay=0.01)
        opt.register(params)
        return opt, params

    batch_opt, batch_params = make()
    single_opt, single_params = make()
    batch_norms = batch_opt.step(batch_params, grads, collect_norms=True)
    monkeypatch.setattr(native_opt, "factored_step_many", lambda *a, **k: None)
    single_norms = single_opt.step(single_params, grads, collect_norms=True)

    for name in base:
        np.testing.assert_array_equal(batch_params[name], single_params[name])
        np.testing.assert_array_equal(batch_opt.state[name]["m"],
                                      single_opt.state[name]["m"])
        assert batch_norms[name] == single_norms[name]


def test_smaulopt_rejects_unknown_gradient_without_updating():
    rng = np.random.default_rng(24)
    w = rng.standard_normal((6,)).astype(np.float32)
    before = w.copy()
    opt = smaulopt.SmaulOPT(lr=1e-3)
    opt.register({"w": w})
    with pytest.raises(ValueError, match="unknown SmaulOPT parameter"):
        opt.step({"w": w}, {"w": np.ones_like(w), "missing": np.ones(3)})
    np.testing.assert_array_equal(w, before)
    assert opt.n_updates == 0
    np.testing.assert_array_equal(opt.state["w"]["m"], np.zeros_like(w))


def test_smaulopt_step_is_atomic_on_invalid_inputs():
    rng = np.random.default_rng(25)
    params = {
        "a": rng.standard_normal((7,)).astype(np.float32),
        "b": rng.standard_normal((6, 8)).astype(np.float32),
    }
    before = {name: value.copy() for name, value in params.items()}
    grads = {
        "a": np.ones_like(params["a"]),
        "b": rng.standard_normal((5, 8)).astype(np.float32),
    }
    opt = smaulopt.SmaulOPT(lr=1e-3)
    opt.register(params)
    with pytest.raises(ValueError, match="shape"):
        opt.step(params, grads)
    for name, value in params.items():
        np.testing.assert_array_equal(value, before[name])
    assert opt.n_updates == 0
    for state in opt.state.values():
        np.testing.assert_array_equal(state["m"], np.zeros_like(state["m"]))

    grads["b"] = np.ones_like(params["b"])
    grads["a"][0] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        opt.step(params, grads)
    for name, value in params.items():
        np.testing.assert_array_equal(value, before[name])
    assert opt.n_updates == 0

    grads["a"][0] = 1.0
    opt.state["b"]["R"] = opt.state["b"]["R"][:-1]
    with pytest.raises(ValueError, match="state"):
        opt.step(params, grads)
    for name, value in params.items():
        np.testing.assert_array_equal(value, before[name])
    assert opt.n_updates == 0


def test_smaulopt_load_state_dict_validates_layout_atomically():
    opt = smaulopt.SmaulOPT()
    good = {"w": {"m": np.zeros((2, 3), np.float32),
                  "R": np.zeros(2, np.float32),
                  "C": np.zeros(3, np.float32)}}
    opt.load_state_dict(good)
    assert opt.state["w"]["factored"] is True

    opt = smaulopt.SmaulOPT()
    with pytest.raises(ValueError, match="both 'R' and 'C'"):
        opt.load_state_dict({"w": {"m": np.zeros((2, 3), np.float32),
                                   "R": np.zeros(2, np.float32)}})
    assert opt.state == {}

    with pytest.raises(ValueError, match="factored_flags"):
        opt.load_state_dict(good, {"w": False})
    assert opt.state == {}


@pytest.mark.parametrize("preset", ["dense-tiny", "moe-tiny"])
def test_model_forward_backward(preset):
    spec = cfg.PRESETS[preset]
    m = model.Model(spec, model.build_params(spec, np.random.default_rng(2)))
    toks = np.random.default_rng(3).integers(0, 256, (8,)).tolist()
    logits, states = m.forward(toks)
    assert logits.shape == (8, spec.vocab_size)
    assert len(states) == spec.n_layers
    loss, grads, _ = m.loss_and_backward(toks)
    assert np.isfinite(loss)
    assert grads and all(np.isfinite(g).all() for g in grads.values())
    opt = smaulopt.SmaulOPT(lr=1e-3)
    opt.register(m.p)
    opt.step(m.p, grads)
    assert opt.n_updates == 1


def test_loss_backward_returns_gradient_buffers_without_copying(monkeypatch):
    spec = cfg.PRESETS["dense-tiny"]
    m = model.Model(spec, model.build_params(spec, np.random.default_rng(27)))
    original_zero_grad = m.zero_grad
    captured = {}

    def capture_before_clear(*args, **kwargs):
        current = {k: v.grad for k, v in m.p.items() if v.grad is not None}
        if current:
            captured.update(current)
        return original_zero_grad(*args, **kwargs)

    monkeypatch.setattr(m, "zero_grad", capture_before_clear)
    loss, grads, _ = m.loss_and_backward(
        np.random.default_rng(28).integers(0, 256, 8).tolist())
    assert np.isfinite(loss)
    assert grads and grads.keys() == captured.keys()
    assert all(grads[name] is captured[name] for name in grads)
    assert all(parameter.grad is None for parameter in m.p.values())
    assert all(np.isfinite(grad).all() for grad in grads.values())


def test_moe_reduced_precision_keeps_sequential_aggregation(monkeypatch):
    spec = cfg.PRESETS["moe-tiny"]
    m = model.Model(spec, model.build_params(spec, np.random.default_rng(29)),
                    activation_storage="bf16")

    def unexpected_batch(*args, **kwargs):
        pytest.fail("reduced-precision MoE must preserve sequential rounding")

    monkeypatch.setattr(ng, "index_add_rows_many", unexpected_batch)
    logits, _ = m.forward(
        np.random.default_rng(30).integers(0, 256, 8).tolist(), train=True)
    assert np.isfinite(logits.data).all()


def test_ternary_training_prefetches_invalidated_weights():
    spec = cfg.PRESETS["dense-tiny"]
    m = model.Model(spec, model.build_params(spec, np.random.default_rng(93)),
                    forward_quant="ternary", backward_mode="ternary")
    toks = np.random.default_rng(94).integers(0, 256, 8).tolist()
    loss, grads, _ = m.loss_and_backward(toks)
    assert np.isfinite(loss)
    opt = smaulopt.SmaulOPT(lr=1e-3)
    opt.register(m.p)
    opt.step(m.p, grads, collect_norms=False)
    m.packed.invalidate(grads.keys())
    stale = set(m.packed.stale)
    assert stale and not m.packed.d
    builds = m.packed.builds
    loss, grads, _ = m.loss_and_backward(toks)
    assert np.isfinite(loss)
    assert not m.packed.stale
    assert stale <= set(m.packed.d)
    assert m.packed.builds == builds + len(stale)


def test_checkpoint_and_inference_roundtrip(tmp_path):
    spec = cfg.PRESETS["dense-tiny"]
    m = model.Model(spec, model.build_params(spec, np.random.default_rng(4)))
    opt = smaulopt.SmaulOPT()
    opt.register(m.p)
    man = ckpt.build_manifest(
        "dense-tiny", {}, {"seq_len": 8},
        ckpt.blank_scheduler_state(), 0,
        {"bit_generator": np.random.default_rng(0).bit_generator.state},
        0, [], {},
    )
    ckpt.save_checkpoint(str(tmp_path / "run"), man, m.p, opt.state_dict())
    blob = ckpt.load_checkpoint(str(tmp_path / "run"))
    assert blob["manifest"]["preset"] == "dense-tiny"
    m2 = model.Model(spec, blob["params"])
    toks = [65, 66, 67, 68]
    a, _ = m.forward(toks)
    b, _ = m2.forward(toks)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    eng = inf.InferenceEngine(m2, seed=0, temperature=0.0)
    eng.prefill(toks)
    first = eng.generate([], max_new_tokens=5)
    eng.reset()
    eng.prefill(toks)
    second = eng.generate([], max_new_tokens=5)
    assert first == second


def test_sweep_stale_checkpoints_reaps_crash_debris(tmp_path):
    """A killed save must not leave debris, and survivors must stay intact."""
    spec = cfg.PRESETS["dense-tiny"]
    m = model.Model(spec, model.build_params(spec, np.random.default_rng(4)))
    opt = smaulopt.SmaulOPT()
    opt.register(m.p)
    man = ckpt.build_manifest(
        "dense-tiny", {}, {"seq_len": 8},
        ckpt.blank_scheduler_state(), 0,
        {"bit_generator": np.random.default_rng(0).bit_generator.state},
        0, [], {},
    )
    ckpt.save_checkpoint(str(tmp_path / "run"), man, m.p, opt.state_dict())

    # Stale publication debris an unlucky SIGKILL would leave behind.
    orphan = tmp_path / (ckpt._CKPT_TMP_PREFIX + "stale")
    orphan.mkdir()
    (orphan / "junk.bin").write_bytes(b"x")
    partial_bak = tmp_path / "run.bak"
    partial_bak.mkdir()
    (partial_bak / "half.safetensors").write_bytes(b"x")
    ancient = time.time() - 3600.0
    for stale in (orphan, partial_bak):
        os.utime(stale, (ancient, ancient))

    # A fresh .ckpt.* tree represents a plausible concurrent writer and must
    # survive the grace window.
    fresh_tmp = tmp_path / (ckpt._CKPT_TMP_PREFIX + "fresh")
    fresh_tmp.mkdir()

    ckpt.save_checkpoint(str(tmp_path / "run"), man, m.p, opt.state_dict())

    assert not orphan.exists()
    assert not partial_bak.exists()
    assert fresh_tmp.exists()
    assert ckpt.load_checkpoint(str(tmp_path / "run"))["manifest"][
        "preset"] == "dense-tiny"

    # The public helper reaps cold .ckpt.* debris without a save, leaving
    # fresh temporaries alone.
    orphan2 = tmp_path / (ckpt._CKPT_TMP_PREFIX + "stale2")
    orphan2.mkdir()
    os.utime(orphan2, (ancient, ancient))
    removed = ckpt.sweep_stale_checkpoints(str(tmp_path / "run"), grace=10.0)
    assert not orphan2.exists()
    assert fresh_tmp.exists()
    assert any(p.endswith("stale2") for p in removed)

    # Crash mid-publish: the renamed old tree (with a manifest) is the only
    # complete checkpoint while the target itself is incomplete.  The sweep
    # must preserve it, and the loader must be able to use it.
    run = tmp_path / "run"
    shutil.copytree(run, run.parent / "run.bak")
    (run / "manifest.json").unlink()
    for path in (run, run.parent / "run.bak"):
        os.utime(path, (ancient, ancient))
    removed = ckpt.sweep_stale_checkpoints(str(run), grace=10.0)
    assert (run.parent / "run.bak").exists()
    shutil.rmtree(run)
    os.rename(run.parent / "run.bak", run)
    assert ckpt.load_checkpoint(str(run))["manifest"]["preset"] == "dense-tiny"
