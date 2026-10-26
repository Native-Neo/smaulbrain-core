import numpy as np
import pytest

from smaulbrain import checkpoint as ckpt
from smaulbrain import config as cfg
from smaulbrain import inference as inf
from smaulbrain import model as model
from smaulbrain import npgrad as ng
from smaulbrain import quant
from smaulbrain import safetensors_np
from smaulbrain import smaulopt


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
