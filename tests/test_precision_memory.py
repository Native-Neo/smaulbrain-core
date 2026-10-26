import gc
import weakref

import numpy as np
import pytest

from smaulbrain import checkpoint as ckpt
from smaulbrain import config as cfg
from smaulbrain import model as M
from smaulbrain import npgrad as ng
from smaulbrain import precision as P
from smaulbrain import quant
from smaulbrain import smaulopt


# ---------------------------------------------------------------------------
# Codecs and storage accounting
# ---------------------------------------------------------------------------
def test_bf16_roundtrip_and_size():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((17, 9)) * 10.0 ** rng.uniform(-8, 8, (17, 9))).astype(np.float32)
    bits = P.bf16_encode(x)
    assert bits.dtype == np.uint16
    assert bits.nbytes == x.size * 2
    decoded = P.bf16_decode(bits)
    assert decoded.dtype == np.float32 and decoded.shape == x.shape
    # bfloat16 keeps 8 exponent and 7 mantissa bits; relative error <= 2^-8.
    np.testing.assert_allclose(decoded, x, rtol=2.0 ** -8, atol=0.0)
    assert np.array_equal(P.bf16_encode(np.array([0.0, -0.0], np.float32)),
                          np.array([0x0000, 0x8000], np.uint16))


def test_bf16_round_to_nearest_even_and_specials():
    # 1.00390625 is exactly halfway between 1.0 and 1.0078125; the even
    # significand is 1.0.  1.01171875 is halfway between the next pair and
    # rounds to 1.015625 (even low bit).
    x = np.array([1.00390625, 1.01171875, np.inf, -np.inf, np.nan], np.float32)
    got = P.bf16_roundtrip(x)
    assert got[0] == np.float32(1.0)
    assert got[1] == np.float32(1.015625)
    assert np.isposinf(got[2]) and np.isneginf(got[3]) and np.isnan(got[4])


def _bf16_reference(a):
    raw = np.ascontiguousarray(a, np.float32).view(np.uint32)
    rounded = raw + np.uint32(0x7FFF) + ((raw >> np.uint32(16)) & np.uint32(1))
    high = (rounded >> np.uint32(16)).astype(np.uint16)
    special = (raw & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    high[special] = (raw[special] >> np.uint32(16)).astype(np.uint16)
    return high


def test_bf16_codec_matches_independent_bit_reference():
    rng = np.random.default_rng(21)
    bits = rng.integers(0, 2 ** 32, 20000, dtype=np.uint64).astype(np.uint32)
    x = bits.view(np.float32)

    encoded = P.bf16_encode(x)
    np.testing.assert_array_equal(encoded, _bf16_reference(x))
    decoded_bits = P.bf16_decode(encoded).view(np.uint32)
    np.testing.assert_array_equal(decoded_bits, encoded.astype(np.uint32) << 16)


def test_bf16_large_finite_path_matches_reference_including_ties():
    # The encoder switches to a wider-array finite path above 128 KiB.  Its
    # flat tie handling must still produce exactly the reference RNE bits.
    rng = np.random.default_rng(22)
    base = rng.standard_normal((300, 128)).astype(np.float32)
    patterns = np.array([0x0000, 0x7FFF, 0x8000, 0x8001, 0xFFFF], np.uint32)
    raw = base.view(np.uint32)
    raw[...] = (raw & np.uint32(0xFFFF0000)) | patterns[
        np.arange(raw.size).reshape(raw.shape) % patterns.size]
    x = raw.view(np.float32)
    assert x.nbytes > 128 * 1024
    np.testing.assert_array_equal(P.bf16_encode(x), _bf16_reference(x))


def test_int8_quantization_zero_clip_and_noncontiguous():
    zeros = np.zeros((4, 3), np.float32)
    codes, scale = P.int8_quantize(zeros)
    assert codes.dtype == np.int8 and scale > 0
    np.testing.assert_array_equal(P.int8_dequantize(codes, scale), zeros)

    extreme = np.array([3.0e38, -3.0e38, 0.0, 1.0e-30], np.float32).reshape(2, 2)
    codes, scale = P.int8_quantize(extreme)
    assert np.isfinite(scale) and codes.min() >= -127 and codes.max() <= 127
    decoded = P.int8_dequantize(codes, scale)
    assert np.isfinite(decoded).all()
    # Rounding error is at most half a scale step.
    np.testing.assert_allclose(decoded, extreme, atol=float(scale) * 0.5 + 1e-30)

    base = np.arange(48, dtype=np.float32).reshape(8, 6) * 7.0 - 100.0
    view = base[:, ::2]
    assert not view.flags.c_contiguous
    codes, scale = P.int8_quantize(view)
    decoded = P.int8_dequantize(codes, scale)
    np.testing.assert_allclose(decoded, view, atol=float(scale) * 0.5 + 1e-6)

    with pytest.raises(ValueError):
        P.int8_quantize(np.array([np.nan], np.float32))
    with pytest.raises(ValueError):
        P.int8_quantize(np.array([np.inf], np.float32), scale=1.0)


def test_int8_subnormal_scale_stays_positive():
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    values = np.array([tiny, -tiny, 5.0 * tiny], np.float32)
    codes, scale = P.int8_quantize(values)
    assert scale > 0.0 and np.isfinite(scale)
    decoded = P.int8_dequantize(codes, scale)
    np.testing.assert_allclose(decoded, values, rtol=0.0, atol=float(scale) * 0.5 + tiny)


def test_int8_computed_scale_keeps_codes_in_range():
    # Computed scales avoid the extra clipping pass, so verify the range
    # guarantee directly across magnitudes and a broad dynamic range.
    rng = np.random.default_rng(17)
    values = (rng.standard_normal((64, 64)) *
              10.0 ** rng.uniform(-30.0, 30.0, (64, 64))).astype(np.float32)
    values[0, :4] = [0.0, np.float32(-0.0),
                     np.nextafter(np.float32(0.0), np.float32(1.0)),
                     -np.nextafter(np.float32(0.0), np.float32(1.0))]
    codes, scale = P.int8_quantize(values)
    assert codes.dtype == np.int8 and codes.min() >= -127 and codes.max() <= 127
    decoded = P.int8_dequantize(codes, scale)
    assert np.isfinite(decoded).all()
    np.testing.assert_allclose(decoded, values, rtol=0.0,
                               atol=float(scale) * 0.5 + 1e-30)


def test_int8_explicit_scale_clips_to_signed_range():
    values = np.array([-1e6, -127.4, -1.0, 0.0, 1.0, 127.6, 1e6], np.float32)
    codes, scale = P.int8_quantize(values, scale=1.0)
    assert scale == np.float32(1.0)
    np.testing.assert_array_equal(codes, np.array([-127, -127, -1, 0, 1, 127, 127],
                                                  np.int8))


@pytest.mark.parametrize("mode", ["bf16", "int8"])
def test_stored_array_accepts_noncontiguous_input(mode):
    base = (np.arange(32 * 4, dtype=np.float32).reshape(32, 4) - 50.0) * 3.25
    view = base[::2, ::-1]
    assert not view.flags.c_contiguous
    stored = P.StoredArray.compress(view, mode)
    assert stored.shape == view.shape
    decoded = stored.dequantize()
    assert decoded.shape == view.shape and decoded.dtype == np.float32
    assert decoded.flags.c_contiguous
    tolerance = (float(np.max(np.abs(view))) * 2.0 ** -8 if mode == "bf16"
                 else float(P.int8_scale(view)) * 0.5 + 1e-6)
    np.testing.assert_allclose(decoded, view, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("mode", ["fp32", "bf16", "int8"])
def test_tensor_storage_dtype_and_bytes(mode):
    rng = np.random.default_rng(11)
    x = ng.Tensor(rng.standard_normal((64, 64)).astype(np.float32), True)
    with P.activation_storage(mode, min_bytes=0):
        y = x * np.float32(2.0)
    logical = y.data.nbytes
    expected = x.data * np.float32(2.0)
    if mode == "fp32":
        assert y.storage_dtype == "fp32"
        assert y.storage_nbytes == logical
        tolerance = 0.0
    elif mode == "bf16":
        assert y.storage_dtype == "bf16"
        assert y._stored.codes.dtype == np.uint16
        assert y.storage_nbytes == logical // 2
        tolerance = float(np.max(expected)) * 2.0 ** -8
    else:
        assert y.storage_dtype == "int8"
        assert y._stored.codes.dtype == np.int8
        assert y.storage_nbytes <= logical // 4 + 4
        tolerance = float(P.int8_scale(expected)) * 0.5 + 1e-7
    assert y.data.shape == (64, 64) and y.data.dtype == np.float32
    np.testing.assert_allclose(y.data, expected, rtol=0.0, atol=tolerance)


def test_model_activation_graph_actually_shrinks():
    spec = cfg.PRESETS["dense-tiny"]
    params = M.build_params(spec, np.random.default_rng(1))
    m = M.Model(spec, params, activation_storage="fp32")
    toks = np.random.default_rng(2).integers(0, 256, (64,)).astype(np.int64)

    def graph_bytes(mode):
        with ng.enable_grad(), P.activation_storage(mode):
            logits, _, _ = m._forward_impl(toks[:-1], return_hidden=True)
            loss = ng.cross_entropy(logits, toks[1:])
        seen, stack = set(), [loss]
        total = 0
        formats = {}
        while stack:
            t = stack.pop()
            if id(t) in seen:
                continue
            seen.add(id(t))
            if not isinstance(t, ng.Parameter):
                total += t.storage_nbytes
                formats[t.storage_dtype] = formats.get(t.storage_dtype, 0) + 1
            stack.extend(t._parents)
        return total, formats

    fp32_bytes, _ = graph_bytes("fp32")
    bf16_bytes, bf16_formats = graph_bytes("bf16")
    int8_bytes, int8_formats = graph_bytes("int8")
    assert bf16_formats.get("bf16", 0) > 0
    assert int8_formats.get("int8", 0) > 0
    assert bf16_bytes < fp32_bytes * 0.8
    assert int8_bytes < fp32_bytes * 0.6
    assert int8_bytes < bf16_bytes


# ---------------------------------------------------------------------------
# Forward/backward and optimizer behavior
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("preset", ["dense-tiny", "moe-tiny"])
@pytest.mark.parametrize("activation_storage,gradient_storage", [
    ("fp32", "fp32"),
    ("bf16", "fp32"),
    ("int8", "fp32"),
    ("bf16", "bf16"),
])
def test_forward_backward_all_storage_modes(preset, activation_storage, gradient_storage):
    spec = cfg.PRESETS[preset]
    params = M.build_params(spec, np.random.default_rng(3), spec.n_experts_init)
    m = M.Model(spec, params, activation_storage=activation_storage,
                gradient_storage=gradient_storage)
    toks = np.random.default_rng(4).integers(0, 256, (32,)).astype(np.int64)
    loss, grads, info = m.loss_and_backward(toks)
    assert np.isfinite(loss) and loss > 0
    assert info["loss_main"] > 0
    assert grads
    for name, g in grads.items():
        dense = P.gradient_array(g)
        assert dense.shape == tuple(m.p[name].shape)
        assert np.isfinite(dense).all()
        if gradient_storage == "bf16":
            assert isinstance(g, P.PackedGradient)
            assert g.nbytes == dense.size * 2
        else:
            assert isinstance(g, np.ndarray)
    if gradient_storage == "bf16":
        # The model clears parameter.grad after returning the copied/packed
        # dict, so reattach for the standalone clipping API.
        for name, g in grads.items():
            m.p[name].grad = g
        norm = ng.clip_grad_norm_(list(m.p.values()), 1.0)
        assert np.isfinite(norm) and norm > 0
    opt = smaulopt.SmaulOPT(lr=1e-3)
    opt.register(m.p)
    opt.step(m.p, grads, collect_norms=False)
    assert opt.n_updates == 1
    assert all(np.isfinite(v.data).all() for v in m.p.values())


def test_packed_gradient_zero_and_clip():
    p = ng.Parameter(np.array([3.0, -4.0], np.float32))
    p.grad = P.PackedGradient.compress(np.array([3.0, -4.0], np.float32))
    norm = ng.clip_grad_norm_([p], 1.0)
    np.testing.assert_allclose(norm, 5.0, rtol=1e-6)
    assert isinstance(p.grad, P.PackedGradient)
    np.testing.assert_allclose(np.asarray(p.grad), np.array([0.6, -0.8], np.float32),
                               rtol=2.0 ** -8, atol=1e-6)
    module = ng.Module()
    module.register_parameter("p", p)
    module.zero_grad(set_to_none=False)
    assert isinstance(p.grad, P.PackedGradient)
    assert np.all(np.asarray(p.grad) == 0)


def test_bf16_gradient_accumulates_in_fp32_once(monkeypatch):
    x = ng.Parameter(np.array([1.0, 2.0, 3.0], np.float32))
    calls = []
    original = P.PackedGradient.compress

    def counted(cls, array, encoder=None):
        calls.append(np.asarray(array).copy())
        return original(array, encoder)

    monkeypatch.setattr(P.PackedGradient, "compress", classmethod(counted))
    with P.gradient_storage("bf16"):
        loss = (x * x).sum() + (np.float32(3.0) * x).sum()
        loss.backward(release_intermediate=True)
    assert len(calls) == 1, "leaf gradient must be encoded once, not per contribution"
    np.testing.assert_allclose(calls[0], 2.0 * x.data + 3.0)
    assert isinstance(x.grad, P.PackedGradient)
    np.testing.assert_allclose(np.asarray(x.grad), 2.0 * x.data + 3.0,
                               rtol=2.0 ** -8, atol=1e-6)
    assert x.grad.nbytes == x.data.size * 2


def test_intermediate_grad_release_does_not_drop_leaf_gradients():
    x = ng.Tensor(np.array([1.0, 2.0], np.float32), requires_grad=True)
    w = ng.Parameter(np.array([0.5, -0.5], np.float32))
    y = (x * w).sum()
    y.backward(release_intermediate=True)
    assert x.grad is not None and w.grad is not None
    np.testing.assert_allclose(x.grad, w.data)


def test_backward_graph_is_freed_without_gc():
    """The reverse traversal must not leave the graph in a reference cycle."""
    gc.disable()

    def run():
        x = ng.Tensor(np.array([1.0, -2.0], np.float32), requires_grad=True)
        y = (x * x).sum()
        peek = weakref.ref(y)
        y.backward()
        return peek, x

    try:
        peek, x = run()
        assert peek() is None
        np.testing.assert_allclose(x.grad, 2.0 * x.data)
    finally:
        gc.enable()


def test_bf16_gradient_accumulates_across_backward_calls():
    p = ng.Parameter(np.array([1.0, -2.0, 3.0], np.float32))
    with P.gradient_storage("bf16"):
        (p * p).sum().backward()
        (np.float32(3.0) * p).sum().backward()
    assert isinstance(p.grad, P.PackedGradient)
    np.testing.assert_allclose(np.asarray(p.grad), 2.0 * p.data + 3.0,
                               rtol=2.0 ** -8, atol=1e-6)


@pytest.mark.parametrize("mode", ["bf16", "int8"])
def test_quantized_activation_loss_close_to_fp32(mode):
    """Storage quantization must not move the forward loss materially."""
    spec = cfg.PRESETS["dense-tiny"]
    params = M.build_params(spec, np.random.default_rng(15))
    toks = np.random.default_rng(16).integers(0, 256, (64,)).astype(np.int64)

    def loss_for(storage):
        m = M.Model(spec, params, activation_storage=storage)
        logits, _ = m.forward(toks[:-1], train=True)
        return float(ng.cross_entropy(logits, toks[1:]).data)

    reference = loss_for("fp32")
    quantized = loss_for(mode)
    assert abs(quantized - reference) <= 0.02 * abs(reference) + 0.05


@pytest.mark.parametrize("mode", ["fp32", "bf16", "int8"])
def test_silu_inplace_matches_allocating_reference(mode):
    rng = np.random.default_rng(17)
    values = (rng.standard_normal((41, 29)) * 3.0).astype(np.float32)
    x = ng.Tensor(values, requires_grad=True)
    with P.activation_storage(mode):
        y = ng.silu(x)
    y.backward(np.ones_like(y.data))
    if mode == "fp32":
        x64 = values.astype(np.float64)
        sig = 1.0 / (1.0 + np.exp(-x64))
        want = (x64 * sig).astype(np.float32)
        grad = (sig * (1.0 + x64 * (1.0 - sig))).astype(np.float32)
    else:
        sig = 1.0 / (1.0 + np.exp(-values))
        want = (values * sig).astype(np.float32)
        grad = (sig * (1.0 + values * (1.0 - sig))).astype(np.float32)
        want = P.StoredArray.compress(want, mode).dequantize()
    np.testing.assert_array_equal(y.data, want)
    np.testing.assert_array_equal(x.grad, grad)


@pytest.mark.parametrize("op", ["matmul", "silu", "gelu", "softplus"])
def test_op_closures_do_not_retain_full_size_float32_copies(op):
    """Compressed ops must decode operands in backward instead of capturing them."""
    rng = np.random.default_rng(12)
    x = ng.Tensor(rng.standard_normal((48, 48)).astype(np.float32), True)
    w = ng.Tensor(rng.standard_normal((48, 48)).astype(np.float32), True)
    with P.activation_storage("bf16", min_bytes=0):
        if op == "matmul":
            y = x @ w
        else:
            y = getattr(ng, op)(x)
    assert y.storage_dtype == "bf16"
    for cell in y._backward.__closure__ or ():
        value = cell.cell_contents
        if isinstance(value, np.ndarray):
            assert value.nbytes < x.data.nbytes // 8


@pytest.mark.parametrize("mode", ["bf16", "int8"])
def test_mtp_loss_weights_and_gradients_with_compressed_activations(mode):
    spec = cfg.PRESETS["dense-tiny"]
    params = M.build_params(spec, np.random.default_rng(13), mtp_depth=2)
    m = M.Model(spec, params, mtp_depth=2, mtp_loss_weight=0.1,
                activation_storage=mode, gradient_storage="bf16")
    toks = np.random.default_rng(14).integers(0, 256, (24,)).astype(np.int64)
    loss, grads, info = m.loss_and_backward(toks)
    assert len(info["mtp_losses"]) == 2
    assert all(np.isfinite(v) for v in info["mtp_losses"])
    expected = info["loss_main"] + 0.1 * sum(info["mtp_losses"])
    np.testing.assert_allclose(loss, expected, rtol=0.0, atol=2e-5)
    assert grads and all(np.isfinite(P.gradient_array(g)).all() for g in grads.values())


def test_fp8_quantized_path_with_bf16_activation_storage():
    spec = cfg.PRESETS["dense-tiny"]
    params = M.build_params(spec, np.random.default_rng(5))
    m = M.Model(spec, params, forward_quant="fp8", backward_mode="fp8",
                activation_storage="bf16", gradient_storage="bf16")
    toks = np.random.default_rng(6).integers(0, 256, (24,)).astype(np.int64)
    loss, grads, _ = m.loss_and_backward(toks)
    assert np.isfinite(loss)
    assert all(np.isfinite(P.gradient_array(g)).all() for g in grads.values())


# ---------------------------------------------------------------------------
# Loss normalization and checkpoint compatibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fp32", "bf16", "int8"])
def test_cross_entropy_normalization_across_lengths(mode):
    rng = np.random.default_rng(7)
    for length in (1, 4, 31):
        logits_np = rng.standard_normal((length, 11)).astype(np.float32)
        targets = rng.integers(0, 11, (length,))
        with P.activation_storage(mode):
            logits = ng.Tensor(logits_np, True)
            mean = ng.cross_entropy(logits, targets, reduction="mean")
            none = ng.cross_entropy(ng.Tensor(logits_np), targets, reduction="none")
        expected = np.asarray(none.data).astype(np.float64).sum() / length
        assert mean.data.dtype == np.float32
        np.testing.assert_allclose(float(mean.data), expected, rtol=1e-6, atol=1e-6)


def test_checkpoint_records_storage_and_remains_loadable(tmp_path):
    spec = cfg.PRESETS["dense-tiny"]
    params = M.build_params(spec, np.random.default_rng(8))
    m = M.Model(spec, params, activation_storage="bf16", gradient_storage="bf16")
    opt = smaulopt.SmaulOPT()
    opt.register(m.p)
    man = ckpt.build_manifest(
        spec.name, m.report(),
        {"activation_storage": m.activation_storage,
         "gradient_storage": m.gradient_storage, "seq_len": 8},
        ckpt.blank_scheduler_state(), 0,
        {"bit_generator": np.random.default_rng(0).bit_generator.state},
        0, [], {},
    )
    ckpt.save_checkpoint(str(tmp_path / "run"), man, m.p, opt.state_dict())
    blob = ckpt.load_checkpoint(str(tmp_path / "run"))
    assert blob["manifest"]["config"]["activation_storage"] == "bf16"
    assert blob["manifest"]["config"]["gradient_storage"] == "bf16"
    m2 = M.Model(spec, blob["params"])  # FP32 fallback load
    toks = [65, 66, 67, 68]
    a, _ = m.forward(toks)
    b, _ = m2.forward(toks)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
