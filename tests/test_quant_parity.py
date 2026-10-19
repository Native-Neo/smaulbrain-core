"""Parity/regression tests for the ternary + FP8 quantization stack.

Covers `smaulbrain.quant`, `smaulbrain._quant_native`, the packed cache and the
optional `smaulbrain.kernels` C++ helpers. References are computed in float64
from the canonical row-major packing (or from `math_ref`) so the kernels are
checked against semantics, not against themselves.
"""
import ctypes
import gc
import os
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from smaulbrain import _quant_native as qn
from smaulbrain import math_ref
from smaulbrain import quant as q

HAS_NATIVE = qn.prepare_ternary_backend()


def _canonical_dense(W):
    """float64 dense matrix for the canonical row-major packing of W."""
    W = np.asarray(W, dtype=np.float32)
    packed, scale, shape, th = qn.pack_ternary(np.ascontiguousarray(W))
    codes = qn.unpack_codes(packed, shape).astype(np.float64)
    return codes * float(scale), packed, float(scale)


def _ref_codes(W, thresh):
    """math_ref.ternary_quantize_ref semantics, vectorized in float64."""
    Wf = np.asarray(W, dtype=np.float64).reshape(-1)
    out = np.zeros(Wf.shape, np.int64)
    out[Wf > thresh] = 1
    out[Wf < -thresh] = -1
    return out.reshape(np.shape(W))


def _unpack_flat(packed, n):
    b = np.asarray(packed, np.uint8).reshape(-1)
    codes = np.empty(b.size * 4, np.int64)
    codes[0::4] = (b & 0x3).astype(np.int64) - 1
    codes[1::4] = ((b >> 2) & 0x3).astype(np.int64) - 1
    codes[2::4] = ((b >> 4) & 0x3).astype(np.int64) - 1
    codes[3::4] = ((b >> 6) & 0x3).astype(np.int64) - 1
    codes = codes[:n]
    return np.where(codes >= 1, 1, codes)


# --------------------------------------------------------------------------
# 1. packing round trip / reference semantics
# --------------------------------------------------------------------------

PACK_CASES = {
    "random": lambda rng: (rng.standard_normal((7, 5)) * 0.8).astype(np.float32),
    "zeros": lambda rng: np.zeros((4, 5), np.float32),
    "constant": lambda rng: np.full((3, 6), 0.25, np.float32),
    "constant_negative": lambda rng: np.full((5, 3), -0.1, np.float32),
    "extreme": lambda rng: np.array(
        [[1e30, -1e30, 1e-30, -1e-30], [0.0, 1.0, -1.0, 3.0]], np.float32),
    "tail1": lambda rng: rng.standard_normal(7).astype(np.float32),
    "tail3": lambda rng: rng.standard_normal(7 * 3).astype(np.float32),
    "one": lambda rng: np.array([[-0.5]], np.float32),
    "below_all": lambda rng: (rng.standard_normal((4, 4)) * 1e-20).astype(np.float32),
}


@pytest.mark.parametrize("name", sorted(PACK_CASES))
def test_pack_ternary_reference_semantics(name):
    W = PACK_CASES[name](np.random.default_rng(0))
    th = qn.ternary_threshold(W)
    ref = _ref_codes(W, th)

    for pack in (qn._pack_ternary_numpy, qn.pack_ternary):
        packed, scale, shape, th_out = pack(W)
        flat = _unpack_flat(packed, W.size)
        assert flat.tolist() == ref.reshape(-1).tolist(), f"{pack.__name__} codes differ"
        assert tuple(shape) == W.shape
        assert th_out == pytest.approx(th, rel=1e-6, abs=1e-45)
        # scale = mean|W| over the quantized-nonzero entries, 1.0 if all zero
        nz = np.abs(np.asarray(W, np.float64).reshape(-1)[ref.reshape(-1) != 0])
        want = float(nz.mean()) if nz.size else float(np.abs(W).astype(np.float64).mean())
        want = want if want > 0 else 1.0
        assert scale == pytest.approx(want, rel=1e-5, abs=1e-45)
        # tail padding is ignored: unpack_codes reproduces the codes exactly
        got = qn.unpack_codes(packed, W.shape) if hasattr(packed, "dtype") else None
        if got is not None:
            assert got.reshape(-1).tolist() == ref.reshape(-1).tolist()
    # scalar math_ref agrees on a couple of entries
    flatW = np.asarray(W, np.float32).reshape(-1)
    for w in flatW[:5]:
        assert int(math_ref.ternary_quantize_ref(float(w), th)) == int(_ref_codes(np.array([w]), th)[0])


def test_pack_ternary_all_zero_scale_is_one():
    W = np.zeros((3, 3), np.float32)
    for pack in (qn._pack_ternary_numpy, qn.pack_ternary):
        packed, scale, shape, th = pack(W)
        assert scale == 1.0
        assert th == 0.0
        assert np.all(_unpack_flat(packed, 9) == 0)
        assert np.all(np.asarray(packed, np.uint8) == 0x55)  # code 1 everywhere


def test_pack_ternary_padded_bytes_do_not_affect_unpack():
    W = (np.random.default_rng(4).standard_normal(5) * 2).astype(np.float32)
    packed, scale, shape, th = qn.pack_ternary(W)
    assert packed.size == (5 + 3) // 4
    codes = qn.unpack_codes(packed, shape)
    assert codes.shape == (5,)
    # tail codes above n are ignored even if the buffer says something else
    dirty = packed.copy()
    dirty[-1] |= 0xC0  # last byte lane 3 -> code 3
    assert qn.unpack_codes(dirty, shape).tolist() == codes.tolist()


# --------------------------------------------------------------------------
# 2. native vs numpy packing parity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("din", [1, 3, 7, 16, 33])
@pytest.mark.parametrize("dout", [1, 4, 15, 16, 17, 31, 64, 65])
def test_pack_ternary_native_numpy_parity(din, dout):
    W = (np.random.default_rng(din * 100 + dout).standard_normal((din, dout)) * 0.7).astype(np.float32)
    n_np, s_np, shape_np, th_np = qn._pack_ternary_numpy(W)
    n_nt, s_nt, shape_nt, th_nt = qn.pack_ternary(W)
    assert np.array_equal(np.asarray(n_np), np.asarray(n_nt))
    assert shape_np == W.shape == shape_nt
    assert s_nt == pytest.approx(float(s_np), rel=1e-6)
    assert th_nt == pytest.approx(float(th_np), rel=1e-6, abs=1e-12)


def test_pack_ternary_numpy_fallback_parity(monkeypatch):
    """quant.pack_ternary must agree when the native backend is unavailable."""
    W = (np.random.default_rng(9).standard_normal((9, 17)) * 1.3).astype(np.float32)
    ref, s_ref, _, _ = qn.pack_ternary(W)
    monkeypatch.setattr(qn, "_NATIVE_LIB", None)
    monkeypatch.setattr(qn, "_NATIVE_ATTEMPTED", True)
    got, s_got, shape, th = q.pack_ternary(W)
    assert np.array_equal(got.numpy(), np.asarray(ref))
    assert s_got == pytest.approx(float(s_ref), rel=1e-6)
    assert shape == W.shape


THRESHOLD_CASES = {
    "random": lambda rng: (rng.standard_normal((7, 5)) * 2.5).astype(np.float32),
    "zeros": lambda rng: np.zeros((4, 5), np.float32),
    "constant_neg": lambda rng: np.full((3, 6), -0.25, np.float32),
    "huge": lambda rng: np.array([[1e30, -1e30, 5.0]], np.float32),
    "tiny": lambda rng: (rng.standard_normal((5, 3)) * 1e-30).astype(np.float32),
    "one": lambda rng: np.array([[-0.5]], np.float32),
}


@pytest.mark.parametrize("name", sorted(THRESHOLD_CASES))
def test_ternary_threshold_torch_numpy_parity(name):
    """quant.ternary_threshold and the module-level helper agree bit for bit."""
    W = THRESHOLD_CASES[name](np.random.default_rng(0))
    th_t = q.ternary_threshold(torch.from_numpy(W))
    th_n = qn.ternary_threshold(W)
    assert th_t == pytest.approx(th_n, rel=1e-12, abs=1e-30)
    expect = 0.7 * float(np.mean(np.abs(W).astype(np.float64)))
    assert th_n == pytest.approx(expect, rel=1e-12, abs=1e-30)


def test_ternary_scale_torch_numpy_parity():
    """quant.ternary_scale mirrors the helper for every code pattern."""
    rng = np.random.default_rng(4)
    W = (rng.standard_normal((6, 7)) * 3).astype(np.float32)
    qv = np.where(W > 0.5, 1, np.where(W < -0.5, -1, 0)).astype(np.int8)
    assert (qv == 0).any() and (qv != 0).any()
    s_t = q.ternary_scale(torch.from_numpy(W), torch.from_numpy(qv))
    s_n = qn.ternary_scale(W, qv)
    assert s_t == pytest.approx(s_n, rel=1e-12, abs=0.0)
    assert s_n == pytest.approx(float(np.mean(np.abs(W[qv != 0]).astype(np.float64))), rel=1e-12)
    # all-zero codes: both fall back to the mean over every |W|
    zq = np.zeros_like(qv)
    s_tz = q.ternary_scale(torch.from_numpy(W), torch.from_numpy(zq))
    s_nz = qn.ternary_scale(W, zq)
    assert s_tz == pytest.approx(s_nz, rel=1e-12, abs=0.0)
    assert s_nz == pytest.approx(float(np.mean(np.abs(W).astype(np.float64))), rel=1e-12)
    # an all-zero matrix yields the 1.0 fallback scale in both
    zW = np.zeros((3, 4), np.float32)
    assert q.ternary_scale(torch.from_numpy(zW), torch.ones((3, 4), dtype=torch.int8)) == 1.0
    assert qn.ternary_scale(zW, np.ones((3, 4), np.int8)) == 1.0
    # non-zero codes whose weights are all zero: same 1.0 fallback
    assert q.ternary_scale(torch.from_numpy(zW), torch.ones((3, 4), dtype=torch.int8)) == 1.0
    assert qn.ternary_scale(np.full((2, 2), 5.0, np.float32), np.zeros((2, 2), np.int8)) == pytest.approx(5.0)


# --------------------------------------------------------------------------
# 3. tiled layout + matmat paths
# --------------------------------------------------------------------------

def _decode_tiled(buf, din, dout):
    """Documented tiled layout, decoded independently in Python."""
    ntiles = (dout + 15) // 16
    b = np.asarray(buf, np.uint8)
    assert b.size == ntiles * din * 4
    rows = b.reshape(ntiles, din, 4)
    lanes = (rows[:, :, :, None] >> (2 * np.arange(4))) & 3  # (tiles, din, 4, 4)
    codes = lanes.reshape(ntiles, din, 16)
    trit = np.where(codes >= 2, 1, np.where(codes == 0, -1, 0))
    return trit.transpose(1, 0, 2).reshape(din, ntiles * 16)


@pytest.mark.parametrize("dout", [1, 7, 15, 16, 17, 31, 32, 48, 64, 65])
def test_pack_ternary_tiled_matches_canonical(dout):
    rng = np.random.default_rng(dout)
    din = 6
    W = (rng.standard_normal((din, dout)) * 0.4).astype(np.float32)
    tiled, scale, shape, th = qn.pack_ternary_tiled(W)
    assert isinstance(tiled, qn.TiledPacked)
    codes_tiled = _decode_tiled(tiled.buf, din, dout)
    ref, _, s_ref = _canonical_dense(W)
    ref_codes = np.sign(ref)  # codes*scale -> {-1,0,1}*scale, scale>0
    assert np.array_equal(codes_tiled[:, :dout], ref_codes.astype(np.int64))
    # padding columns decode as zero weight
    if codes_tiled.shape[1] > dout:
        assert np.all(codes_tiled[:, dout:] == 0)
    assert scale == pytest.approx(float(s_ref), rel=1e-6)
    # the packed kernels consume the tiled buffer with the expected values
    X = rng.standard_normal((3, din)).astype(np.float32)
    got = qn.packed_matmat(X, tiled, shape, scale)
    assert np.allclose(got, X.astype(np.float64) @ ref, atol=2e-5, rtol=2e-5)


def test_pack_ternary_tiled_retile_fallback(monkeypatch):
    """Simulate a build without the fused SIMD packer: retile must agree."""
    W = (np.random.default_rng(5).standard_normal((7, 31)) * 0.6).astype(np.float32)
    ref, _, s_ref = _canonical_dense(W)
    tiled_a, scale_a, _, _ = qn.pack_ternary_tiled(W)
    X = np.random.default_rng(6).standard_normal((2, 7)).astype(np.float32)
    want = X.astype(np.float64) @ ref
    # force the pack_ternary -> retile path by hiding the fused entry point
    calls = {"n": 0}
    real = qn._NATIVE_LIB.smaulbrain_pack_ternary_tiled

    def deny(*_a):
        calls["n"] += 1
        return 1

    monkeypatch.setattr(qn._NATIVE_LIB, "smaulbrain_pack_ternary_tiled", deny, raising=False)
    tiled_b, scale_b, shape_b, _ = qn.pack_ternary_tiled(W)
    assert calls["n"] == 1
    assert isinstance(tiled_b, qn.TiledPacked)
    assert np.array_equal(_decode_tiled(tiled_a.buf, 7, 31), _decode_tiled(tiled_b.buf, 7, 31))
    assert scale_b == pytest.approx(float(scale_a), rel=1e-6)
    got = qn.packed_matmat(X, tiled_b, shape_b, scale_b)
    assert np.allclose(got, want, atol=2e-5, rtol=2e-5)


MATMAT_DIN = [1, 3, 7, 16, 17]
MATMAT_DOUT = [1, 4, 7, 8, 12, 15, 16, 17, 31, 64, 65]


@pytest.mark.parametrize("din", MATMAT_DIN)
@pytest.mark.parametrize("dout", MATMAT_DOUT)
def test_packed_matmat_paths(din, dout):
    assert HAS_NATIVE
    rng = np.random.default_rng(din * 1000 + dout)
    W = (rng.standard_normal((din, dout)) * 0.3).astype(np.float32)
    packed, scale, shape, _ = qn.pack_ternary(W)
    ref, _, s_ref = _canonical_dense(W)
    assert scale == pytest.approx(s_ref, rel=1e-6)
    for batch in (1, 2, 3, 5):
        X = rng.standard_normal((batch, din)).astype(np.float32)
        want = X.astype(np.float64) @ ref
        got = qn.packed_matmat(X, packed, shape, scale)
        assert np.allclose(got, want, atol=2e-5, rtol=2e-5), (din, dout, batch)
        got_scalar = qn.packed_matmat_scalar(X, packed, shape, scale)
        assert np.allclose(got_scalar, want, atol=2e-5, rtol=2e-5)
        dY = rng.standard_normal((batch, dout)).astype(np.float32)
        want_t = dY.astype(np.float64) @ ref.T
        got_t = qn.packed_matmat_transpose(dY, packed, shape, scale)
        assert np.allclose(got_t, want_t, atol=2e-5, rtol=2e-5)
        got_ts = qn.packed_matmat_transpose_scalar(dY, packed, shape, scale)
        assert np.allclose(got_ts, want_t, atol=2e-5, rtol=2e-5)
    x1 = rng.standard_normal(din).astype(np.float32)
    want1 = x1.astype(np.float64) @ ref
    got1 = qn.packed_matmat(x1, packed, shape, scale)
    assert got1.shape == (dout,)
    assert np.allclose(got1, want1, atol=2e-5, rtol=2e-5)
    dy1 = rng.standard_normal(dout).astype(np.float32)
    wantt1 = dy1.astype(np.float64) @ ref.T
    gott1 = qn.packed_matmat_transpose(dy1, packed, shape, scale)
    assert gott1.shape == (din,)
    assert np.allclose(gott1, wantt1, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("din,dout,batch", [(129, 257, 5), (300, 64, 1), (256, 512, 8), (512, 256, 3)])
def test_packed_matmat_large_threaded(din, dout, batch):
    """Large shapes cross the thread-pool threshold in the tiled kernels."""
    assert HAS_NATIVE
    rng = np.random.default_rng(din + dout)
    W = (rng.standard_normal((din, dout)) * 0.2).astype(np.float32)
    tiled, scale, shape, _ = qn.pack_ternary_tiled(W)
    ref, _, _ = _canonical_dense(W)
    X = rng.standard_normal((batch, din)).astype(np.float32)
    got = qn.packed_matmat(X, tiled, shape, scale)
    assert np.allclose(got, X.astype(np.float64) @ ref, atol=3e-4, rtol=3e-4)
    dY = rng.standard_normal((batch, dout)).astype(np.float32)
    got_t = qn.packed_matmat_transpose(dY, tiled, shape, scale)
    assert np.allclose(got_t, dY.astype(np.float64) @ ref.T, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("dout", [17, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quant_wrapper_uses_tiled_ternary(dout, dtype):
    """quant.packed_matmat/transpose must dispatch the tiled layout correctly."""
    assert HAS_NATIVE
    din = 6
    rng = np.random.default_rng(dout)
    W = (rng.standard_normal((din, dout)) * 0.4).astype(np.float32)
    packed, scale, shape, _ = q.pack_ternary_tiled(W)
    assert isinstance(packed, q.TiledPacked)
    ref, _, _ = _canonical_dense(W)
    X = torch.tensor(rng.standard_normal((3, din)).astype(np.float32), dtype=dtype)
    got = q.packed_matmat(X, packed, shape, scale)
    assert np.allclose(got.numpy(), X.numpy().astype(np.float64) @ ref, atol=2e-5, rtol=2e-5)
    dY = torch.tensor(rng.standard_normal((3, dout)).astype(np.float32), dtype=dtype)
    got_t = q.packed_matmat_transpose(dY, packed, shape, scale)
    assert np.allclose(got_t.numpy(), dY.numpy().astype(np.float64) @ ref.T, atol=2e-5, rtol=2e-5)


def test_quant_linear_forward_tiled_matches_reference():
    from smaulbrain.model import _QuantLinear

    din, dout = 7, 31
    rng = np.random.default_rng(11)
    W = torch.tensor((rng.standard_normal((din, dout)) * 0.4).astype(np.float32))
    packed, scale, shape, _ = q.pack_ternary_tiled(W.numpy())
    ref, _, _ = _canonical_dense(W.numpy())
    X = torch.tensor(rng.standard_normal((4, din)).astype(np.float32))
    out = _QuantLinear.apply(X, W, torch.empty(0), "ternary", "fp32", packed, scale, shape)
    assert np.allclose(out.numpy(), X.numpy().astype(np.float64) @ ref, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dout", [1, 17, 32, 64, 65])
def test_tiled_unpack_and_scalar_references(dout):
    """unpack_codes and the portable scalar references must undetile."""
    assert HAS_NATIVE
    din = 5
    rng = np.random.default_rng(dout + 1)
    W = (rng.standard_normal((din, dout)) * 0.5).astype(np.float32)
    packed, scale, shape, _ = q.pack_ternary_tiled(W)
    assert isinstance(packed, q.TiledPacked)
    ref_codes = qn.unpack_codes(qn.pack_ternary(W)[0], W.shape).astype(np.int64)
    got_codes = q.unpack_codes(packed, shape).numpy().astype(np.int64)
    assert np.array_equal(got_codes, ref_codes)
    X = rng.standard_normal((3, din)).astype(np.float32)
    want = qn.packed_matmat_scalar(X, qn.pack_ternary(W)[0], shape, scale)
    got = q.packed_matmat_scalar(X, packed, shape, scale).numpy()
    assert np.allclose(got, want, atol=2e-5, rtol=2e-5)
    dY = rng.standard_normal((3, dout)).astype(np.float32)
    want_t = qn.packed_matmat_transpose_scalar(dY, qn.pack_ternary(W)[0], shape, scale)
    got_t = q.packed_matmat_transpose_scalar(dY, packed, shape, scale).numpy()
    assert np.allclose(got_t, want_t, atol=2e-5, rtol=2e-5)


def test_tiled_accelerator_fallback_expression():
    """The non-CPU fallback decodes codes and multiplies with torch; the tiled
    pack must decode to the canonical matrix there too (no GPU needed to run
    the same expressions)."""
    assert HAS_NATIVE
    din, dout = 6, 33
    rng = np.random.default_rng(3)
    W = (rng.standard_normal((din, dout)) * 0.4).astype(np.float32)
    packed, scale, shape, _ = q.pack_ternary_tiled(W)
    ref, _, _ = _canonical_dense(W)
    X = torch.tensor(rng.standard_normal((4, din)).astype(np.float32))
    codes = q.unpack_codes(packed, shape).to(dtype=torch.int8)
    got = (X @ codes.to(X.dtype)) * float(scale)
    assert np.allclose(got.numpy(), X.numpy().astype(np.float64) @ ref, atol=2e-5, rtol=2e-5)
    dY = torch.tensor(rng.standard_normal((4, dout)).astype(np.float32))
    codes_t = q.unpack_codes(packed, shape).to(dtype=torch.int8)
    got_t = (dY @ codes_t.to(dY.dtype).T) * float(scale)
    assert np.allclose(got_t.numpy(), dY.numpy().astype(np.float64) @ ref.T, atol=2e-5, rtol=2e-5)


def test_tiled_pack_usable_without_native_backend(monkeypatch):
    """A TiledPacked value must stay consumable when the native backend is
    unavailable: the compiler-free fallback must undetile it (via
    _rowmajor_bytes) instead of dereferencing the missing native library."""
    assert HAS_NATIVE
    din, dout = 6, 33
    rng = np.random.default_rng(9)
    W = (rng.standard_normal((din, dout)) * 0.4).astype(np.float32)
    tiled, scale, shape, _ = qn.pack_ternary_tiled(W)
    assert isinstance(tiled, qn.TiledPacked)
    ref, _, _ = _canonical_dense(W)
    X = rng.standard_normal((3, din)).astype(np.float32)
    dY = rng.standard_normal((3, dout)).astype(np.float32)
    monkeypatch.setattr(qn, "_NATIVE_LIB", None)
    monkeypatch.setattr(qn, "_NATIVE_ATTEMPTED", True)
    got = qn.packed_matmat(X, tiled, shape, scale)
    assert np.allclose(got, X.astype(np.float64) @ ref, atol=2e-5, rtol=2e-5)
    got_t = qn.packed_matmat_transpose(dY, tiled, shape, scale)
    assert np.allclose(got_t, dY.astype(np.float64) @ ref.T, atol=2e-5, rtol=2e-5)
    got1 = qn.packed_matmat(X[0], tiled, shape, scale)
    assert got1.shape == (dout,)
    assert np.allclose(got1, X[0].astype(np.float64) @ ref, atol=2e-5, rtol=2e-5)
    got_t1 = qn.packed_matmat_transpose(dY[0], tiled, shape, scale)
    assert got_t1.shape == (din,)
    assert np.allclose(got_t1, dY[0].astype(np.float64) @ ref.T, atol=2e-5, rtol=2e-5)
    # the quant.py wrappers must take the same path
    got_w = q.packed_matmat(torch.tensor(X), tiled, shape, scale)
    assert np.allclose(got_w.numpy(), X.astype(np.float64) @ ref, atol=2e-5, rtol=2e-5)
    got_wt = q.packed_matmat_transpose(torch.tensor(dY), tiled, shape, scale)
    assert np.allclose(got_wt.numpy(), dY.astype(np.float64) @ ref.T, atol=2e-5, rtol=2e-5)


def test_numpy_fallback_matmat_and_transpose_parity(monkeypatch):
    """Without a C++ compiler the row-major fallback (decode + BLAS) must
    match the float64 canonical reference for odd and 1-D shapes."""
    monkeypatch.setattr(qn, "_NATIVE_LIB", None)
    monkeypatch.setattr(qn, "_NATIVE_ATTEMPTED", True)
    for din, dout in ((1, 1), (3, 8), (7, 15), (6, 16), (5, 33)):
        rng = np.random.default_rng(din * 100 + dout)
        W = (rng.standard_normal((din, dout)) * 0.4).astype(np.float32)
        packed, scale, shape, _ = qn.pack_ternary(W)
        assert not isinstance(packed, qn.TiledPacked)
        ref, _, _ = _canonical_dense(W)
        for batch in (1, 3):
            X = rng.standard_normal((batch, din)).astype(np.float32)
            want = X.astype(np.float64) @ ref
            got = qn.packed_matmat(X, packed, shape, scale)
            assert np.allclose(got, want, atol=2e-5, rtol=2e-5), (din, dout, batch)
            assert np.allclose(qn.packed_matmat_scalar(X, packed, shape, scale), want,
                               atol=2e-5, rtol=2e-5)
            dY = rng.standard_normal((batch, dout)).astype(np.float32)
            want_t = dY.astype(np.float64) @ ref.T
            got_t = qn.packed_matmat_transpose(dY, packed, shape, scale)
            assert np.allclose(got_t, want_t, atol=2e-5, rtol=2e-5), (din, dout, batch)
        x1 = rng.standard_normal(din).astype(np.float32)
        assert np.allclose(qn.packed_matmat(x1, packed, shape, scale),
                           x1.astype(np.float64) @ ref, atol=2e-5, rtol=2e-5)
        dy1 = rng.standard_normal(dout).astype(np.float32)
        assert np.allclose(qn.packed_matmat_transpose(dy1, packed, shape, scale),
                           dy1.astype(np.float64) @ ref.T, atol=2e-5, rtol=2e-5)
        # the quant.py wrappers must hit the same fallback when run on CPU
        X2 = torch.tensor(rng.standard_normal((2, din)).astype(np.float32))
        got_w = q.packed_matmat(X2, packed, shape, scale)
        assert np.allclose(got_w.numpy(), X2.numpy().astype(np.float64) @ ref,
                           atol=2e-5, rtol=2e-5)
        dY2 = torch.tensor(rng.standard_normal((2, dout)).astype(np.float32))
        got_wt = q.packed_matmat_transpose(dY2, packed, shape, scale)
        assert np.allclose(got_wt.numpy(), dY2.numpy().astype(np.float64) @ ref.T,
                           atol=2e-5, rtol=2e-5)


def test_dense_reconstruction_guard():
    """The forbidden full-materialization flag must raise, not silently pass."""
    W = (np.random.default_rng(2).standard_normal((6, 9)) * 0.5).astype(np.float32)
    packed, scale, shape, _ = qn.pack_ternary(W)
    X = np.ones((2, 6), np.float32)
    with pytest.raises(RuntimeError, match="forbidden"):
        qn.packed_matmat(X, packed, shape, scale, allow_dense_reconstruction=True)
    with pytest.raises(RuntimeError, match="forbidden"):
        qn.packed_matvec(X, packed, shape, scale, allow_dense_reconstruction=True)
    with pytest.raises(RuntimeError, match="forbidden"):
        q.packed_matmat(torch.tensor(X), packed, shape, scale, allow_dense_reconstruction=True)
    with pytest.raises(RuntimeError, match="forbidden"):
        q.packed_matvec(torch.tensor(X), packed, shape, scale, allow_dense_reconstruction=True)


# --------------------------------------------------------------------------
# 5. _QuantLinear gradients (STE) including 1-D inputs
# --------------------------------------------------------------------------

QUANT_COMBOS = [("fp32", "fp32"), ("fp8", "fp8"), ("ternary", "fp32"),
                ("ternary", "fp8"), ("ternary", "ternary")]


def _ste_reference(kind, mode, X, W, dY):
    """float64 reference for the implemented straight-through backward."""
    X1 = np.atleast_2d(np.asarray(X, np.float64))
    dY1 = np.atleast_2d(np.asarray(dY, np.float64))
    Wf = np.asarray(W, np.float64)

    def q8(a):
        return qn.fp8_e4m3_quantize(np.asarray(a, np.float32))[0].astype(np.float64)

    if kind == "fp32":
        return dY1 @ Wf.T, X1.T @ dY1
    if kind == "fp8":
        Wq = q8(W)
        if mode == "fp8":
            return q8(dY) @ Wq.T, np.atleast_2d(q8(X)).T @ np.atleast_2d(q8(dY))
        return dY1 @ Wq.T, X1.T @ dY1
    codes, _, _ = _canonical_dense(W)
    dX = dY1 @ codes.T
    if mode == "fp8":
        return dX, np.atleast_2d(q8(X)).T @ np.atleast_2d(q8(dY))
    return dX, X1.T @ dY1


def _linear_setup(kind, din, dout, seed):
    from smaulbrain.model import _QuantLinear

    rng = np.random.default_rng(seed)
    Wn = (rng.standard_normal((din, dout)) * 0.4).astype(np.float32)
    Wt = torch.tensor(Wn, requires_grad=True)
    Wq = torch.empty(0)
    packed = None
    scale = 1.0
    shape = tuple(Wt.shape)
    if kind == "fp8":
        Wq, scale = q.fp8_e4m3_quantize(Wt.detach())
    elif kind == "ternary":
        packed, scale, shape, _ = q.pack_ternary_tiled(Wt.detach().numpy())
    return _QuantLinear, Wt, Wq, packed, scale, shape


@pytest.mark.parametrize("kind,mode", QUANT_COMBOS)
@pytest.mark.parametrize("one_d", [False, True])
def test_quant_linear_ste_gradients(kind, mode, one_d):
    din, dout, batch = 7, 31, 4
    _QuantLinear, Wt, Wq, packed, scale, shape = _linear_setup(kind, din, dout, 21)
    rng = np.random.default_rng(22)
    Xn = (rng.standard_normal((batch, din)) * 0.7).astype(np.float32)
    X = torch.tensor(Xn, requires_grad=True)
    Xa = X[0] if one_d else X
    out = _QuantLinear.apply(Xa, Wt, Wq, kind, mode, packed, scale, shape)
    assert out.shape == ((dout,) if one_d else (batch, dout))
    g = torch.tensor(rng.standard_normal(out.shape).astype(np.float32))
    (out * g).sum().backward()
    assert Wt.grad is not None and X.grad is not None
    assert Wt.grad.shape == (din, dout), Wt.grad.shape
    assert X.grad.shape == X.shape
    rX, rW = _ste_reference(kind, mode, Xn[0] if one_d else Xn, Wt.detach().numpy(), g.numpy())
    assert np.allclose(Wt.grad.numpy().astype(np.float64), rW, atol=3e-5, rtol=3e-5)
    got_dx = X.grad.numpy().astype(np.float64)
    if one_d:
        # only the selected row takes part in the loss
        assert np.all(got_dx[1:] == 0.0)
        got_dx = got_dx[0]
    assert np.allclose(got_dx, np.asarray(rX).reshape(got_dx.shape), atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("kind,mode", [("fp32", "fp32"), ("ternary", "fp32"), ("ternary", "ternary")])
def test_quant_linear_dX_finite_difference(kind, mode):
    """dX is the exact derivative of the implemented (piecewise constant) pass."""
    din, dout, batch = 5, 7, 3
    _QuantLinear, Wt, Wq, packed, scale, shape = _linear_setup(kind, din, dout, 33)
    rng = np.random.default_rng(34)
    Xn = (rng.standard_normal((batch, din)) * 0.7).astype(np.float32)
    g = rng.standard_normal((batch, dout)).astype(np.float32)
    X = torch.tensor(Xn, requires_grad=True)
    out = _QuantLinear.apply(X, Wt, Wq, kind, mode, packed, scale, shape)
    (out * torch.tensor(g)).sum().backward()

    def fwd(Xa):
        return float((_QuantLinear.apply(torch.tensor(Xa), Wt.detach(),
                                         Wq.clone(), kind, mode, packed, scale, shape)
                      * torch.tensor(g)).sum())

    eps = 1e-3
    fd = np.zeros((batch, din))
    for i in range(batch):
        for j in range(din):
            Xp = Xn.copy(); Xp[i, j] += eps
            Xm = Xn.copy(); Xm[i, j] -= eps
            fd[i, j] = (fwd(Xp) - fwd(Xm)) / (2 * eps)
    assert np.allclose(X.grad.numpy(), fd, atol=2e-3, rtol=2e-3)


# --------------------------------------------------------------------------
# 5b. model-level integration (dense-tiny end to end)
# --------------------------------------------------------------------------

def test_dense_tiny_ternary_matches_reconstructed_reference():
    """Ternary forward on a whole model must equal the fp32 model whose
    matrices are the canonical ternary reconstructions, and the backward must
    flow finite gradients through the packed kernels on every layer."""
    assert HAS_NATIVE
    from smaulbrain import config as cfg
    from smaulbrain import model as M

    spec = cfg.PRESETS["dense-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(0))
    recon = {}
    for k, v in params.items():
        if v.dim() == 2 and k != "embed":
            packed, scale, shape, _ = q.pack_ternary_tiled(np.ascontiguousarray(v.numpy()))
            codes = q.unpack_codes(packed, shape).numpy().astype(np.float64)
            recon[k] = torch.tensor((codes * float(scale)).astype(np.float32))
        else:
            recon[k] = v.clone()
    m_ter = M.Model(spec, dict(params), forward_quant="ternary", backward_mode="fp32")
    m_ref = M.Model(spec, recon, forward_quant="fp32")
    tokens = [65, 66, 67, 68, 60]
    logits_ter, _ = m_ter.forward(tokens)
    logits_ref, _ = m_ref.forward(tokens)
    assert logits_ter.shape == logits_ref.shape == (len(tokens), spec.vocab_size)
    assert torch.allclose(logits_ter, logits_ref, atol=5e-5, rtol=5e-5)
    assert m_ter.packed.builds > 0
    loss, grads, _ = m_ter.loss_and_backward(tokens)
    loss_ref, grads_ref, _ = m_ref.loss_and_backward(tokens)
    assert np.isfinite(loss) and np.isfinite(loss_ref)
    assert abs(loss - loss_ref) < 1e-4
    assert sorted(grads) == sorted(m_ter.p)
    for k, g in grads.items():
        assert g.shape == m_ter.p[k].shape
        assert torch.isfinite(g).all()
        assert torch.allclose(g, grads_ref[k], atol=1e-4, rtol=1e-4), k


# --------------------------------------------------------------------------
# 4. FP8 E4M3 software quantization
# --------------------------------------------------------------------------

FP8_VALS = qn._FP8_POSITIVE_VALUES.copy()

FP8_CASES = {
    "random": np.random.default_rng(1).standard_normal((7, 5)).astype(np.float32) * 3,
    "zeros": np.zeros((4, 4), np.float32),
    "nan": np.array([[np.nan, 1.0, -1.0]], np.float32),
    "inf": np.array([[np.inf, -np.inf, 2.0]], np.float32),
    "huge": np.array([[1e30, -1e30, 1.0]], np.float32),
    "tiny": np.array([[1e-30, -1e-30, 0.0]], np.float32),
    "sat": np.array([[448.0, -448.0, 449.0, -1e9]], np.float32),
    "denorm": np.array([[2 ** -9, -(2 ** -9), 2 ** -10, 0.0]], np.float32),
}


def _fp8_ref_value(W, scale):
    """Ideal float32 output of the documented nearest-representable rule."""
    wf = np.asarray(W, np.float64)
    s = float(scale)
    norm = wf / s
    mag = np.where(np.isfinite(norm), np.minimum(np.abs(norm), 448.0), 0.0)
    hi = np.searchsorted(FP8_VALS, mag, side="left")
    hi = np.minimum(hi, FP8_VALS.size - 1)
    lo = np.maximum(hi - 1, 0)
    lower, upper = FP8_VALS[lo], FP8_VALS[hi]
    nearest = np.where((upper - mag) < (mag - lower), upper, lower)
    sign = np.where(norm < 0.0, -1.0, 1.0)
    ideal = np.where(np.isfinite(norm), sign * nearest, norm)
    return np.float32(ideal * s)


@pytest.mark.parametrize("name", sorted(FP8_CASES))
def test_fp8_quantize_matches_native(name):
    W = FP8_CASES[name]
    a, sa = q.fp8_e4m3_quantize(torch.from_numpy(W))
    b, sb = qn.fp8_e4m3_quantize(W)
    assert a.dtype == torch.float32 and a.shape == W.shape
    assert np.array_equal(a.numpy().view(np.uint8), np.asarray(b, np.float32).view(np.uint8))
    assert float(sa) == float(sb)


@pytest.mark.parametrize("scale", [None, 0.5, 1.0, 1e-3, 1e3, 448.0, -2.0, 0.0])
def test_fp8_quantize_explicit_scale(scale):
    W = np.random.default_rng(2).standard_normal((6, 9)).astype(np.float32) * 10
    a, sa = q.fp8_e4m3_quantize(torch.from_numpy(W), scale)
    b, sb = qn.fp8_e4m3_quantize(W, scale)
    assert np.array_equal(a.numpy().view(np.uint8), np.asarray(b, np.float32).view(np.uint8))
    assert float(sa) == float(sb)


@pytest.mark.parametrize("name", ["random", "huge", "tiny", "sat"])
def test_fp8_outputs_are_exactly_representable(name):
    W = FP8_CASES[name]
    out, scale = q.fp8_e4m3_quantize(torch.from_numpy(W))
    assert np.array_equal(out.numpy().view(np.uint8), _fp8_ref_value(W, scale).view(np.uint8))
    # after undoing the scale (modulo float32 product rounding) each value is
    # on the E4M3 grid
    finite = np.isfinite(W)
    mag = np.abs(out.numpy().astype(np.float64))[finite] / float(scale)
    if mag.size:
        dist = np.min(np.abs(mag[:, None] - FP8_VALS[None, :]), axis=1) if mag.size <= 64 else None
        if dist is not None:
            assert np.all(dist <= 448.0 * 2 ** -20)


@pytest.mark.parametrize("name", ["random", "huge", "tiny", "sat"])
def test_fp8_error_bound_and_saturation(name):
    W = FP8_CASES[name]
    out, scale = q.fp8_e4m3_quantize(torch.from_numpy(W))
    s = float(scale)
    finite = np.isfinite(W)
    err = np.abs(out.numpy().astype(np.float64)[finite] - W[finite].astype(np.float64))
    bound = q.fp8_error_bound(s)
    assert np.all(err <= bound)
    in_range = np.abs(W[finite].astype(np.float64)) <= 448.0 * abs(s)
    assert np.all(err[in_range] <= bound)
    # values above the representable range saturate at 448*scale
    over = finite & (np.abs(W.astype(np.float64)) > 448.0 * abs(s))
    if np.any(over):
        assert np.allclose(np.abs(out.numpy()[over]), 448.0 * abs(s), rtol=1e-6)


def test_fp8_signed_zero_maps_to_positive_zero():
    W = np.array([-0.0, 0.0], np.float32)
    out, scale = q.fp8_e4m3_quantize(torch.from_numpy(W))
    assert not np.any(np.signbit(out.numpy())), "negative zero must map to +0"
    nout, _ = qn.fp8_e4m3_quantize(W)
    assert not np.any(np.signbit(nout))
    # nonzero negative magnitudes keep their sign
    out2, _ = q.fp8_e4m3_quantize(torch.from_numpy(np.array([-1e-30], np.float32)))
    assert np.signbit(out2.numpy()[0]) and out2.numpy()[0] < 0


def test_fp8_nan_inf_are_preserved():
    W = np.array([np.nan, np.inf, -np.inf, 1.0], np.float32)
    out, scale = q.fp8_e4m3_quantize(torch.from_numpy(W))
    got = out.numpy()
    assert np.isnan(got[0]) and np.isinf(got[1]) and got[1] > 0 and np.isinf(got[2]) and got[2] < 0
    nout, _ = qn.fp8_e4m3_quantize(W)
    assert np.isnan(nout[0]) and np.isinf(nout[1]) and np.isinf(nout[2])


# --------------------------------------------------------------------------
# 6. packed / fp8 caches
# --------------------------------------------------------------------------

def test_packed_cache_version_and_replacement(tmp_path):
    from smaulbrain.model import PackedCache

    assert HAS_NATIVE
    cache = PackedCache()
    W = torch.tensor((np.random.default_rng(7).standard_normal((8, 32)) * 0.5).astype(np.float32))
    p1, s1, sh1 = cache.get("k", W)
    assert cache.builds == 1 and cache.reuse_hits == 0
    p2, _, _ = cache.get("k", W)
    assert cache.builds == 1 and cache.reuse_hits == 1 and p2 is p1
    with torch.no_grad():
        W.add_(0.25)
    p3, s3, _ = cache.get("k", W)
    assert cache.builds == 2, "in-place add_ must rebuild"
    ref = qn.pack_ternary(W.numpy())
    assert np.array_equal(qn.unpack_codes(p3, (8, 32)), qn.unpack_codes(ref[0], (8, 32)))
    with torch.no_grad():
        W.mul_(0.5)
    cache.get("k", W)
    assert cache.builds == 3, "in-place mul_ must rebuild"
    W2 = W.clone()
    cache.get("k", W2)
    assert cache.builds == 4, "a replacement tensor must rebuild"
    assert cache.d["k"]["ref"]() is W2


def test_packed_cache_weakref_and_invalidate():
    from smaulbrain.model import PackedCache

    cache = PackedCache()
    W = torch.zeros(4, 4)
    cache.get("a", W)
    W2 = torch.ones(4, 4)
    cache.get("b", W2)
    cache.invalidate(["a"])
    assert sorted(cache.d) == ["b"]
    cache.invalidate()
    assert cache.d == {}
    W3 = torch.zeros(2, 2)
    cache.get("dead", W3)
    del W3
    gc.collect()
    assert cache.d["dead"]["ref"]() is None
    W4 = torch.zeros(2, 2)
    cache.get("dead", W4)  # dead weakref must not be treated as a hit
    assert cache.builds == 4


def test_packed_cache_max_entries_eviction():
    from smaulbrain.model import PackedCache

    cache = PackedCache(max_entries=3)
    Ws = [torch.full((2, 2), float(i)) for i in range(5)]
    for i, W in enumerate(Ws):
        cache.get(f"k{i}", W)
    assert sorted(cache.d) == ["k2", "k3", "k4"]
    # re-getting an existing key must not evict another entry
    cache.get("k3", Ws[3])
    assert sorted(cache.d) == ["k2", "k3", "k4"]


def test_fp8_cache_invalidation():
    from smaulbrain.model import FP8Cache

    cache = FP8Cache()
    W = torch.tensor((np.random.default_rng(8).standard_normal((5, 7)) * 2).astype(np.float32))
    q1, s1 = cache.get("k", W)
    q2, s2 = cache.get("k", W)
    assert cache.builds == 1 and cache.reuse_hits == 1 and q1.data_ptr() == q2.data_ptr()
    with torch.no_grad():
        W.add_(0.1)
    q3, s3 = cache.get("k", W)
    assert cache.builds == 2
    expect, exp_scale = qn.fp8_e4m3_quantize(W.numpy())
    assert np.array_equal(q3.numpy().view(np.uint8), expect.view(np.uint8))
    assert float(s3) == float(exp_scale)
    cache2 = FP8Cache(max_entries=1)
    A, B = torch.randn(3, 3), torch.randn(3, 3)
    cache2.get("a", A)
    cache2.get("b", B)
    assert list(cache2.d) == ["b"]
    with torch.no_grad():
        W.fill_(0.0)
    cache.get("k", W)
    assert cache.builds == 3


# --------------------------------------------------------------------------
# 8. native recurrent scan / backward scan
# --------------------------------------------------------------------------

def _py_scan(kv, alpha, s0):
    s = np.asarray(s0, np.float64).copy()
    out = np.empty((kv.shape[0], kv.shape[1]))
    for t in range(kv.shape[0]):
        s = alpha * s + kv[t]
        out[t] = s
    return out


def _fd_scan(kv, alpha, s0, w, eps=1e-5):
    def loss(kv_, a_):
        st = _py_scan(np.asarray(kv_, np.float64), np.asarray(a_, np.float64), s0)
        return float((st * w).sum())

    dk = np.zeros(kv.shape)
    for t in range(kv.shape[0]):
        for j in range(kv.shape[1]):
            kp = kv.astype(np.float64).copy(); kp[t, j] += eps
            km = kv.astype(np.float64).copy(); km[t, j] -= eps
            dk[t, j] = (loss(kp, alpha) - loss(km, alpha)) / (2 * eps)
    da = np.zeros(alpha.shape[0])
    for j in range(alpha.shape[0]):
        ap = alpha.astype(np.float64).copy(); ap[j] += eps
        am = alpha.astype(np.float64).copy(); am[j] -= eps
        da[j] = (loss(kv, ap) - loss(kv, am)) / (2 * eps)
    return dk, da


@pytest.mark.parametrize("T,d", [(0, 4), (1, 1), (1, 8), (5, 3), (17, 16), (33, 7)])
def test_recurrent_scan_matches_reference(T, d):
    assert HAS_NATIVE
    rng = np.random.default_rng(T * 100 + d)
    kv = rng.standard_normal((T, d)).astype(np.float32)
    alpha = rng.uniform(0.0, 1.0, d).astype(np.float32)
    s0 = rng.standard_normal(d).astype(np.float32)
    state = s0.copy()
    states = np.zeros((T, d), np.float32)
    assert qn.recurrent_scan(kv, alpha, state, states)
    ref = _py_scan(kv, alpha, s0)
    if T:
        assert np.allclose(states, ref, atol=1e-5, rtol=1e-5)
        assert np.allclose(state, ref[-1], atol=1e-5, rtol=1e-5)
    else:
        assert np.array_equal(state, s0)


@pytest.mark.parametrize("T,d", [(0, 4), (1, 1), (4, 3), (9, 8), (25, 16)])
def test_recurrent_backward_scan_matches_finite_differences(T, d):
    assert HAS_NATIVE
    rng = np.random.default_rng(T * 7 + d)
    kv = rng.standard_normal((T, d)).astype(np.float32)
    alpha = rng.uniform(0.1, 1.0, d).astype(np.float32)
    s0 = np.zeros(d, np.float32)
    w = rng.standard_normal((T, d))
    state = s0.copy(); states = np.zeros((T, d), np.float32)
    assert qn.recurrent_scan(kv, alpha, state, states)
    dKV = np.zeros((T, d), np.float32); dalpha = np.zeros(d, np.float64)
    assert qn.recurrent_backward_scan(w.astype(np.float32), states, alpha, dKV, dalpha)
    fdk, fda = _fd_scan(kv, alpha, s0, w)
    assert np.allclose(dKV, fdk, atol=5e-4, rtol=5e-4)
    assert np.allclose(dalpha, fda, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("T,d", [(1, 1), (4, 3), (9, 8), (25, 16)])
def test_recurrent_backward_scan_nonzero_initial_state(T, d):
    """dalpha must include the dependence of s_0 on alpha through s0."""
    assert HAS_NATIVE
    rng = np.random.default_rng(T * 13 + d)
    kv = rng.standard_normal((T, d)).astype(np.float32)
    alpha = rng.uniform(0.1, 1.0, d).astype(np.float32)
    s0 = rng.standard_normal(d).astype(np.float32)
    w = rng.standard_normal((T, d))
    state = s0.copy(); states = np.zeros((T, d), np.float32)
    assert qn.recurrent_scan(kv, alpha, state, states)
    dKV = np.zeros((T, d), np.float32); dalpha = np.zeros(d, np.float64)
    assert qn.recurrent_backward_scan(w.astype(np.float32), states, alpha, dKV, dalpha, s0=s0)
    fdk, fda = _fd_scan(kv, alpha, s0, w)
    assert np.allclose(dKV, fdk, atol=5e-4, rtol=5e-4)
    assert np.allclose(dalpha, fda, atol=5e-4, rtol=5e-4)


def test_recurrent_backward_scan_zero_state_default_unchanged():
    """Omitting s0 keeps the documented zero-initial-state semantics."""
    assert HAS_NATIVE
    rng = np.random.default_rng(5)
    T, d = 6, 4
    kv = rng.standard_normal((T, d)).astype(np.float32)
    alpha = rng.uniform(0.1, 1.0, d).astype(np.float32)
    s0 = np.zeros(d, np.float32)
    w = rng.standard_normal((T, d)).astype(np.float32)
    state = s0.copy(); states = np.zeros((T, d), np.float32)
    assert qn.recurrent_scan(kv, alpha, state, states)
    dKV = np.zeros((T, d), np.float32); dalpha = np.zeros(d, np.float64)
    assert qn.recurrent_backward_scan(w, states, alpha, dKV, dalpha)
    dKV2 = np.zeros((T, d), np.float32); dalpha2 = np.zeros(d, np.float64)
    assert qn.recurrent_backward_scan(w, states, alpha, dKV2, dalpha2, s0=s0)
    assert np.array_equal(dKV, dKV2) and np.array_equal(dalpha, dalpha2)


def test_recurrent_scan_edge_cases():
    assert HAS_NATIVE
    d = 4
    alpha = np.ones(d, np.float32)
    state = np.zeros(d, np.float32)
    states = np.zeros((0, d), np.float32)
    assert qn.recurrent_scan(np.zeros((0, d), np.float32), alpha, state, states)
    dKV = np.zeros((0, d), np.float32)
    dalpha = np.full(d, 7.0)
    assert qn.recurrent_backward_scan(np.zeros((0, d), np.float32), np.zeros((0, d), np.float32),
                                      alpha, dKV, dalpha)
    assert np.all(dalpha == 0.0)
    with pytest.raises(ValueError):
        qn.recurrent_backward_scan(np.zeros((0, d), np.float32), np.zeros((0, d), np.float32),
                                   alpha, dKV, dalpha, s0=np.zeros(d + 1, np.float32))


# --------------------------------------------------------------------------
# 7. optional smaulbrain.kernels C++ helpers
# --------------------------------------------------------------------------

def _load_optional_kernels():
    from smaulbrain import kernels

    lib, info = kernels.try_load()
    if info.get("available"):
        return kernels, lib
    try:
        kernels.build_here(verbose=False)
    except RuntimeError as exc:  # no compiler / source missing
        pytest.skip(f"optional kernels unavailable: {exc}")
    kernels.try_load.cache_clear()
    lib, info = kernels.try_load()
    if not info.get("available"):
        pytest.skip(f"optional kernels unavailable: {info.get('reason')}")
    return kernels, lib


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _p32(a):
    return _f32(a).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _p8(a):
    return np.ascontiguousarray(a, dtype=np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


@pytest.mark.parametrize("n", [1, 2, 7, 8, 31, 64, 100])
def test_optional_rmsnorm_matches_reference(n):
    _, lib = _load_optional_kernels()
    rng = np.random.default_rng(n)
    x = (rng.standard_normal(n) * 3).astype(np.float32)
    g = rng.standard_normal(n).astype(np.float32)
    y = np.zeros(n, np.float32)
    lib.rmsnorm_fwd(_p32(x), _p32(g), _p32(y), n, ctypes.c_float(1e-6))
    ref = np.array(math_ref.rmsnorm_ref(x.tolist(), g.tolist(), 1e-6))
    assert np.allclose(y, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("din,dout", [(1, 1), (3, 5), (16, 16), (7, 31), (9, 17), (4, 4)])
def test_optional_ternary_pack_matches_reference(din, dout):
    _, lib = _load_optional_kernels()
    W = (np.random.default_rng(din * 31 + dout).standard_normal((din, dout)) * 0.9).astype(np.float32)
    n = W.size
    packed = np.zeros((n + 3) // 4, np.uint8)
    scale = ctypes.c_float(0.0)
    thresh = ctypes.c_float(0.0)
    lib.ternary_pack(_p32(W), _p8(packed), n, ctypes.byref(scale), ctypes.byref(thresh))
    ref_packed, ref_scale, _, ref_thresh = qn._pack_ternary_numpy(W)
    assert np.array_equal(packed, ref_packed)
    assert scale.value == pytest.approx(float(ref_scale), rel=1e-6)
    assert thresh.value == pytest.approx(float(ref_thresh), rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("din,dout", [(1, 1), (3, 5), (16, 16), (7, 31), (32, 33), (17, 64)])
def test_optional_packed_matvec_matches_reference(din, dout):
    _, lib = _load_optional_kernels()
    W = (np.random.default_rng(din * 41 + dout).standard_normal((din, dout)) * 0.5).astype(np.float32)
    packed, scale, shape, _ = qn.pack_ternary(W)
    x = np.random.default_rng(din + dout).standard_normal(din).astype(np.float32)
    y = np.zeros(dout, np.float32)
    lib.packed_matvec(_p32(x), _p8(packed), _p32(y), din, dout, ctypes.c_float(float(scale)))
    ref = qn.packed_matvec_scalar(x, packed, shape, float(scale))
    assert np.allclose(y, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("d", [1, 3, 8, 17, 64])
def test_optional_recurrent_step_matches_reference(d):
    _, lib = _load_optional_kernels()
    rng = np.random.default_rng(d)
    s = rng.standard_normal(d).astype(np.float32)
    k = rng.standard_normal(d).astype(np.float32)
    v = rng.standard_normal(d).astype(np.float32)
    r = rng.standard_normal(d).astype(np.float32)
    alpha = rng.uniform(0, 1, d).astype(np.float32)
    u = rng.standard_normal(d).astype(np.float32)
    s_new = np.zeros(d, np.float32)
    y = np.zeros(d, np.float32)
    lib.recurrent_step(_p32(s), _p32(k), _p32(v), _p32(r), _p32(alpha), _p32(u),
                       _p32(s_new), _p32(y), d)
    ref_s, ref_y = math_ref.recurrent_step_ref(s.tolist(), k.tolist(), v.tolist(), r.tolist(),
                                               alpha.tolist(), u.tolist())
    assert np.allclose(s_new, ref_s, atol=1e-5, rtol=1e-5)
    assert np.allclose(y, ref_y, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# 9. kernels build / try_load error handling
# --------------------------------------------------------------------------

def _isolated_kernels(tmp_path, monkeypatch):
    """Copy cpu_kernels.cpp to a tmp dir and point the kernels module at it."""
    import shutil as _shutil

    from smaulbrain import kernels

    real = Path(kernels.__file__).resolve().parent
    d = tmp_path / "kernels"
    d.mkdir()
    _shutil.copy(real / "cpu_kernels.cpp", d / "cpu_kernels.cpp")
    monkeypatch.setattr(kernels, "__file__", str(d / "__init__.py"))
    kernels.try_load.cache_clear()
    return kernels, d


def test_kernels_build_requires_compiler(monkeypatch):
    import shutil as _shutil

    from smaulbrain import kernels

    monkeypatch.setattr(_shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="requires a C\\+\\+ compiler"):
        kernels.build_here(verbose=False)


def test_kernels_build_missing_source(tmp_path, monkeypatch):
    from smaulbrain import kernels

    d = tmp_path / "empty"
    d.mkdir()
    monkeypatch.setattr(kernels, "__file__", str(d / "__init__.py"))
    with pytest.raises(RuntimeError, match="source missing"):
        kernels.build_here(verbose=False)


def test_kernels_try_load_ignores_partial_builds(tmp_path, monkeypatch):
    kernels, d = _isolated_kernels(tmp_path, monkeypatch)
    try:
        built = Path(kernels.build_here(verbose=False))
        assert built.is_file()
        partial = d / (built.name + ".4242.deadbeef.tmp.so")
        partial.write_bytes(b"partial build")
        os.utime(partial, None)
        kernels.try_load.cache_clear()
        lib, info = kernels.try_load()
        assert info.get("available"), info
        assert info.get("path") == str(built)
        lib.rmsnorm_fwd  # the scalar signatures were declared
    finally:
        kernels.try_load.cache_clear()


def test_kernels_build_dir_on_other_filesystem(tmp_path, monkeypatch):
    kernels, d = _isolated_kernels(tmp_path, monkeypatch)
    scratch = None
    for cand in ("/dev/shm", "/run", "/var/tmp"):
        if os.path.isdir(cand) and os.stat(cand).st_dev != os.stat(d).st_dev:
            scratch = cand
            break
    if scratch is None:
        pytest.skip("no writable directory on a different filesystem")
    try:
        built = Path(kernels.build_here(build_dir=scratch, verbose=False))
        assert built.is_file()
        assert built.parent == d
    finally:
        kernels.try_load.cache_clear()
