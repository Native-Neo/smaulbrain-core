"""NumPy quantization API with an optional packed C++ CPU backend.

All trainable weights and activations are ``numpy.ndarray`` float32 buffers.
The native backend uses a narrow CPU buffer bridge for packed ternary matrix
operations; it never stores a dense dequantized weight matrix. CPU FP8 is
software-emulated (Ivy Bridge has no FP8 instructions).
"""
from __future__ import annotations

import numpy as np

from . import _quant_native as _native
from .npgrad import Tensor

FP8_E4M3_MAX = _native.FP8_E4M3_MAX
FP8_E4M3_BIAS = _native.FP8_E4M3_BIAS
TiledPacked = _native.TiledPacked
prepare_ternary_backend = _native.prepare_ternary_backend
ternary_backend_name = _native.ternary_backend_name
recurrent_scan = _native.recurrent_scan
recurrent_backward_scan = _native.recurrent_backward_scan
retile_ternary = _native.retile_ternary


def _np(x, dtype=None):
    if isinstance(x, Tensor):
        x = x.data
    a = np.ascontiguousarray(x)
    if dtype is not None:
        a = a.astype(dtype, copy=False)
    return a


def _tensor(x, dtype=np.float32):
    if isinstance(x, Tensor):
        return x if dtype is None else x.to(dtype=dtype)
    return Tensor(np.asarray(x, dtype=dtype))


def ternary_threshold(W):
    return _native.ternary_threshold(_np(W, np.float32))


def ternary_scale(W, q):
    return _native.ternary_scale(_np(W, np.float32), _np(q, np.int8))


def _pack_ternary_numpy(W):
    packed, scale, shape, th = _native._pack_ternary_numpy(_np(W, np.float32))
    return packed, float(scale), shape, float(th)


def pack_ternary(W):
    a = _np(W, np.float32)
    packed, scale, shape, threshold = _native.pack_ternary(a)
    return np.array(packed, copy=True, dtype=np.uint8), float(scale), tuple(shape), float(threshold)


def pack_ternary_tiled(W):
    a = _np(W, np.float32)
    packed, scale, shape, threshold = _native.pack_ternary_tiled(a)
    if isinstance(packed, TiledPacked):
        packed = TiledPacked(packed.buf.copy())
    else:
        packed = np.array(packed, copy=True, dtype=np.uint8)
    return packed, float(scale), tuple(shape), float(threshold)


def unpack_codes(packed, shape):
    if isinstance(packed, Tensor):
        packed = _np(packed, np.uint8)
    return np.array(_native.unpack_codes(packed, shape), copy=True)


def unpack_ternary(packed, shape, scale):
    return unpack_codes(packed, shape).astype(np.float32) * np.float32(scale)


def packed_matvec_scalar(x, packed, shape, scale):
    return packed_matmat_scalar(x, packed, shape, scale)


def packed_matmat_scalar(X, packed, shape, scale):
    p = _kernel_pack(packed)
    return np.array(_native.packed_matmat_scalar(_np(X, np.float32), p, shape, scale), copy=True)


def _kernel_pack(packed):
    """Native kernel view of a packed buffer.

    Tiled packs stay typed so the native dispatcher can pick the tiled kernels
    instead of reading the tile layout as row-major 2-bit codes.
    """
    if isinstance(packed, TiledPacked):
        return packed
    if isinstance(packed, Tensor):
        return _np(packed, np.uint8)
    return _np(packed, np.uint8)


def packed_matmat(X, packed, shape, scale, allow_dense_reconstruction=False):
    if allow_dense_reconstruction:
        raise RuntimeError("forbidden path: dense ternary reconstruction is not allowed")
    x = _np(X, np.float32)
    p = _kernel_pack(packed)
    y = _native.packed_matmat(x, p, shape, scale, False)
    return np.asarray(y, dtype=np.float32)


def packed_matmat_transpose_scalar(dY, packed, shape, scale):
    p = _kernel_pack(packed)
    return np.array(_native.packed_matmat_transpose_scalar(_np(dY, np.float32), p, shape, scale), copy=True)


def packed_matmat_transpose(dY, packed, shape, scale):
    dy = _np(dY, np.float32)
    p = _kernel_pack(packed)
    out = _native.packed_matmat_transpose(dy, p, shape, scale)
    return np.asarray(out, dtype=np.float32)


def sample_ternary_gradient_updates(gradient, rate: float, seed: int):
    """Return sparse direct-ternary update arrays, or None without the backend."""
    grad = _np(gradient, np.float32)
    sampled = _native.sample_ternary_gradient_updates(grad, float(rate), int(seed))
    if sampled is None:
        return None
    indices, directions = sampled
    return np.asarray(indices), np.asarray(directions)


def sample_ternary_gradient_update_chunk(gradient, rate: float, seed: int,
                                         global_mean_abs: float, full_cols: int,
                                         column_offset: int):
    """Sample one column-chunk of ternary code updates.

    Returns flat logical indices into the ``(rows, full_cols)`` code matrix and
    directions in {-1, 0, 1}. Sampling uses the full-matrix mean absolute
    gradient so splitting a weight matrix into chunks does not change the
    per-coordinate update probability.
    """
    grad = _np(gradient, np.float32)
    if grad.ndim != 2:
        raise ValueError("gradient chunk must be rank 2")
    rows, cols = grad.shape
    total_cols = int(full_cols)
    offset = int(column_offset)
    if offset < 0 or offset + cols > total_cols:
        raise ValueError("gradient chunk exceeds the declared column count")
    mean_abs = float(global_mean_abs)
    if not mean_abs > 0.0 or float(rate) <= 0.0:
        empty = np.empty(0, dtype=np.int32)
        return empty, np.empty(0, dtype=np.int8)
    importance = np.abs(grad)
    probabilities = np.minimum(importance * (float(rate) / mean_abs), 1.0)
    rng = np.random.default_rng(int(seed))
    selected = (rng.random(grad.shape, dtype=np.float32) < probabilities) & (importance > 0)
    local = np.flatnonzero(selected.reshape(-1))
    if local.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8)
    local_rows = local // cols
    local_cols = local % cols
    indices = (local_rows * total_cols + offset + local_cols).astype(np.int32)
    directions = (-np.sign(grad.reshape(-1)[local])).astype(np.int8)
    return indices, directions


def apply_ternary_update_segments(packed, shape, index_segments, direction_segments):
    """Apply queued sparse ternary update segments in native code when available."""
    if isinstance(packed, Tensor):
        packed = packed.data
    packed_cpu = np.asarray(packed)
    if packed_cpu.dtype != np.uint8 or not packed_cpu.flags.c_contiguous:
        return None
    index_segments = list(index_segments)
    direction_segments = list(direction_segments)
    if len(index_segments) != len(direction_segments):
        raise ValueError("index and direction segment counts differ")
    index_arrays = []
    direction_arrays = []
    for indices, directions in zip(index_segments, direction_segments):
        idx = np.ascontiguousarray(indices, dtype=np.int32)
        direction = np.ascontiguousarray(directions, dtype=np.int8)
        index_arrays.append(idx)
        direction_arrays.append(direction)
    return _native.apply_ternary_update_segments(
        packed_cpu, int(shape[0]) * int(shape[1]), index_arrays, direction_arrays
    )


def packed_matvec(x, packed, shape, scale, allow_dense_reconstruction=False):
    return packed_matmat(x, packed, shape, scale, allow_dense_reconstruction)


_FP8_POSITIVE_VALUES = _native._FP8_POSITIVE_VALUES


def fp8_e4m3_quantize(W, scale=None):
    Wf = _np(W, np.float64) if not isinstance(W, Tensor) else np.asarray(W.data, dtype=np.float64)
    return _native.fp8_e4m3_quantize(Wf, scale)


def fp8_error_bound(scale: float) -> float:
    return abs(float(scale)) * (FP8_E4M3_MAX / 16.0)
