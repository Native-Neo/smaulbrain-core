"""PyTorch quantization API with an optional packed C++ CPU backend.

All trainable weights and activations remain torch.Tensor objects. The native
backend uses a narrow CPU buffer bridge for packed ternary matrix operations;
it never stores a dense dequantized weight matrix.
"""
import ctypes
import torch
from . import _quant_native as _native

FP8_E4M3_MAX = _native.FP8_E4M3_MAX
TiledPacked = _native.TiledPacked
prepare_ternary_backend = _native.prepare_ternary_backend
ternary_backend_name = _native.ternary_backend_name
recurrent_scan = _native.recurrent_scan
recurrent_backward_scan = _native.recurrent_backward_scan
retile_ternary = _native.retile_ternary


def _tensor(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor): return x.to(device=device or x.device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _np(x, dtype=None):
    a = x.detach().to("cpu").contiguous()
    if dtype is not None: a = a.to(dtype=dtype)
    return a.numpy()


def ternary_threshold(W):
    W = _tensor(W); return float(0.7 * W.abs().double().mean().item())


def ternary_scale(W, q):
    W = _tensor(W); q = _tensor(q, device=W.device, dtype=torch.int8)
    flat = W.reshape(-1)
    nz = flat.abs()[q.reshape(-1) != 0] if bool((q != 0).any()) else flat.abs()
    scale = float(nz.double().mean().item()) if nz.numel() else 1.0
    return scale if scale > 0 else 1.0


def _pack_ternary_numpy(W):
    packed, scale, shape, th = _native._pack_ternary_numpy(_np(_tensor(W)))
    return packed, float(scale), shape, float(th)


def pack_ternary(W):
    a = _np(_tensor(W))
    packed, scale, shape, threshold = _native.pack_ternary(a)
    return torch.from_numpy(packed.copy()), float(scale), tuple(shape), float(threshold)


def pack_ternary_tiled(W):
    a = _np(_tensor(W))
    packed, scale, shape, threshold = _native.pack_ternary_tiled(a)
    if isinstance(packed, TiledPacked):
        packed = TiledPacked(packed.buf.copy())
    else:
        packed = torch.from_numpy(packed.copy())
    return packed, float(scale), tuple(shape), float(threshold)


def unpack_codes(packed, shape):
    if isinstance(packed, torch.Tensor): packed = _np(packed, torch.uint8)
    return torch.from_numpy(_native.unpack_codes(packed, shape).copy())


def unpack_ternary(packed, shape, scale):
    return unpack_codes(packed, shape).float() * float(scale)


def packed_matvec_scalar(x, packed, shape, scale):
    return packed_matmat_scalar(x, packed, shape, scale)


def packed_matmat_scalar(X, packed, shape, scale):
    p = _kernel_pack(packed)
    x = _tensor(X).detach().cpu().contiguous().numpy()
    return torch.from_numpy(_native.packed_matmat_scalar(x, p, shape, scale).copy()).to(_tensor(X).device)


def _kernel_pack(packed):
    """Native kernel view of a packed buffer: tiled packs stay typed so the
    native dispatcher can pick the tiled kernels instead of reading the tile
    layout as if it were row-major 2-bit codes."""
    if isinstance(packed, TiledPacked):
        return packed
    if isinstance(packed, torch.Tensor):
        return _np(packed, torch.uint8)
    return packed


def packed_matmat(X, packed, shape, scale, allow_dense_reconstruction=False):
    if allow_dense_reconstruction:
        raise RuntimeError("forbidden path: dense ternary reconstruction is not allowed")
    x = _tensor(X)
    p = _kernel_pack(packed)
    if x.device.type == "cpu":
        y = _native.packed_matmat(_np(x), p, shape, scale, False)
        return torch.from_numpy(y).to(dtype=x.dtype)
    # Accelerator fallback: decode only to signed ternary codes for the matmul;
    # no latent FP32/FP8 master weight is reconstructed or cached.
    codes = unpack_codes(packed, shape).to(device=x.device, dtype=torch.int8)
    return (x @ codes.to(x.dtype)) * float(scale)


def packed_matmat_transpose_scalar(dY, packed, shape, scale):
    p = _kernel_pack(packed)
    dy = _np(_tensor(dY))
    out = _native.packed_matmat_transpose_scalar(dy, p, shape, scale)
    return torch.from_numpy(out.copy()).to(_tensor(dY).device)


def packed_matmat_transpose(dY, packed, shape, scale):
    dy = _tensor(dY)
    p = _kernel_pack(packed)
    if dy.device.type == "cpu":
        out = _native.packed_matmat_transpose(_np(dy), p, shape, scale)
        return torch.from_numpy(out).to(dtype=dy.dtype)
    codes = unpack_codes(packed, shape).to(device=dy.device, dtype=torch.int8)
    return (dy @ codes.to(dy.dtype).T) * float(scale)


def sample_ternary_gradient_updates(gradient, rate: float, seed: int):
    """Return sparse direct-ternary update arrays, or None without the C++ backend."""
    grad = _tensor(gradient)
    if grad.device.type != "cpu":
        return None
    sampled = _native.sample_ternary_gradient_updates(
        grad.detach().numpy(), float(rate), int(seed)
    )
    if sampled is None:
        return None
    indices, directions = sampled
    return torch.from_numpy(indices), torch.from_numpy(directions)


def apply_ternary_update_segments(packed, shape, index_segments, direction_segments):
    """Apply queued sparse ternary update segments in native code when available."""
    if not isinstance(packed, torch.Tensor) or packed.device.type != "cpu":
        return None
    packed_cpu = packed.detach()
    if packed_cpu.dtype != torch.uint8 or not packed_cpu.is_contiguous():
        return None
    index_segments = list(index_segments)
    direction_segments = list(direction_segments)
    if len(index_segments) != len(direction_segments):
        raise ValueError("index and direction segment counts differ")
    index_arrays = []
    direction_arrays = []
    for indices, directions in zip(index_segments, direction_segments):
        idx = indices.detach().to(device="cpu", dtype=torch.int32).contiguous()
        direction = directions.detach().to(device="cpu", dtype=torch.int8).contiguous()
        index_arrays.append(idx.numpy())
        direction_arrays.append(direction.numpy())
    return _native.apply_ternary_update_segments(
        packed_cpu.numpy(), int(shape[0]) * int(shape[1]),
        index_arrays, direction_arrays,
    )


def packed_matvec(x, packed, shape, scale, allow_dense_reconstruction=False):
    return packed_matmat(x, packed, shape, scale, allow_dense_reconstruction)


_FP8_TABLES = {}


def _fp8_table(device):
    key = str(device)
    table = _FP8_TABLES.get(key)
    if table is None:
        table = torch.as_tensor(_native._FP8_POSITIVE_VALUES.copy(), device=device, dtype=torch.float64)
        _FP8_TABLES[key] = table
    return table


_FP8_CHUNK = 1 << 18


def fp8_e4m3_quantize(W, scale=None):
    W = _tensor(W)
    table = _fp8_table(W.device)
    flat = W.reshape(-1)
    if scale is None:
        mx = 0.0
        for c in flat.split(_FP8_CHUNK):
            finite = c[torch.isfinite(c)]
            if finite.numel():
                mx = max(mx, float(finite.abs().max().item()))
        scale = mx / FP8_E4M3_MAX if mx > 0 else 1.0
    try:
        scale = ctypes.c_float(float(scale)).value or 1.0
    except OverflowError:
        scale = float("inf")
    # Chunked so the float64 working set is bounded instead of ~10 full-size float64 temporaries.
    out = torch.empty_like(flat, dtype=torch.float32)
    for c, o in zip(flat.split(_FP8_CHUNK), out.split(_FP8_CHUNK)):
        norm = c.double() / scale
        finite = torch.isfinite(norm)
        mag = torch.where(finite, norm.abs().clamp(max=FP8_E4M3_MAX), torch.zeros_like(norm))
        hi = torch.searchsorted(table, mag).clamp(max=table.numel() - 1)
        lo = (hi - 1).clamp(min=0)
        lower, upper = table[lo], table[hi]
        nearest = torch.where((upper - mag) < (mag - lower), upper, lower)
        sign = torch.where(norm < 0, -1.0, 1.0)
        o.copy_(torch.where(finite, sign * nearest, norm) * scale)
    return out.reshape(W.shape), scale


def fp8_error_bound(scale: float) -> float:
    return abs(float(scale)) * (FP8_E4M3_MAX / 16.0)
