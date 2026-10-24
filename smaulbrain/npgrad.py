"""Tiny NumPy reverse-mode autograd used by the CPU SmaulBRAIN runtime.

This is deliberately small: only the differentiable primitives the model and
tokenizer use are implemented, each with an explicit vectorized backward
closure. It is not a general-purpose framework and it never creates a tensor
per scalar element.

Design notes
------------
* Runtime dtype is float32 (matching the original torch graph). A few
  reductions (``sigmoid`` inputs, cross entropy, gradient norms) use float64
  internally exactly where the reference implementation did.
* Recurrent reads use a chunked closed-form scan (see :func:`scan`) so there is
  no Python loop over positions or elements. The chunk size adapts to the decay
  magnitude to avoid overflow.
* ``make_op`` exposes the same primitive for hand-written kernels such as the
  quantized linear layer and interpolation.
"""
from __future__ import annotations

import contextlib
import math
import numpy as np

DTYPE = np.float32
_grad_enabled = True
_rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------
def default_rng(seed=None):
    global _rng
    if seed is not None:
        _rng = np.random.default_rng(int(seed))
    return _rng


def manual_seed(seed):
    default_rng(seed)


def get_rng():
    return _rng


def randn(shape, generator=None):
    return (generator or _rng).standard_normal(shape, dtype=DTYPE)


def randn_scaled(shape, scale=1.0, generator=None):
    return randn(shape, generator).astype(DTYPE) * DTYPE(scale)


def is_grad_enabled():
    return _grad_enabled


@contextlib.contextmanager
def no_grad():
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


@contextlib.contextmanager
def enable_grad():
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = True
    try:
        yield
    finally:
        _grad_enabled = previous


def inference_mode():
    return no_grad()


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------
class Tensor:
    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._parents = tuple(_parents)
        self._backward = _backward

    # -- basic properties --------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return "cpu"

    @property
    def T(self):
        return transpose(self)

    def numel(self):
        return int(self.data.size)

    def dim(self):
        return self.data.ndim

    def size(self, dim=None):
        return self.data.size if dim is None else self.data.shape[dim]

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def numpy(self):
        return self.data

    def copy(self):
        return Tensor(self.data.copy(), self.requires_grad)

    def clone(self):
        return Tensor(self.data.copy(), self.requires_grad)

    def detach(self):
        return Tensor(self.data, False)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data), self.requires_grad)

    def to(self, dtype=None, device=None, **kwargs):
        if dtype is not None:
            dtype = np.dtype(dtype)
            if dtype != self.data.dtype:
                if (np.issubdtype(dtype, np.floating)
                        and np.issubdtype(self.data.dtype, np.floating)):
                    out = self.data.astype(dtype)
                    return make_op(out, (self,), lambda g: (g.astype(self.data.dtype),))
                return Tensor(self.data.astype(dtype), self.requires_grad)
        return self

    def float(self):
        return self.to(dtype=DTYPE)

    def long(self):
        return self.to(dtype=np.int64)

    def bool(self):
        return self.to(dtype=np.bool_)

    def t(self):
        return transpose(self)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def view(self, *shape):
        return self.reshape(*shape)

    def unsqueeze(self, dim):
        if dim < 0:
            dim += self.ndim + 1
        return reshape(self, self.shape[:dim] + (1,) + self.shape[dim:])

    def squeeze(self, dim=None):
        if dim is None:
            return reshape(self, tuple(s for s in self.shape if s != 1))
        if dim < 0:
            dim += self.ndim
        if self.shape[dim] != 1:
            return self
        return reshape(self, self.shape[:dim] + self.shape[dim + 1:])

    def transpose(self, dim0, dim1):
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return transpose(self, tuple(axes))

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return transpose(self, tuple(axes))

    def mean(self, dim=None, keepdim=False, axis=None, keepdims=None):
        if axis is not None:
            dim = axis
        if keepdims is not None:
            keepdim = keepdims
        return mean(self, axis=dim, keepdims=keepdim)

    def sum(self, dim=None, keepdim=False, dtype=None, axis=None, keepdims=None):
        if axis is not None:
            dim = axis
        if keepdims is not None:
            keepdim = keepdims
        out = sum(self, axis=dim, keepdims=keepdim)
        if dtype is not None and np.dtype(dtype) != out.dtype:
            return Tensor(out.data.astype(dtype), out.requires_grad)
        return out

    def square(self):
        return square(self)

    def sqrt(self):
        return sqrt(self)

    def rsqrt(self):
        return rsqrt(self)

    def exp(self):
        return exp(self)

    def log(self):
        return log(self)

    def abs(self):
        return abs_op(self)

    def pow(self, exponent):
        return power(self, exponent)

    def clamp(self, min=None, max=None, **kwargs):
        if min is None:
            min = kwargs.get("min_value")
        if max is None:
            max = kwargs.get("max_value")
        return clamp(self, min, max)

    def clamp_min(self, value):
        return clamp(self, value, None)

    def shape_as(self, other):
        return self.reshape(other.shape)

    # -- autograd ----------------------------------------------------------
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar output")
            grad = np.ones_like(self.data, dtype=DTYPE)
        elif isinstance(grad, Tensor):
            grad = grad.data
        grad = np.asarray(grad, dtype=self.data.dtype if np.issubdtype(self.data.dtype, np.floating) else DTYPE)

        topo = []
        seen = set()

        def visit(node):
            key = id(node)
            if key in seen:
                return
            seen.add(key)
            for parent in node._parents:
                if parent.requires_grad:
                    visit(parent)
            topo.append(node)

        visit(self)
        pending = {id(self): grad}
        for node in reversed(topo):
            g = pending.pop(id(node), None)
            if g is None:
                continue
            if node.requires_grad:
                node.grad = g.copy() if node.grad is None else node.grad + g
            if node._backward is None:
                continue
            parent_grads = node._backward(g)
            if parent_grads is None:
                continue
            for parent, pg in zip(node._parents, parent_grads):
                if pg is None or not parent.requires_grad:
                    continue
                key = id(parent)
                pg = np.asarray(pg)
                if np.issubdtype(parent.data.dtype, np.floating) and pg.dtype != parent.data.dtype:
                    pg = pg.astype(parent.data.dtype)
                pending[key] = pg if key not in pending else pending[key] + pg

    def zero_grad(self):
        self.grad = None

    def requires_grad_(self, value=True):
        self.requires_grad = bool(value)
        return self

    def __float__(self):
        return float(self.data)

    def __int__(self):
        return int(self.data)

    def __index__(self):
        return int(self.data)

    # -- operators ---------------------------------------------------------
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __pow__(self, exponent):
        return power(self, exponent)

    def __neg__(self):
        return neg(self)

    def __getitem__(self, key):
        return getitem(self, key)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def __array__(self, dtype=None):
        return self.data if dtype is None else self.data.astype(dtype)


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)

    def detach(self):
        return Tensor(self.data, False)


def as_tensor(value, dtype=None):
    if isinstance(value, Tensor):
        return value if dtype is None else value.to(dtype=dtype)
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return Tensor(arr, False)


def _t(value, dtype=None):
    return as_tensor(value, dtype=dtype)


def _unbroadcast(grad, shape):
    grad = np.asarray(grad)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def make_op(data, inputs, backward):
    """Build a Tensor with an explicit vectorized backward closure.

    ``backward(grad)`` must return one gradient (or None) per input.
    """
    inputs = tuple(_t(x) for x in inputs)
    requires = _grad_enabled and any(x.requires_grad for x in inputs)
    out = Tensor(np.asarray(data), requires)
    if requires:
        out._parents = inputs
        out._backward = backward
    return out


# ---------------------------------------------------------------------------
# Elementwise ops
# ---------------------------------------------------------------------------
def add(a, b):
    a, b = _t(a), _t(b)

    def bw(g):
        return _unbroadcast(g, a.shape), _unbroadcast(g, b.shape)

    return make_op(a.data + b.data, (a, b), bw)


def sub(a, b):
    a, b = _t(a), _t(b)

    def bw(g):
        return _unbroadcast(g, a.shape), _unbroadcast(-g, b.shape)

    return make_op(a.data - b.data, (a, b), bw)


def mul(a, b):
    a, b = _t(a), _t(b)

    def bw(g):
        return _unbroadcast(g * b.data, a.shape), _unbroadcast(g * a.data, b.shape)

    return make_op(a.data * b.data, (a, b), bw)


def div(a, b):
    a, b = _t(a), _t(b)

    def bw(g):
        ga = _unbroadcast(g / b.data, a.shape)
        gb = _unbroadcast(-g * a.data / (b.data * b.data), b.shape)
        return ga, gb

    return make_op(a.data / b.data, (a, b), bw)


def neg(a):
    a = _t(a)
    return make_op(-a.data, (a,), lambda g: (-g,))


def power(a, exponent):
    a = _t(a)
    e = np.asarray(exponent, dtype=a.data.dtype)

    def bw(g):
        return (_unbroadcast(g * e * np.power(a.data, e - 1), a.shape),)

    return make_op(np.power(a.data, e), (a,), bw)


def square(a):
    a = _t(a)
    return make_op(a.data * a.data, (a,), lambda g: (_unbroadcast(2.0 * g * a.data, a.shape),))


def sqrt(a):
    a = _t(a)
    out = np.sqrt(a.data)
    return make_op(out, (a,), lambda g: (_unbroadcast(g * 0.5 / out, a.shape),))


def rsqrt(a):
    a = _t(a)
    out = np.reciprocal(np.sqrt(a.data))
    return make_op(out, (a,), lambda g: (_unbroadcast(-0.5 * g * out * out * out, a.shape),))


def exp(a):
    a = _t(a)
    out = np.exp(a.data)
    return make_op(out, (a,), lambda g: (_unbroadcast(g * out, a.shape),))


def log(a):
    a = _t(a)
    return make_op(np.log(a.data), (a,), lambda g: (_unbroadcast(g / a.data, a.shape),))


def abs_op(a):
    a = _t(a)
    return make_op(np.abs(a.data), (a,), lambda g: (_unbroadcast(g * np.sign(a.data), a.shape),))


def sigmoid(a):
    a = _t(a)
    out = 0.5 * (1.0 + np.tanh(0.5 * a.data.astype(np.float64)))
    out = out.astype(a.data.dtype)

    def bw(g):
        return (_unbroadcast(g * out * (1.0 - out), a.shape),)

    return make_op(out, (a,), bw)


def tanh(a):
    a = _t(a)
    out = np.tanh(a.data)
    return make_op(out, (a,), lambda g: (_unbroadcast(g * (1.0 - out * out), a.shape),))


def softplus(a):
    a = _t(a)
    out = np.logaddexp(0.0, a.data.astype(np.float64)).astype(a.data.dtype)
    sig = 1.0 / (1.0 + np.exp(-a.data.astype(np.float64)))
    sig = sig.astype(a.data.dtype)

    def bw(g):
        return (_unbroadcast(g * sig, a.shape),)

    return make_op(out, (a,), bw)


def silu(a):
    a = _t(a)
    x = a.data.astype(np.float64)
    s = 1.0 / (1.0 + np.exp(-x))
    out = (x * s).astype(a.data.dtype)
    grad = (s * (1.0 + x * (1.0 - s))).astype(a.data.dtype)

    def bw(g):
        return (_unbroadcast(g * grad, a.shape),)

    return make_op(out, (a,), bw)


def gelu(a, approximate="tanh"):
    a = _t(a)
    x = a.data.astype(np.float64)
    if approximate != "tanh":
        raise NotImplementedError("only tanh GELU is used")
    c = 0.7978845608028654
    z = c * (x + 0.044715 * x ** 3)
    t = np.tanh(z)
    out = (0.5 * x * (1.0 + t)).astype(a.data.dtype)
    grad = (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 0.134145 * x * x))
    grad = grad.astype(a.data.dtype)

    def bw(g):
        return (_unbroadcast(g * grad, a.shape),)

    return make_op(out, (a,), bw)


def clamp(a, min_value=None, max_value=None):
    a = _t(a)
    out = a.data
    if min_value is not None:
        out = np.maximum(out, min_value)
    if max_value is not None:
        out = np.minimum(out, max_value)
    mask = np.ones(a.shape, dtype=bool)
    if min_value is not None:
        mask &= a.data >= min_value
    if max_value is not None:
        mask &= a.data <= max_value

    def bw(g):
        return (_unbroadcast(g * mask, a.shape),)

    return make_op(out, (a,), bw)


def where(condition, a, b):
    a, b = _t(a), _t(b)
    condition = np.asarray(condition, dtype=bool)
    out = np.where(condition, a.data, b.data)

    def bw(g):
        return _unbroadcast(np.where(condition, g, 0), a.shape), _unbroadcast(np.where(condition, 0, g), b.shape)

    return make_op(out, (a, b), bw)


def masked_fill(a, mask, value):
    a = _t(a)
    mask = np.asarray(mask, dtype=bool)
    out = np.where(mask, value, a.data)

    def bw(g):
        return (_unbroadcast(np.where(mask, 0, g), a.shape),)

    return make_op(out, (a,), bw)


# ---------------------------------------------------------------------------
# Shape / reduction ops
# ---------------------------------------------------------------------------
def reshape(a, shape):
    a = _t(a)
    original = a.shape
    return make_op(a.data.reshape(shape), (a,), lambda g: (g.reshape(original),))


def flatten(a):
    a = _t(a)
    return reshape(a, (a.data.size,))


def transpose(a, axes=None):
    a = _t(a)
    if axes is None:
        axes = tuple(reversed(range(a.ndim)))
    else:
        axes = tuple(axes)
    inverse = tuple(np.argsort(axes))

    def bw(g):
        return (np.transpose(g, inverse),)

    return make_op(np.transpose(a.data, axes), (a,), bw)


def sum(a, axis=None, keepdims=False):
    a = _t(a)

    def bw(g):
        gg = np.asarray(g)
        if axis is not None and not keepdims:
            gg = np.expand_dims(gg, axis)
        return (_unbroadcast(np.broadcast_to(gg, a.shape).copy(), a.shape),)

    return make_op(np.sum(a.data, axis=axis, keepdims=keepdims), (a,), bw)


def mean(a, axis=None, keepdims=False):
    a = _t(a)
    n = a.data.size if axis is None else a.data.shape[axis]

    def bw(g):
        gg = np.asarray(g) / n
        if axis is not None and not keepdims:
            gg = np.expand_dims(gg, axis)
        return (_unbroadcast(np.broadcast_to(gg, a.shape).copy(), a.shape),)

    return make_op(np.mean(a.data, axis=axis, keepdims=keepdims), (a,), bw)


def matmul(a, b):
    a, b = _t(a), _t(b)
    a1 = a.shape
    b1 = b.shape
    A = a.data[None, :] if a.data.ndim == 1 else a.data
    B = b.data[:, None] if b.data.ndim == 1 else b.data
    out = a.data @ b.data

    def bw(g):
        G = np.asarray(g)
        if a.data.ndim == 1 and b.data.ndim == 1:
            G = G.reshape(1, 1)
        elif a.data.ndim == 1:
            G = G.reshape(1, -1)
        elif b.data.ndim == 1:
            G = G.reshape(-1, 1)
        ga = np.matmul(G, np.swapaxes(B, -1, -2))
        gb = np.matmul(np.swapaxes(A, -1, -2), G)
        ga = _unbroadcast(ga, A.shape)
        gb = _unbroadcast(gb, B.shape)
        if a.data.ndim == 1:
            ga = ga.reshape(a1)
        if b.data.ndim == 1:
            gb = gb.reshape(b1)
        return ga, gb

    return make_op(out, (a, b), bw)


def concat(tensors, axis=0):
    tensors = tuple(_t(t) for t in tensors)
    sizes = [t.shape[axis] for t in tensors]
    out = np.concatenate([t.data for t in tensors], axis=axis)

    def bw(g):
        bounds = np.cumsum([0] + sizes)
        grads = []
        for i in range(len(tensors)):
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(bounds[i], bounds[i + 1])
            grads.append(g[tuple(sl)])
        return tuple(grads)

    return make_op(out, tensors, bw)


def stack(tensors, axis=0):
    tensors = tuple(_t(t) for t in tensors)
    out = np.stack([t.data for t in tensors], axis=axis)

    def bw(g):
        return tuple(np.take(g, i, axis=axis) for i in range(len(tensors)))

    return make_op(out, tensors, bw)


def split(a, sizes, axis=0):
    a = _t(a)
    sizes = [int(s) for s in sizes]
    starts = np.cumsum([0] + sizes)
    outs = []
    for i, size in enumerate(sizes):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(starts[i], starts[i] + size)
        outs.append(getitem(a, tuple(sl)))
    return outs


def getitem(a, key):
    a = _t(a)
    out = a.data[key]

    def bw(g):
        gb = np.zeros_like(a.data)
        np.add.at(gb, key, g)
        return (gb,)

    return make_op(np.array(out, copy=True), (a,), bw)


def gather_last(a, indices):
    """Differentiable ``take_along_axis`` on the last dimension."""
    a = _t(a)
    indices = np.asarray(indices, dtype=np.int64)
    out = np.take_along_axis(a.data, indices, axis=-1)

    def bw(g):
        gb = np.zeros_like(a.data)
        grids = np.indices(indices.shape)
        grids[-1] = indices
        np.add.at(gb, tuple(grids), g)
        return (gb,)

    return make_op(out, (a,), bw)


def index_add_rows(dst, ids, src):
    """Out-of-place ``dst.index_add_(0, ids, src)`` (no in-place graph mutation)."""
    dst, src = _t(dst), _t(src)
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    out = dst.data.copy()
    np.add.at(out, ids, src.data)

    def bw(g):
        return g.copy(), g[ids]

    return make_op(out, (dst, src), bw)


def embedding(ids, weight):
    weight = _t(weight)
    ids = np.asarray(ids, dtype=np.int64)
    out = weight.data[ids]

    def bw(g):
        gw = np.zeros_like(weight.data)
        np.add.at(gw, ids.reshape(-1), g.reshape(-1, weight.shape[-1]))
        return (gw,)

    return make_op(out, (weight,), bw)


def flip(a, axis):
    a = _t(a)
    return make_op(np.flip(a.data, axis=axis), (a,), lambda g: (np.flip(g, axis=axis),))


def cumsum(a, axis):
    a = _t(a)
    out = np.cumsum(a.data, axis=axis)

    def bw(g):
        return (np.flip(np.cumsum(np.flip(g, axis=axis), axis=axis), axis=axis),)

    return make_op(out, (a,), bw)


# ---------------------------------------------------------------------------
# Softmax / losses
# ---------------------------------------------------------------------------
def softmax(a, axis=-1):
    a = _t(a)
    shifted = a.data - np.max(a.data, axis=axis, keepdims=True)
    e = np.exp(shifted)
    out = e / np.sum(e, axis=axis, keepdims=True)

    def bw(g):
        s = np.sum(g * out, axis=axis, keepdims=True)
        return (out * (g - s),)

    return make_op(out, (a,), bw)


def log_softmax(a, axis=-1):
    a = _t(a)
    shifted = a.data - np.max(a.data, axis=axis, keepdims=True)
    lse = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    out = shifted - lse

    def bw(g):
        s = np.sum(g, axis=axis, keepdims=True)
        return (g - np.exp(out) * s,)

    return make_op(out, (a,), bw)


def cross_entropy(logits, targets, reduction="mean", ignore_index=-100):
    logits = _t(logits)
    targets = np.asarray(targets, dtype=np.int64)
    flat_logits = logits.data.reshape(-1, logits.data.shape[-1])
    flat_targets = targets.reshape(-1)
    rows = np.arange(flat_targets.size)
    valid = flat_targets != ignore_index
    safe_targets = np.where(valid, flat_targets, 0)

    shifted = flat_logits - np.max(flat_logits, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(shifted), axis=-1))
    picked = shifted[rows, safe_targets]
    losses = np.where(valid, lse - picked, 0.0).astype(DTYPE)
    denom = max(1, int(valid.sum())) if reduction == "mean" else 1
    if reduction == "mean":
        out = np.asarray(losses.sum() / denom, dtype=DTYPE)
    elif reduction == "sum":
        out = np.asarray(losses.sum(), dtype=DTYPE)
    elif reduction == "none":
        out = losses.reshape(targets.shape)
    else:
        raise ValueError(f"unknown reduction {reduction!r}")

    def bw(g):
        probs = np.exp(shifted - lse[:, None])
        probs[rows, safe_targets] -= 1.0
        probs[~valid] = 0.0
        if reduction == "mean":
            probs *= np.asarray(g) / denom
        elif reduction == "sum":
            probs *= np.asarray(g)
        else:
            probs *= np.asarray(g).reshape(-1, 1)
        return (probs.reshape(logits.shape),)

    return make_op(out, (logits,), bw)


def mse_loss(a, b, reduction="mean"):
    a, b = _t(a), _t(b)
    diff = a.data - b.data
    sq = diff * diff
    n = diff.size
    out = np.asarray(sq.mean() if reduction == "mean" else sq.sum(), dtype=DTYPE)

    def bw(g):
        scale = np.asarray(g) * (1.0 / n if reduction == "mean" else 1.0)
        return (_unbroadcast(2.0 * diff * scale, a.shape), _unbroadcast(-2.0 * diff * scale, b.shape))

    return make_op(out, (a, b), bw)


# ---------------------------------------------------------------------------
# Recurrent scan (chunked closed form, explicit backward)
# ---------------------------------------------------------------------------
def _scan_chunk_size(alpha):
    amin = float(np.min(alpha)) if np.size(alpha) else 1.0
    amin = max(amin, 1e-12)
    exponent = -math.log(amin)
    if exponent <= 1e-6:
        return 128
    return max(1, min(128, int(60.0 / exponent)))


def _scan_forward(values, alpha, chunk, s0):
    lead = values.shape[:-2]
    T, D = values.shape[-2], values.shape[-1]
    state = np.array(s0, dtype=DTYPE, copy=True)
    outs = np.empty_like(values, dtype=DTYPE)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        L = end - start
        steps = np.arange(1, L + 1, dtype=DTYPE).reshape((1,) * len(lead) + (L, 1))
        powers = np.power(alpha.reshape((1,) * len(lead) + (1, D)).astype(DTYPE), steps)
        chunk_vals = values[..., start:end, :]
        weighted = chunk_vals / powers
        prefix = np.cumsum(weighted, axis=-2)
        scanned = prefix * powers + powers * state[..., None, :]
        outs[..., start:end, :] = scanned
        state = scanned[..., -1, :].copy()
    return outs


def scan(values, alpha, s0=None):
    """s_t = alpha * s_{t-1} + values_t, returned for every t.

    ``values`` has shape (..., T, D); ``alpha`` has shape (D,). ``s0`` is an
    optional (..., D) initial state. No Python loop over positions: each chunk
    uses a cumulative sum of exponentially weighted values.
    """
    values, alpha = _t(values), _t(alpha)
    one_dim = values.ndim == 2
    v = values.data[None, :, :] if one_dim else values.data
    a = alpha.data.astype(DTYPE)
    chunk = _scan_chunk_size(a)
    initial = None if s0 is None else (_t(s0).data[None, :] if one_dim else _t(s0).data)
    if initial is None:
        initial = np.zeros(v.shape[:-2] + (v.shape[-1],), dtype=DTYPE)
    out = _scan_forward(v, a, chunk, initial)

    def bw(g):
        g = g[None, :, :] if one_dim else g
        dvalues, dalpha, ds0 = _scan_backward(g, out, a, initial, chunk)
        if one_dim:
            dvalues = dvalues[0]
            ds0 = ds0[0]
        if s0 is None:
            return dvalues, dalpha.astype(alpha.data.dtype)
        return dvalues, dalpha.astype(alpha.data.dtype), ds0.astype(_t(s0).data.dtype)

    inputs = (values, alpha) if s0 is None else (values, alpha, s0)
    return make_op(out[0] if one_dim else out, inputs, bw)


def _scan_backward(grad, states, alpha, s0, chunk):
    lead = grad.shape[:-2]
    T, D = grad.shape[-2], grad.shape[-1]
    dvalues = np.empty_like(grad, dtype=DTYPE)
    dalpha = np.zeros(alpha.shape, dtype=np.float64)
    carry = np.zeros(lead + (D,), dtype=DTYPE)
    for end in range(T, 0, -chunk):
        start = max(0, end - chunk)
        L = end - start
        g = grad[..., start:end, :]
        steps = np.arange(1, L + 1, dtype=DTYPE).reshape((1,) * len(lead) + (L, 1))
        powers = np.power(alpha.reshape((1,) * len(lead) + (1, D)).astype(DTYPE), steps)
        rev = np.flip(g, axis=-2)
        weighted = rev / powers
        prefix = np.cumsum(weighted, axis=-2)
        scanned_rev = prefix * powers + powers * carry[..., None, :]
        chunk_out = np.flip(scanned_rev, axis=-2)
        dvalues[..., start:end, :] = chunk_out
        carry = chunk_out[..., 0, :].copy()
        prev = states[..., start - 1:end - 1, :] if start > 0 else np.concatenate(
            [np.broadcast_to(s0[..., None, :], lead + (1, D)), states[..., :end - 1, :]], axis=-2)
        dalpha += np.sum(chunk_out.astype(np.float64) * prev.astype(np.float64), axis=tuple(range(len(lead) + 1)))
    ds0 = carry * alpha
    return dvalues, dalpha, ds0


# ---------------------------------------------------------------------------
# Interpolation / pooling
# ---------------------------------------------------------------------------
def interpolate1d(a, size, mode="linear"):
    """``F.interpolate(..., mode=linear/area)`` on the last axis."""
    a = _t(a)
    size = int(size)
    T = a.shape[-1]
    if T == size:
        return a

    if mode == "linear":
        scale = T / float(size)
        src = (np.arange(size, dtype=np.float64) + 0.5) * scale - 0.5
        src = np.clip(src, 0.0, T - 1)
        i0 = np.floor(src).astype(np.int64)
        i1 = np.minimum(i0 + 1, T - 1)
        w = (src - i0).astype(DTYPE)
        out = a.data[..., i0] * (1.0 - w) + a.data[..., i1] * w

        def bw(g):
            ga = np.zeros_like(a.data)
            flat = ga.reshape(-1, T)
            gf = g.reshape(-1, size)
            for idx, weight in ((i0, 1.0 - w), (i1, w)):
                contrib = gf * weight
                row = np.repeat(np.arange(flat.shape[0]), size)
                np.add.at(flat, (row, np.tile(idx, flat.shape[0])), contrib.reshape(-1))
            return (ga,)

        return make_op(out, (a,), bw)

    if mode in ("area", "adaptive_avg_pool1d"):
        starts = np.floor(np.arange(size) * T / size).astype(np.int64)
        ends = np.ceil((np.arange(size) + 1) * T / size).astype(np.int64)
        ends = np.minimum(ends, T)
        lengths = np.maximum(ends - starts, 1)
        out = np.empty(a.shape[:-1] + (size,), dtype=DTYPE)
        for i in range(size):
            out[..., i] = a.data[..., starts[i]:ends[i]].mean(axis=-1) if lengths[i] > 1 else a.data[..., starts[i]]
        if T < size:
            return interpolate1d(a, size, "linear")

        def bw(g):
            ga = np.zeros_like(a.data)
            for i in range(size):
                ga[..., starts[i]:ends[i]] += (g[..., i] / lengths[i])[..., None]
            return (ga,)

        return make_op(out, (a,), bw)

    raise ValueError(f"unsupported interpolate mode {mode!r}")


def adaptive_avg_pool1d(a, size):
    return interpolate1d(a, size, "area")


# ---------------------------------------------------------------------------
# Modules and parameters
# ---------------------------------------------------------------------------
class Module:
    def __init__(self):
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name, value):
        if name in ("training", "_parameters", "_buffers", "_modules"):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Tensor) and value.requires_grad and not isinstance(value, Parameter):
            value = Parameter(value.data)
        if isinstance(value, Parameter):
            self._parameters[name] = value
            self._modules.pop(name, None)
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Module):
            self._modules[name] = value
            self._parameters.pop(name, None)
            object.__setattr__(self, name, value)
            return
        if name in self._parameters:
            self._parameters.pop(name, None)
        if name in self._modules:
            self._modules.pop(name, None)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        self._parameters.pop(name, None)
        self._buffers.pop(name, None)
        self._modules.pop(name, None)
        object.__delattr__(self, name)

    def register_parameter(self, name, param):
        setattr(self, name, param)

    def register_buffer(self, name, value):
        arr = value.data if isinstance(value, Tensor) else np.asarray(value)
        self._buffers[name] = arr
        object.__setattr__(self, name, arr)

    def _named(self, prefix="", seen=None, kind="parameters"):
        seen = seen if seen is not None else set()
        if kind == "modules":
            for name, value in self._modules.items():
                full = f"{prefix}{name}"
                yield full, value
                yield from value._named(full + ".", seen, kind)
            return
        store = self._parameters if kind == "parameters" else self._buffers
        for name, value in store.items():
            full = f"{prefix}{name}"
            if id(value) in seen:
                continue
            seen.add(id(value))
            yield full, value
        for name, value in self._modules.items():
            yield from value._named(f"{prefix}{name}.", seen, kind)

    def named_parameters(self, prefix=""):
        return self._named(prefix, kind="parameters")

    def parameters(self):
        return (value for _, value in self.named_parameters())

    def named_buffers(self, prefix=""):
        return self._named(prefix, kind="buffers")

    def buffers(self):
        return (value for _, value in self.named_buffers())

    def named_modules(self, prefix=""):
        yield prefix.rstrip("."), self
        for name, child in self._modules.items():
            full = f"{prefix}{name}."
            yield from child.named_modules(full)

    def modules(self):
        return (module for _, module in self.named_modules())

    def train(self, mode=True):
        self.training = bool(mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def cuda(self, *args, **kwargs):
        return self

    def zero_grad(self, set_to_none=True):
        for param in self.parameters():
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad[...] = 0

    def state_dict(self):
        out = {}
        for name, param in self.named_parameters():
            out[name] = param.data
        for name, buf in self.named_buffers():
            out[name] = buf
        return out

    def load_state_dict(self, state, strict=True):
        own = self.state_dict()
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        if strict and (missing or unexpected):
            raise RuntimeError(f"state_dict mismatch; missing={missing[:8]} unexpected={unexpected[:8]}")
        for name, value in state.items():
            if name not in own:
                continue
            target = own[name]
            arr = value.data if isinstance(value, Tensor) else np.asarray(value)
            if tuple(arr.shape) != tuple(target.shape):
                raise RuntimeError(f"shape mismatch for {name}: {arr.shape} vs {target.shape}")
            target[...] = arr
        return missing, unexpected

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ParameterList(list):
    def __init__(self, values=()):
        super().__init__(Parameter(v.data if isinstance(v, Tensor) else v) for v in values)


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
            object.__setattr__(self, str(i), module)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, index):
        return self._modules[str(index)]


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, generator=None):
        super().__init__()
        rng = generator or _rng
        bound = 1.0 / math.sqrt(in_features)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(rng.uniform(-bound, bound, (out_features, in_features)).astype(DTYPE))
        if bias:
            self.bias = Parameter(rng.uniform(-bound, bound, (out_features,)).astype(DTYPE))
        else:
            self.bias = None

    def forward(self, x):
        y = matmul(x, transpose(self.weight))
        if self.bias is not None:
            y = add(y, self.bias)
        return y


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, generator=None):
        super().__init__()
        rng = generator or _rng
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = Parameter(rng.standard_normal((num_embeddings, embedding_dim)).astype(DTYPE))

    def forward(self, ids):
        return embedding(ids, self.weight)


class GELU(Module):
    def forward(self, x):
        return gelu(x)


# ---------------------------------------------------------------------------
# Functional conveniences (mirror the torch spellings used by the port)
# ---------------------------------------------------------------------------
def linear(x, weight, bias=None):
    y = matmul(x, transpose(weight))
    return y if bias is None else add(y, bias)


def clip_grad_norm_(parameters, max_norm):
    params = list(parameters)
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        total_sq += float(np.sum(p.grad.astype(np.float64) ** 2))
    total = math.sqrt(total_sq)
    if total > max_norm and total > 0:
        scale = max_norm / total
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return total


def state_rng_dict(rng):
    state = rng.bit_generator.state
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in state.items()}


def set_rng_state_dict(rng, state):
    rng.bit_generator.state = state
