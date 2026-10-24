"""NumPy tensor operators used by SmaulBRAIN."""
from __future__ import annotations

import math

import numpy as np

from . import npgrad as ng
from .npgrad import Tensor

EPS = 1e-6


def _tensor(x, device=None):
    if isinstance(x, Tensor):
        return x.float()
    return Tensor(np.asarray(x, dtype=np.float32), False)


def rmsnorm_fwd(x, g, eps=EPS):
    x = _tensor(x).float()
    g = _tensor(g).float()
    ms = x.square().mean(dim=-1, keepdim=True)
    inv = ng.rsqrt(ms + eps)
    xn = x * inv
    return xn * g, (x, g, xn, ms)


def rmsnorm_bwd(dy, cache, eps=EPS):
    x, g, xn, ms = cache
    dy = _tensor(dy).float()
    n = x.shape[-1]
    inv = 1.0 / np.sqrt(ms.data + eps)
    dxn = dy.data * g.data
    dot = np.sum(dxn * x.data, axis=-1, keepdims=True)
    dx = (dxn - x.data * dot / (n * (ms.data + eps))) * inv
    dims = tuple(range(dy.ndim - 1))
    dg = np.sum(dy.data * xn.data, axis=dims) if dims else dy.data * xn.data
    return Tensor(dx.astype(np.float32)), Tensor(dg.astype(np.float32))


def sigmoid(x):
    return ng.sigmoid(x)


def silu(x):
    return ng.silu(x)


def gelu_tanh(x):
    return ng.gelu(x, approximate="tanh")


def gelu_tanh_grad(x):
    x = _tensor(x)
    c = 0.7978845608028654
    z = c * (x + 0.044715 * x.pow(3))
    t = ng.tanh(z)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t.square()) * c * (1 + 0.134145 * x.square())


def softplus(x):
    return ng.softplus(x)


def scatter_add_rows(dst, ids, src):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    return ng.index_add_rows(dst, ids, src)


class RecurrentAttention:
    """Standalone NumPy recurrent-attention block used by math tests/tools."""

    def __init__(self, d, rng=None, device="cpu"):
        if rng is None:
            rng = ng.default_rng()
            rand = lambda shape: Tensor(ng.randn(shape) / math.sqrt(d))
            w_data = ng.randn((d,)) * np.float32(0.1)
        else:
            rand = lambda shape: Tensor(rng.standard_normal(shape).astype(np.float32) / math.sqrt(d))
            w_data = (rng.standard_normal(d) * 0.1).astype(np.float32)
        self.w = Tensor(w_data)
        self.Wr = rand((d, d))
        self.Wk = rand((d, d))
        self.Wv = rand((d, d))
        self.Wo = rand((d, d))
        self.u = Tensor(np.zeros(d, dtype=np.float32))

    def forward(self, X, s0=None):
        X = _tensor(X).float()
        Wr, Wk, Wv, Wo = (getattr(self, k).float() for k in ("Wr", "Wk", "Wv", "Wo"))
        w = self.w.float()
        u = self.u.float()
        alpha = ng.exp(-ng.softplus(w))
        state0 = None if s0 is None else _tensor(s0).float()
        r = X @ Wr
        k = X @ Wk
        v = X @ Wv
        kv = k * v
        if X.shape[0] == 0:
            state = Tensor(np.zeros_like(alpha.data))
            Y = Tensor(np.zeros((0, Wo.shape[1]), dtype=np.float32))
        else:
            states = ng.scan(kv, alpha)
            state = states[-1]
            Y = ng.sigmoid(r) * (states + ng.mul(u, kv))
        outputs = Y @ Wo
        if state0 is not None:
            # The closed-form scan above starts from zero. Re-run only when a
            # non-zero initial state is requested (inference/tests).
            s = state0
            rows = []
            for t in range(X.shape[0]):
                s = alpha * s + kv[t]
                rows.append(ng.sigmoid(r[t]) * (s + u * kv[t]))
            outputs = ng.stack(rows) @ Wo if rows else outputs
            state = s
        return outputs, (state.clone(), (X,))
