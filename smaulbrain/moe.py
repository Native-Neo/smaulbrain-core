"""NumPy top-k router and mixture-of-experts diagnostics."""
from __future__ import annotations

import math

import numpy as np

from .npgrad import Tensor, default_rng


def _arr(x, dtype=np.float32):
    if isinstance(x, Tensor):
        return np.asarray(x.data, dtype=dtype)
    return np.asarray(x, dtype=dtype)


class Router:
    def __init__(self, d: int, n_experts: int, rng=None, device="cpu"):
        if isinstance(rng, np.random.Generator):
            generator = rng
        else:
            seed = 0 if rng is None else (int(rng.integers(0, 2**63 - 1)) if hasattr(rng, "integers") else int(rng))
            generator = np.random.default_rng(seed)
        self.W = (generator.standard_normal((d, n_experts)) / math.sqrt(d)).astype(np.float32)

    @property
    def n_experts(self):
        return self.W.shape[1]

    def route(self, X, top_k: int, capacity_factor: float = 1.25, disabled=()):
        X = _arr(X)
        if not np.isfinite(X).all():
            raise ValueError("non-finite router inputs")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        disabled = sorted(set(int(i) for i in disabled))
        enabled = self.n_experts - len(disabled)
        if top_k > enabled:
            raise ValueError(f"top_k={top_k} exceeds enabled experts={enabled}")
        scores = X @ self.W
        if not np.isfinite(scores).all():
            raise ValueError("non-finite router scores")
        if disabled:
            if any(i < 0 or i >= self.n_experts for i in disabled):
                raise ValueError("disabled expert index out of range")
            scores = scores.copy()
            scores[:, disabled] = -np.inf
        idx = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        top_scores = np.take_along_axis(scores, idx, axis=1)
        shifted = top_scores - np.max(top_scores, axis=1, keepdims=True)
        exp = np.exp(shifted)
        raw_weights = exp / np.sum(exp, axis=1, keepdims=True)
        T = X.shape[0]
        flat = idx.reshape(-1)
        # capacity_factor=None disables capacity limiting (speculative verification).
        cap = flat.size if capacity_factor is None else max(
            1, math.ceil(capacity_factor * T * top_k / self.n_experts))
        order = np.argsort(flat, kind="stable")
        sorted_flat = flat[order]
        counts = np.bincount(sorted_flat, minlength=self.n_experts)
        starts = np.cumsum(counts) - counts
        rank = np.arange(flat.size) - starts[sorted_flat]
        keep = np.ones(flat.shape, dtype=bool)
        keep[order[rank >= cap]] = False
        keep = keep.reshape(idx.shape)
        kept = raw_weights * keep.astype(raw_weights.dtype)
        denom = kept.sum(axis=1, keepdims=True)
        weights = np.where(denom > 0, kept / np.maximum(denom, 1e-12), np.zeros_like(kept))
        load = np.minimum(counts, cap)
        info = {"overflow_dropped": int(flat.size - load.sum()), "cap_per_expert": cap,
                "load": load.tolist()}
        return idx, weights.astype(np.float32), info, (X, scores, idx, weights, keep)

    def backward(self, dWeights, cache):
        X, scores, idx, weights, keep = cache
        dWeights = _arr(dWeights, dtype=weights.dtype)
        row_dot = np.sum(weights * dWeights, axis=1, keepdims=True)
        dtop = np.where(keep, weights * (dWeights - row_dot), np.zeros_like(weights))
        dScores = np.zeros_like(scores)
        np.put_along_axis(dScores, idx, dtop, axis=1)
        return dScores @ self.W.T, X.T @ dScores


def softmax_rows(x):
    x = _arr(x)
    shifted = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=1, keepdims=True)


def balance_aux_loss(weights, top_idx, n_experts):
    weights = _arr(weights)
    top_idx = np.asarray(top_idx, dtype=np.int64)
    if weights.shape[0] == 0:
        return 0.0
    counts = np.bincount(top_idx.reshape(-1), minlength=n_experts).astype(np.float32)
    sums = np.zeros(n_experts, dtype=np.float32)
    np.add.at(sums, top_idx.reshape(-1), weights.reshape(-1))
    return float(n_experts * ((counts / weights.shape[0]) * (sums / weights.shape[0])).sum())


def expert_output_stats(top_idx, weights, n_experts):
    top_idx = np.asarray(top_idx, dtype=np.int64)
    counts = np.bincount(top_idx.reshape(-1), minlength=n_experts)
    return {"tokens_total": int(top_idx.shape[0]), "per_expert_counts": counts.tolist(),
            "mean_load": float(counts.astype(np.float32).mean()), "min_load": int(counts.min()),
            "max_load": int(counts.max()), "unused_experts": np.nonzero(counts == 0)[0].tolist()}
