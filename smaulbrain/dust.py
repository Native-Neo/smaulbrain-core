"""Forward-only Dust training backend (experimental).

This is a NumPy-native implementation of the estimator described in

    Dahal, Mandal, Gulbahar, Vegesna, "Dust: Pretraining Transformers
    Without Backpropagation", Q Labs Research, October 2026.
    https://qlabs.sh/research/dust

and its official reference implementation, https://github.com/qlabs-eng/dust.

The estimator
-------------
For a linear layer ``y_t = W x_t`` we add independent Gaussian noise to the
layer output at every token, ``y_t -> y_t + sigma * a_t`` with
``a_t ~ N(0, I)``, and evaluate the next-byte loss again.  With the clean
per-token loss ``l_t`` and the perturbed loss ``l~_t`` the centered reward is
``c_t = l_t - l~_t``.  The reward of a draw at token ``t`` may also include
future tokens, ``r_t = sum_{s >= t} gamma**(s-t) c_s``.  Over ``K`` draws the
output-error estimate is

    g_hat_t = -(1 / (K sigma)) sum_i r_t^(i) a_t^(i)

and the weight gradient is the same local outer product backprop uses,

    G_hat_W = sum_t g_hat_t x_t^T.

Attention internals are not credited by the token loss directly.  They are
scored by how much their perturbation changes the attention output, using the
estimated error at that output:

    c_s = -<g_hat_s, Delta o_s>.

This module implements that estimator for the dense SmaulBRAIN architecture.
It never calls the autograd engine or ``Model.loss_and_backward``.

Deviations from the paper's architecture
----------------------------------------
SmaulBRAIN is not the modded-nanoGPT model used in the paper.  This backend
supports the parts that map directly onto the verified estimator:

* dense recurrent linear attention (``Wr``, ``Wk``, ``Wv``, ``Wo``);
* gated channel-mixing FFN (``Wa``, ``Wb``, ``Wc``);
* the tied byte embedding / LM head;
* ``attn.w`` (recurrence decay), ``attn.u``, and RMSNorm gains only through
  the paper's two-sided weight-space ES fallback (disabled by default).

Unsupported and rejected with a clear error: mixture-of-experts routing,
multi-token-prediction modules, paged experts, quantized forwards, and
reduced-precision activation/gradient storage.  The paper's estimator assumes
a continuous dense path; discrete top-k routing and capacity drops are not
covered by the verified material.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
import math
import os
import time

import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------------------
# Numeric helpers (mirror smaulbrain.npgrad and smaulbrain.ops)
# ---------------------------------------------------------------------------
def _sigmoid(x):
    out = 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))
    return out.astype(np.float32)


def _sigmoid_fast(x):
    """Float32-only sigmoid for perturbed forwards.

    The clean forward keeps the reference float64 intermediates so it remains
    bit-comparable with ``Model.forward``.  Perturbed forwards have a stochastic
    reward already; computing their nonlinearities in float32 removes the
    dominant elementwise cost without changing the estimator's definition.
    """
    x = np.asarray(x, dtype=np.float32)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softplus(x):
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64)).astype(np.float32)


def _silu(x):
    x64 = np.asarray(x, dtype=np.float64)
    s = 1.0 / (1.0 + np.exp(-x64))
    return (x64 * s).astype(np.float32)


def _silu_fast(x):
    """Float32-only SiLU for perturbed forwards; see ``_sigmoid_fast``."""
    x = np.asarray(x, dtype=np.float32)
    s = 1.0 / (1.0 + np.exp(-x))
    return x * s


def _rmsnorm(x, gain):
    x = np.asarray(x, dtype=np.float32)
    ms = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    inv = np.reciprocal(np.sqrt(ms + np.float32(EPS)))
    return x * inv * np.asarray(gain, dtype=np.float32)


def _scan_chunk_size(alpha):
    amin = float(np.min(alpha)) if np.size(alpha) else 1.0
    amin = max(amin, 1e-12)
    exponent = -math.log(amin)
    if exponent <= 1e-6:
        return 128
    return max(1, min(128, int(60.0 / exponent)))


def _scan(values, alpha):
    """s_t = alpha * s_{t-1} + values_t for every position.

    Mirrors the active ``npgrad`` scan implementation and accepts leading draw
    axes, so a population of independent sequences can be evaluated together.
    The single-sequence path shares the native sequential kernel with the
    autograd runtime, keeping Dust's clean forward on the same scan path as
    ``Model._forward_impl`` when that backend is available.
    """
    v = np.asarray(values, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    one_dim = v.ndim == 2
    if (one_dim and v.flags.c_contiguous and a.flags.c_contiguous
            and v.shape[1] == a.shape[0]):
        from . import _quant_native as _qn
        if _qn._NATIVE_LIB is None:
            _qn.prepare_ternary_backend()
        if _qn._NATIVE_LIB is not None:
            states = np.empty_like(v)
            state = np.zeros(v.shape[1], dtype=np.float32)
            if _qn.recurrent_scan(v, a, state, states):
                return states
    if one_dim:
        v = v[None, :, :]
    lead = v.shape[:-2]
    T, D = v.shape[-2], v.shape[-1]
    chunk = _scan_chunk_size(a)
    state = np.zeros(lead + (D,), dtype=np.float32)
    out = np.empty_like(v)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        L = end - start
        steps = np.arange(1, L + 1, dtype=np.float32).reshape((1,) * len(lead) + (L, 1))
        powers = np.power(a.reshape((1,) * len(lead) + (1, D)), steps)
        chunk_vals = v[..., start:end, :]
        prefix = np.cumsum(chunk_vals / powers, axis=-2)
        scanned = prefix * powers + powers * state[..., None, :]
        out[..., start:end, :] = scanned
        state = scanned[..., -1, :].copy()
    return out[0] if one_dim else out


def per_token_cross_entropy(logits, targets):
    """Cross entropy with ``reduction='none'`` in the runtime's fp32 path.

    ``logits`` may have leading draw axes.  ``targets`` has shape ``[T]`` and is
    broadcast over those axes.  The returned array has ``logits.shape[:-1]``.
    """
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    target_shape = logits.shape[:-1]
    tgt = np.broadcast_to(np.asarray(targets, dtype=np.int64), target_shape)
    flat = logits.reshape(-1, logits.shape[-1])
    flat_t = tgt.reshape(-1)
    if flat_t.size and (flat_t.min() < 0 or flat_t.max() >= logits.shape[-1]):
        raise ValueError("target token outside the vocabulary")
    shifted = flat - np.max(flat, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(shifted), axis=-1))
    picked = shifted[np.arange(flat_t.size), flat_t]
    return (lse - picked).reshape(target_shape)


def future_credit(score, gamma):
    """``out_t = sum_{s >= t} gamma**(s-t) score_s`` for ``score`` [..., T].

    The reverse recursion is used directly.  The algebraically equivalent
    ``gamma**-t`` prefix-sum form loses all precision for small gamma and long
    sequences, which matters here because SmaulBRAIN's recurrence decay is
    roughly 0.5 rather than the paper's 0.98 credit decay.
    """
    score = np.asarray(score, dtype=np.float64)
    if gamma == 0.0:
        return score.astype(np.float32)
    T = score.shape[-1]
    out = np.empty_like(score)
    carry = np.zeros(score.shape[:-1], dtype=np.float64)
    for t in range(T - 1, -1, -1):
        carry = score[..., t] + gamma * carry
        out[..., t] = carry
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DustConfig:
    """Verified estimator constants plus CPU population controls.

    The estimator and its normalization follow the paper.  The constants are
    re-tuned for SmaulBRAIN's activation scale and recurrent linear attention:
    the paper's sigma=0.2/0.4 perturbations are too large here (they bias the
    direct output-error estimate), while the paper's gamma=0.98 credit decay
    does not match this model's recurrence decay (~0.5).  The documentation
    records the controlled comparisons behind these values.  Draw counts are
    deliberately small CPU values rather than the paper's 256--16384
    population; the head is cheap and gets a larger fixed budget.
    """

    writer_draws: int = 2
    hidden_draws: int = 2
    hub_draws: int = 2
    embed_draws: int = 2
    head_draws: int = 1024
    local_draws: int = 2
    scalar_draws: int = 0
    chunk: int = 8
    sigma: float = 0.05
    hub_sigma: float = 0.05
    hub_sigma_deep: float = 0.05
    local_sigma: float = 0.05
    head_sigma: float = 0.05
    scalar_sigma: float = 0.03
    gamma: float = 0.5
    head_slab: int = 256
    seed: int = 0
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    grad_clip: float = 0.0

    def validate(self, vocab_size: int) -> None:
        for name in ("writer_draws", "hidden_draws", "hub_draws", "embed_draws",
                     "head_draws"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"dust {name} must be an integer >= 1, got {value!r}")
        if not isinstance(self.local_draws, (int, np.integer)) or int(self.local_draws) < 0:
            raise ValueError("dust local_draws must be an integer >= 0")
        if not isinstance(self.scalar_draws, (int, np.integer)) or int(self.scalar_draws) < 0:
            raise ValueError("dust scalar_draws must be an integer >= 0")
        if not isinstance(self.chunk, (int, np.integer)) or int(self.chunk) < 1:
            raise ValueError("dust chunk must be an integer >= 1")
        for name in ("sigma", "hub_sigma", "hub_sigma_deep", "local_sigma",
                     "head_sigma", "scalar_sigma"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"dust {name} must be a finite positive number")
        if not (0.0 <= float(self.gamma) <= 1.0):
            raise ValueError("dust gamma must be in [0, 1]")
        if not isinstance(self.head_slab, (int, np.integer)) or int(self.head_slab) < 1:
            raise ValueError("dust head_slab must be an integer >= 1")
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0.0:
            raise ValueError("dust lr must be a finite positive number")
        if not (0.0 <= float(self.momentum) < 1.0):
            raise ValueError("dust momentum must be in [0, 1)")
        if not isinstance(self.seed, (int, np.integer)) or int(self.seed) < 0:
            raise ValueError("dust seed must be an integer >= 0")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0.0:
            raise ValueError("dust weight_decay must be a finite non-negative number")
        if not math.isfinite(float(self.grad_clip)) or float(self.grad_clip) < 0.0:
            raise ValueError("dust grad_clip must be a finite non-negative number")
        if self.head_slab > vocab_size:
            self.head_slab = int(vocab_size)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, blob):
        if not isinstance(blob, dict):
            raise ValueError("checkpoint dust config must be an object")
        allowed = {f.name for f in fields(cls)}
        unknown = set(blob) - allowed
        if unknown:
            raise ValueError(f"unknown dust config keys: {sorted(unknown)}")
        return cls(**blob)


def config_from_args(args) -> DustConfig:
    seed = getattr(args, "dust_seed", None)
    if seed is None:
        seed = int(getattr(args, "seed", 0)) + 0xD057
    return DustConfig(
        writer_draws=int(getattr(args, "dust_writer_draws", 2)),
        hidden_draws=int(getattr(args, "dust_hidden_draws", 2)),
        hub_draws=int(getattr(args, "dust_hub_draws", 2)),
        embed_draws=int(getattr(args, "dust_embed_draws", 2)),
        head_draws=int(getattr(args, "dust_head_draws", 1024)),
        local_draws=int(getattr(args, "dust_local_draws", 2)),
        scalar_draws=int(getattr(args, "dust_scalar_draws", 0)),
        chunk=int(getattr(args, "dust_chunk", 8)),
        sigma=float(getattr(args, "dust_sigma", 0.05)),
        hub_sigma=float(getattr(args, "dust_hub_sigma", 0.05)),
        hub_sigma_deep=float(getattr(args, "dust_hub_sigma_deep", 0.05)),
        local_sigma=float(getattr(args, "dust_local_sigma", 0.05)),
        head_sigma=float(getattr(args, "dust_head_sigma", 0.05)),
        scalar_sigma=float(getattr(args, "dust_scalar_sigma", 0.03)),
        gamma=float(getattr(args, "dust_gamma", 0.5)),
        head_slab=int(getattr(args, "dust_head_slab", 256)),
        seed=int(seed),
        lr=float(getattr(args, "dust_lr", 0.05)),
        momentum=float(getattr(args, "dust_momentum", 0.9)),
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
        grad_clip=float(getattr(args, "dust_grad_clip", 0.0)),
    )


def check_supported(model, config: DustConfig) -> None:
    """Reject architectures the verified estimator cannot train faithfully."""
    spec = model.spec
    if spec.kind != "dense":
        raise ValueError(
            "Dust supports dense models only. The paper's estimator assumes a "
            "continuous dense path; discrete top-k MoE routing, capacity drops, "
            "and paged experts are not covered by the verified method.")
    if getattr(model, "mtp_depth", 0) != 0:
        raise ValueError(
            "Dust does not support MTP heads: the objective is a sum of several "
            "token cross-entropies and the paper's estimator only defines the "
            "next-token loss. Use --mtp-depth 0.")
    if getattr(model, "forward_quant", "fp32") != "fp32":
        raise ValueError("Dust requires --forward-quant fp32")
    if getattr(model, "backward_mode", "fp32") != "fp32":
        raise ValueError("Dust requires --backward-mode fp32")
    if getattr(model, "activation_storage", "fp32") != "fp32":
        raise ValueError(
            "Dust uses its own fp32 clean-activation cache; "
            "--activation-storage other than fp32 is not supported")
    if getattr(model, "gradient_storage", "fp32") != "fp32":
        raise ValueError("Dust does not use --gradient-storage; require fp32")
    if any(name.startswith("expert.") or name == "router" for name in model.p):
        raise ValueError("Dust does not support mixture-of-experts parameters")
    config.validate(spec.vocab_size)


# ---------------------------------------------------------------------------
# Forward-only model
# ---------------------------------------------------------------------------
class _Clean:
    """Cached clean forward.  Arrays are float32 and owned by this object."""

    __slots__ = ("tok", "targets", "T", "x0", "x_in", "h", "r", "k", "v",
                 "state", "y", "h2", "z", "final_h", "logits", "loss",
                 "nbytes")

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)

        def nbytes(value):
            if isinstance(value, np.ndarray):
                return int(value.nbytes)
            if isinstance(value, (list, tuple)):
                return int(sum(nbytes(item) for item in value))
            return 0

        self.nbytes = int(sum(
            nbytes(getattr(self, key))
            for key in self.__slots__
            if key not in ("tok", "targets", "T", "nbytes")))


class DustForward:
    """The dense SmaulBRAIN forward, instrumented for the Dust estimator.

    This is intentionally independent of the autograd graph.  It mirrors
    ``Model._forward_impl`` for dense models and is checked against it by the
    test suite.
    """

    def __init__(self, model):
        self.model = model
        self.spec = model.spec
        self.d = self.spec.d_model
        self.L = self.spec.n_layers
        p = model.p
        # Parameter arrays are updated in place by the optimizer, so holding
        # references is safe.  This avoids rebuilding parameter names and
        # walking the parameter dict on every inner-loop block evaluation.
        self._layers = []
        for layer in range(self.L):
            px = f"L{layer}."
            self._layers.append({
                "n1": p[px + "n1"].data,
                "Wr": p[px + "attn.Wr"].data,
                "Wk": p[px + "attn.Wk"].data,
                "Wv": p[px + "attn.Wv"].data,
                "w": p[px + "attn.w"].data,
                "u": p[px + "attn.u"].data,
                "Wo": p[px + "attn.Wo"].data,
                "n2": p[px + "n2"].data,
                "Wa": p[px + "ffn.Wa"].data,
                "Wb": p[px + "ffn.Wb"].data,
                "Wc": p[px + "ffn.Wc"].data,
            })
        self._final_norm = p["final_norm"].data
        self._embed = p["embed"].data

    @property
    def p(self):
        return self.model.p

    def _block(self, layer, x, jitter=None, fast=False, collect=True):
        lp = self._layers[layer]
        jit = jitter or {}
        sigmoid = _sigmoid_fast if fast else _sigmoid
        silu = _silu_fast if fast else _silu
        h = _rmsnorm(x, lp["n1"])
        r = h @ lp["Wr"]
        k = h @ lp["Wk"]
        v = h @ lp["Wv"]
        key = f"L{layer}.attn.r"
        if key in jit:
            r = r + jit[key]
        key = f"L{layer}.attn.k"
        if key in jit:
            k = k + jit[key]
        key = f"L{layer}.attn.v"
        if key in jit:
            v = v + jit[key]

        # attn.w can be changed by the optional scalar ES path, so recompute
        # alpha instead of caching its value.
        alpha = np.exp(-_softplus(lp["w"]))
        kv = k * v
        state = _scan(kv, alpha)
        y = sigmoid(r) * (state + lp["u"] * kv)
        key = f"L{layer}.attn.hub"
        if key in jit:
            y = y + jit[key]
        o = y @ lp["Wo"]
        key = f"L{layer}.attn.out"
        if key in jit:
            o = o + jit[key]
        x = x + o

        h2 = _rmsnorm(x, lp["n2"])
        a = h2 @ lp["Wa"]
        b = h2 @ lp["Wb"]
        key = f"L{layer}.ffn.a"
        if key in jit:
            a = a + jit[key]
        key = f"L{layer}.ffn.b"
        if key in jit:
            b = b + jit[key]
        z = silu(a) * b
        f = z @ lp["Wc"]
        key = f"L{layer}.ffn.out"
        if key in jit:
            f = f + jit[key]
        x = x + f
        if not collect:
            return x
        return x, {
            "h": h, "r": r, "k": k, "v": v, "state": state, "y": y, "o": o,
            "h2": h2, "a": a, "b": b, "z": z, "f": f,
        }

    def run_blocks(self, start, x, jitter=None, fast=False):
        jit = jitter or {}
        x = np.asarray(x, dtype=np.float32)
        if "embed" in jit:
            x = x + jit["embed"]
        for layer in range(int(start), self.L):
            x = self._block(layer, x, jit, fast=fast, collect=False)
        h = _rmsnorm(x, self._final_norm)
        logits = h @ self._embed.T
        return logits, h, x

    def loss_only(self, tokens):
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if tokens.size < 2:
            raise ValueError("training sequence needs at least two tokens")
        x0 = self.p["embed"].data[tokens[:-1]]
        logits, _, _ = self.run_blocks(0, x0[None, :, :])
        return float(np.mean(per_token_cross_entropy(logits, tokens[1:])))

    def capture(self, tokens):
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if tokens.size < 2:
            raise ValueError("training sequence needs at least two tokens")
        tok = tokens[:-1]
        targets = tokens[1:]
        T = tok.size
        x0 = self.p["embed"].data[tok]
        x = x0
        xs, hs, rs, ks, vs, states, ys = [], [], [], [], [], [], []
        h2s, zs = [], []
        for layer in range(self.L):
            x_in = x
            x, parts = self._block(layer, x, None)
            xs.append(x_in)
            hs.append(parts["h"])
            rs.append(parts["r"])
            ks.append(parts["k"])
            vs.append(parts["v"])
            states.append(parts["state"])
            ys.append(parts["y"])
            h2s.append(parts["h2"])
            zs.append(parts["z"])
        final_h = _rmsnorm(x, self.p["final_norm"].data)
        logits = final_h @ self.p["embed"].data.T
        loss = per_token_cross_entropy(logits, targets)
        if not np.isfinite(logits).all() or not np.isfinite(loss).all():
            raise FloatingPointError("non-finite clean forward in Dust")
        return _Clean(
            tok=tok, targets=targets, T=T, x0=x0, x_in=xs, h=hs, r=rs, k=ks,
            v=vs, state=states, y=ys, h2=h2s, z=zs, final_h=final_h,
            logits=logits, loss=loss,
        )


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------
class DustEngine:
    """One Dust update: clean forward, perturbation estimates, gradients."""

    def __init__(self, model, config: DustConfig):
        check_supported(model, config)
        self.model = model
        self.config = config
        self.fw = DustForward(model)
        self.spec = model.spec
        self.L = self.spec.n_layers
        self.d = self.spec.d_model
        self.V = self.spec.vocab_size
        self._matrix_sites = self._collect_matrix_sites()

    @property
    def p(self):
        return self.model.p

    def _collect_matrix_sites(self):
        names = set()
        for layer in range(self.L):
            names.update({
                f"L{layer}.attn.Wr", f"L{layer}.attn.Wk",
                f"L{layer}.attn.Wv", f"L{layer}.attn.Wo",
                f"L{layer}.ffn.Wa", f"L{layer}.ffn.Wb",
                f"L{layer}.ffn.Wc",
            })
        names.add("embed")
        return names

    def config_dict(self):
        return self.config.to_dict()

    # -- direct token-loss sites -------------------------------------------
    def _direct_group(self, base_x, start, sites, widths, draws, sigma, rng,
                      clean, stats):
        """Reward-weighted output error for one same-forward site group."""
        n_max = min(int(self.config.chunk), int(draws))
        errors = {site: np.zeros((clean.T, widths[site]), dtype=np.float32)
                  for site in sites}
        remaining = int(draws)
        while remaining > 0:
            n = min(n_max, remaining)
            noise = {site: rng.standard_normal(
                (n, clean.T, widths[site]), dtype=np.float32) for site in sites}
            jitter = {site: noise[site] * np.float32(sigma) for site in sites}
            x = np.broadcast_to(base_x, (n, clean.T, self.d))
            logits, _, _ = self.fw.run_blocks(start, x, jitter, fast=True)
            perturbed = per_token_cross_entropy(logits, clean.targets)
            stats["temp_bytes"] = max(
                stats["temp_bytes"],
                int(logits.nbytes + perturbed.nbytes))
            reward = clean.loss[None, :] - perturbed
            if n > 1:
                reward = reward - reward.mean(axis=0, keepdims=True)
            for site in sites:
                errors[site] -= np.einsum(
                    "nt,ntd->td", reward, noise[site],
                    optimize=True) / np.float32(sigma)
            stats["block_evals"] += n * (self.L - int(start))
            stats["draws"] += n
            stats["draw_batches"] += 1
            stats["temp_bytes"] = max(
                stats["temp_bytes"], int(sum(a.nbytes for a in noise.values())))
            remaining -= n
        for site in sites:
            errors[site] /= np.float32(draws)
        return errors

    def _direct_groups(self, base_x, start, groups, rng, clean, stats):
        """Evaluate several independent direct-output groups in one forward.

        Each batch row belongs to exactly one group and has zero jitter at
        every other group's sites.  The batched forward is therefore identical
        to running each group separately, but it amortizes the Python and
        dispatch overhead of the recurrent block loop.  Noise is still drawn
        group-by-group in the original order, so default draw counts preserve
        the previous random stream.
        """
        n_groups = len(groups)
        errors = {}
        site_widths = {}
        for group in groups:
            for site in group["sites"]:
                errors[site] = np.zeros((clean.T, group["widths"][site]),
                                        dtype=np.float32)
                site_widths[site] = group["widths"][site]
        remaining = [int(group["draws"]) for group in groups]
        n_max = [min(int(self.config.chunk), n) for n in remaining]
        while any(n > 0 for n in remaining):
            counts = [min(n_max[i], remaining[i]) for i in range(n_groups)]
            total_n = sum(counts)
            if total_n <= 0:
                break
            # Allocate one zero jitter plane per active site.  Only the
            # group's own rows are non-zero, which makes the batch entries
            # mathematically independent as in the unbatched estimator.
            jitter = {site: np.zeros((total_n, clean.T, width),
                                     dtype=np.float32)
                      for site, width in site_widths.items()}
            group_noise = [dict() for _ in range(n_groups)]
            offset = 0
            for gi, group in enumerate(groups):
                n = counts[gi]
                if n <= 0:
                    continue
                noise = {}
                sigma = np.float32(group["sigma"])
                for site in group["sites"]:
                    values = rng.standard_normal(
                        (n, clean.T, group["widths"][site]), dtype=np.float32)
                    noise[site] = values
                    jitter[site][offset:offset + n] = values * sigma
                group_noise[gi] = noise
                offset += n
            x = np.broadcast_to(base_x, (total_n, clean.T, self.d))
            logits, _, _ = self.fw.run_blocks(
                start, x, jitter, fast=True)
            perturbed = per_token_cross_entropy(logits, clean.targets)
            stats["temp_bytes"] = max(
                stats["temp_bytes"], int(logits.nbytes + perturbed.nbytes))
            offset = 0
            for gi, group in enumerate(groups):
                n = counts[gi]
                if n <= 0:
                    continue
                n_draws = int(group["draws"])
                sigma = np.float32(group["sigma"])
                reward = clean.loss[None, :] - perturbed[offset:offset + n]
                if n > 1:
                    reward = reward - reward.mean(axis=0, keepdims=True)
                for site in group["sites"]:
                    errors[site] -= np.einsum(
                        "nt,ntd->td", reward, group_noise[gi][site],
                        optimize=True) / sigma / np.float32(n_draws)
                offset += n
            stats["block_evals"] += total_n * (self.L - int(start))
            stats["draws"] += total_n
            stats["draw_batches"] += 1
            stats["temp_bytes"] = max(
                stats["temp_bytes"],
                int(sum(a.nbytes for a in jitter.values())))
            for i in range(n_groups):
                remaining[i] -= counts[i]
        return errors

    def _head_errors(self, rng, clean, stats):
        cfg = self.config
        slab = min(int(cfg.head_slab), self.V)
        starts = list(range(0, self.V, slab))
        n_slabs = len(starts)
        per_slab = max(1, int(cfg.head_draws) // n_slabs)
        # The head is a pure logit/softmax estimator, so it can use float32
        # throughout once the clean logits have been max-shifted.  Keeping the
        # clean partition in float64 and the perturbed slab in float32 avoids
        # the expensive float64 exp while preserving a stable denominator.
        clean64 = clean.logits.astype(np.float64)
        m64 = np.max(clean64, axis=-1)
        exp_clean = np.exp(clean64 - m64[:, None])
        total32 = np.sum(exp_clean, axis=-1).astype(np.float32)
        m32 = m64.astype(np.float32)
        clean_target32 = clean64[
            np.arange(clean.T), clean.targets].astype(np.float32)
        clean_loss32 = clean.loss.astype(np.float32)
        rows = np.arange(clean.T)
        error = np.zeros((clean.T, self.V), dtype=np.float32)

        for lo in starts:
            hi = min(lo + slab, self.V)
            width = hi - lo
            slab_sum32 = np.sum(exp_clean[:, lo:hi], axis=-1).astype(np.float32)
            outside = total32 - slab_sum32
            clean_slab = clean.logits[:, lo:hi]
            in_slab = (clean.targets >= lo) & (clean.targets < hi)
            target_idx = np.clip(clean.targets - lo, 0, width - 1)
            target_clean = clean_slab[rows, target_idx]
            remaining = per_slab
            # Head rows are independent and cheap, so batch more of them than
            # the direct-output chunks.  At long sequence lengths the batch is
            # reduced to keep both transient head arrays near 4 MiB each;
            # otherwise the parallel head pass raises peak RSS substantially.
            budget_rows = max(1, (4 * 1024 * 1024 // 4) // (clean.T * width))
            n_max = min(per_slab, max(1, min(32, int(budget_rows))))
            while remaining > 0:
                n = min(n_max, remaining)
                noise = rng.standard_normal(
                    (n, clean.T, width), dtype=np.float32)
                sigma = np.float32(cfg.head_sigma)
                changed_exp = np.exp(
                    clean_slab[None, :, :] + sigma * noise
                    - m32[None, :, None])
                partition = outside[None, :] + changed_exp.sum(axis=-1)
                if np.any(partition <= 0):
                    raise FloatingPointError("degenerate head partition in Dust")
                log_part = m32[None, :] + np.log(partition)
                changed_target = target_clean[None, :] + sigma * noise[
                    :, rows, target_idx]
                target_logit = np.where(
                    in_slab[None, :], changed_target, clean_target32[None, :])
                loss = log_part - target_logit
                reward = clean_loss32[None, :] - loss
                if n > 1:
                    reward = reward - reward.mean(axis=0, keepdims=True)
                error[:, lo:hi] -= np.einsum(
                    "nt,ntd->td", reward / sigma, noise,
                    optimize=False)
                stats["head_column_evals"] += n * clean.T * width
                stats["head_draws"] += n
                stats["temp_bytes"] = max(
                    stats["temp_bytes"],
                    int(noise.nbytes + changed_exp.nbytes))
                remaining -= n
        error /= np.float32(per_slab)
        return error

    # -- attention internals ------------------------------------------------
    def _local_error_block(self, kind, layer, draws, sigma, target, rng,
                           clean, stats):
        """Estimated error at R, K or V for one block.

        ``target`` is the estimated error at the attention output, shape
        ``[T, d]``.  Q/gate gets its own token's score; K/V get future credit,
        because later tokens read the recurrent state.
        """
        cfg = self.config
        h = clean.h[layer]
        r = clean.r[layer]
        k = clean.k[layer]
        v = clean.v[layer]
        state = clean.state[layer]
        y = clean.y[layer]
        alpha = np.exp(-_softplus(self.p[f"L{layer}.attn.w"].data))
        u = self.p[f"L{layer}.attn.u"].data
        error = np.zeros((clean.T, self.d), dtype=np.float32)
        n_max = min(int(cfg.chunk), int(draws))
        remaining = int(draws)
        while remaining > 0:
            n = min(n_max, remaining)
            noise = rng.standard_normal((n, clean.T, self.d), dtype=np.float32)
            delta_scaled = noise * np.float32(sigma)
            if kind == "r":
                rp = r[None, :, :] + delta_scaled
                yp = _sigmoid_fast(rp) * (
                    state[None, :, :] + u * (k * v)[None, :, :])
            elif kind == "k":
                kp = k[None, :, :] + delta_scaled
                kvp = kp * v[None, :, :]
                sp = _scan(kvp, alpha)
                yp = _sigmoid_fast(r[None, :, :]) * (sp + u * kvp)
            elif kind == "v":
                vp = v[None, :, :] + delta_scaled
                kvp = k[None, :, :] * vp
                sp = _scan(kvp, alpha)
                yp = _sigmoid_fast(r[None, :, :]) * (sp + u * kvp)
            else:  # pragma: no cover - internal contract
                raise ValueError(f"unknown local kind {kind!r}")
            delta = yp - y[None, :, :]
            score = np.einsum("td,ntd->nt", target, delta, optimize=True)
            if kind != "r":
                score = future_credit(score, cfg.gamma)
            error += np.einsum(
                "nt,ntd->td", score, noise,
                optimize=True) / np.float32(sigma)
            stats["local_block_evals"] += n
            stats["local_draws"] += n
            stats["temp_bytes"] = max(
                stats["temp_bytes"],
                int(noise.nbytes + delta.nbytes + score.nbytes))
            remaining -= n
        error /= np.float32(draws)
        return error

    def _local_errors(self, kind, draws, sigma, targets, rng, clean, stats):
        return [self._local_error_block(
            kind, layer, draws, sigma, targets[layer], rng, clean, stats)
            for layer in range(self.L)]

    # -- non-matrix parameters ---------------------------------------------
    def _scalar_gradients(self, tokens, rng, clean, stats):
        draws = int(self.config.scalar_draws)
        if draws <= 0:
            return {}
        names = [name for name in self.p if name not in self._matrix_sites]
        if not names:
            return {}
        sigma = np.float32(self.config.scalar_sigma)
        clean_values = {name: self.p[name].data.copy() for name in names}
        grads = {name: np.zeros_like(clean_values[name]) for name in names}
        token_input = tokens
        try:
            for _ in range(draws):
                noise = {name: rng.standard_normal(
                    clean_values[name].shape, dtype=np.float32) for name in names}
                losses = []
                for sign in (1.0, -1.0):
                    for name in names:
                        self.p[name].data[...] = (
                            clean_values[name] + np.float32(sign) * sigma * noise[name])
                    losses.append(self.fw.loss_only(token_input))
                    stats["scalar_forwards"] += 1
                difference = (losses[0] - losses[1]) / (2.0 * float(sigma))
                if not math.isfinite(difference):
                    raise FloatingPointError("non-finite scalar ES loss difference")
                for name in names:
                    grads[name] += np.float32(difference / draws) * noise[name]
        finally:
            for name in names:
                self.p[name].data[...] = clean_values[name]
        return grads

    def _advance_to_head_stream(self, rng, T, cfg):
        """Advance a cloned RNG past the current serial pre-head draws.

        The head pass consumes the same generator after all direct, local, and
        embedding draws in the serial path.  Keeping that stream position makes
        the parallel head estimator byte-identical to the serial code; the
        actual direct passes can then use their own generator concurrently.
        """
        d = self.d
        ffn = self.spec.ffn_hidden

        def consume(draws, width, sites):
            remaining = int(draws)
            n_max = min(int(cfg.chunk), remaining)
            while remaining > 0:
                n = min(n_max, remaining)
                for _ in range(sites):
                    rng.standard_normal((n, T, width), dtype=np.float32)
                remaining -= n

        for _layer in range(self.L):
            # Match the combined-forward draw order used by _direct_groups:
            # groups are interleaved by chunk, not drawn group-by-group.
            groups = ((cfg.hub_draws, d, 1),
                      (cfg.writer_draws, d, 2),
                      (cfg.hidden_draws, ffn, 2))
            remaining = [int(draws) for draws, _, _ in groups]
            n_max = [min(int(cfg.chunk), n) for n in remaining]
            while any(n > 0 for n in remaining):
                counts = [min(n_max[i], remaining[i])
                          for i in range(len(groups))]
                if sum(counts) <= 0:  # pragma: no cover - validated config
                    break
                for i, (_, width, sites) in enumerate(groups):
                    n = counts[i]
                    if n <= 0:
                        continue
                    for _ in range(sites):
                        rng.standard_normal((n, T, width), dtype=np.float32)
                for i in range(len(groups)):
                    remaining[i] -= counts[i]
            if cfg.local_draws > 0:
                for _kind in range(3):
                    consume(cfg.local_draws, d, 1)
        consume(cfg.embed_draws, d, 1)

    # -- public API ---------------------------------------------------------
    def loss_and_gradients(self, tokens, step=0):
        cfg = self.config
        cfg.validate(self.V)
        started = time.perf_counter()
        clean = self.fw.capture(tokens)
        clean_s = time.perf_counter() - started
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), int(step)]))
        stats = {"block_evals": 0, "local_block_evals": 0, "head_column_evals": 0,
                 "draws": 0, "local_draws": 0, "head_draws": 0,
                 "scalar_forwards": 0, "draw_batches": 0, "temp_bytes": 0}
        grads = {}
        inv_t = np.float32(1.0 / clean.T)
        phase = {"direct": 0.0, "local": 0.0, "reassemble": 0.0, "scalar": 0.0}
        peak_error_bytes = 0

        # The head estimator depends only on the clean logits and its own
        # random draws, not on the recurrent block perturbations.  For the
        # default head budget, overlap it with the direct/local passes.  Use a
        # dedicated deterministic stream so scheduling cannot affect results.
        head_future = None
        head_executor = None
        head_rng = None
        head_stats = None
        if cfg.head_draws >= 256 and (os.cpu_count() or 1) > 1:
            head_stats = {"head_column_evals": 0, "head_draws": 0,
                          "temp_bytes": 0}
            head_rng = np.random.default_rng(
                np.random.SeedSequence([int(cfg.seed), int(step)]))
            self._advance_to_head_stream(head_rng, clean.T, cfg)
            head_executor = ThreadPoolExecutor(max_workers=1)
            head_future = head_executor.submit(
                self._head_errors, head_rng, clean, head_stats)

        try:
            # Per block: estimate the attention-output target, then the two
            # writer sites, the two FFN hidden sites and the three attention
            # internals.  Reassemble and release each block's error arrays
            # before moving on, so the estimator's persistent footprint is one
            # block, not all L blocks.
            for layer in range(self.L):
                t0 = time.perf_counter()
                sigma_hub = (cfg.hub_sigma if layer < self.L // 2
                             else cfg.hub_sigma_deep)
                hub_key = f"L{layer}.attn.hub"
                wo_key = f"L{layer}.attn.out"
                wc_key = f"L{layer}.ffn.out"
                wa_key = f"L{layer}.ffn.a"
                wb_key = f"L{layer}.ffn.b"
                combined = self._direct_groups(
                    clean.x_in[layer], layer, [
                        {"sites": [hub_key], "widths": {hub_key: self.d},
                         "draws": cfg.hub_draws, "sigma": sigma_hub},
                        {"sites": [wo_key, wc_key],
                         "widths": {wo_key: self.d, wc_key: self.d},
                         "draws": cfg.writer_draws, "sigma": cfg.sigma},
                        {"sites": [wa_key, wb_key],
                         "widths": {wa_key: self.spec.ffn_hidden,
                                    wb_key: self.spec.ffn_hidden},
                         "draws": cfg.hidden_draws, "sigma": cfg.sigma},
                    ], rng, clean, stats)
                hub_err = combined[hub_key]
                writer = {wo_key: combined[wo_key], wc_key: combined[wc_key]}
                hidden = {wa_key: combined[wa_key], wb_key: combined[wb_key]}
                phase["direct"] += time.perf_counter() - t0

                local = {}
                if cfg.local_draws > 0:
                    t0 = time.perf_counter()
                    for kind in ("r", "k", "v"):
                        local[kind] = self._local_error_block(
                            kind, layer, cfg.local_draws, cfg.local_sigma,
                            hub_err, rng, clean, stats)
                    phase["local"] += time.perf_counter() - t0

                block_bytes = int(hub_err.nbytes)
                block_bytes += int(sum(e.nbytes for e in writer.values()))
                block_bytes += int(sum(e.nbytes for e in hidden.values()))
                block_bytes += int(sum(e.nbytes for e in local.values()))
                peak_error_bytes = max(peak_error_bytes, block_bytes)

                t0 = time.perf_counter()
                grads[f"L{layer}.attn.Wo"] = (
                    clean.y[layer].T @ writer[wo_key] * inv_t)
                grads[f"L{layer}.ffn.Wc"] = (
                    clean.z[layer].T @ writer[wc_key] * inv_t)
                grads[f"L{layer}.ffn.Wa"] = (
                    clean.h2[layer].T @ hidden[wa_key] * inv_t)
                grads[f"L{layer}.ffn.Wb"] = (
                    clean.h2[layer].T @ hidden[wb_key] * inv_t)
                if local:
                    grads[f"L{layer}.attn.Wr"] = (
                        clean.h[layer].T @ local["r"] * inv_t)
                    grads[f"L{layer}.attn.Wk"] = (
                        clean.h[layer].T @ local["k"] * inv_t)
                    grads[f"L{layer}.attn.Wv"] = (
                        clean.h[layer].T @ local["v"] * inv_t)
                phase["reassemble"] += time.perf_counter() - t0

            # Token embedding; the head pass is already running in parallel
            # when the default head budget is large enough.
            t0 = time.perf_counter()
            embed_err = self._direct_group(
                clean.x0, 0, ["embed"], {"embed": self.d},
                cfg.embed_draws, cfg.sigma, rng, clean, stats)["embed"]
            if head_future is not None:
                head_err = head_future.result()
                for key in ("head_column_evals", "head_draws"):
                    stats[key] += head_stats[key]
                stats["temp_bytes"] = max(
                    stats["temp_bytes"], head_stats["temp_bytes"])
            else:
                head_err = self._head_errors(rng, clean, stats)
            phase["direct"] += time.perf_counter() - t0
        finally:
            if head_executor is not None:
                head_executor.shutdown(wait=True)

        peak_error_bytes = max(
            peak_error_bytes, int(embed_err.nbytes + head_err.nbytes))
        t0 = time.perf_counter()
        embed_grad = np.zeros_like(self.p["embed"].data)
        np.add.at(embed_grad, clean.tok, embed_err)
        embed_grad += head_err.T @ clean.final_h
        grads["embed"] = embed_grad * inv_t
        phase["reassemble"] += time.perf_counter() - t0
        del embed_err, head_err, embed_grad

        # Optional two-sided ES for non-matrix parameters (the paper's
        # fallback for residual mixing scalars, generalized here).
        t0 = time.perf_counter()
        scalar = self._scalar_gradients(
            np.asarray(tokens, np.int64),
            head_rng if head_future is not None else rng, clean, stats)
        phase["scalar"] = time.perf_counter() - t0
        for name, value in scalar.items():
            grads[name] = value.astype(np.float32)

        for name, value in grads.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"non-finite Dust estimate for {name}")

        frozen = sorted(name for name in self.p if name not in grads)
        error_bytes = int(peak_error_bytes)
        info = {
            "algorithm": "dust",
            "dust": {
                "step": int(step),
                "config": cfg.to_dict(),
                "loss": float(np.mean(clean.loss)),
                "clean_s": clean_s,
                "direct_s": phase["direct"],
                "local_s": phase["local"],
                "scalar_s": phase["scalar"],
                "reassemble_s": phase["reassemble"],
                "total_s": time.perf_counter() - started,
                "clean_cache_bytes": clean.nbytes,
                "estimator_bytes": error_bytes,
                "temp_bytes": int(stats["temp_bytes"]),
                "block_evals": int(stats["block_evals"]),
                "local_block_evals": int(stats["local_block_evals"]),
                "head_column_evals": int(stats["head_column_evals"]),
                "full_forward_equivalents": float(
                    (stats["block_evals"] + stats["local_block_evals"])
                    / max(1, self.L) + stats["scalar_forwards"]),
                "draws": int(stats["draws"]),
                "local_draws": int(stats["local_draws"]),
                "head_draws": int(stats["head_draws"]),
                "scalar_forwards": int(stats["scalar_forwards"]),
                "frozen_params": frozen,
            },
        }
        return float(np.mean(clean.loss)), grads, info


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def _array_of(value):
    """Return the float array behind a Tensor or a plain ndarray.

    ``numpy.ndarray`` also exposes a ``.data`` attribute, but it is a memoryview;
    using it by mistake makes standalone optimizer updates silent no-ops.
    """
    data = getattr(value, "data", None)
    return data if isinstance(data, np.ndarray) else value


class DustSGD:
    """SGD with momentum, matching the paper's main-comparison optimizer."""

    def __init__(self, lr=0.05, momentum=0.9, weight_decay=0.0):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError("DustSGD lr must be finite and > 0")
        if not math.isfinite(self.momentum) or not 0.0 <= self.momentum < 1.0:
            raise ValueError("DustSGD momentum must be finite and in [0, 1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("DustSGD weight_decay must be finite and >= 0")
        self.state = {}
        self.n_updates = 0

    def register(self, params):
        for name, value in params.items():
            if name not in self.state:
                self.state[name] = {
                    "m": np.zeros_like(_array_of(value), dtype=np.float32)}

    def step(self, params, grads, collect_norms=False):
        # Validate the whole update before touching a single parameter.  A
        # failed estimate must not leave the model half-updated.
        prepared = []
        for name, grad in grads.items():
            if name not in params:
                raise ValueError(f"DustSGD has no parameter {name!r}")
            value = params[name]
            w = _array_of(value)
            g = np.ascontiguousarray(grad, dtype=np.float32)
            if g.shape != np.shape(w):
                raise ValueError(
                    f"DustSGD gradient shape {g.shape} != parameter {name!r} "
                    f"shape {np.shape(w)}")
            if not np.isfinite(g).all():
                raise FloatingPointError(
                    f"non-finite Dust gradient for {name!r}; update rejected")
            prepared.append((name, w, g))
        norms = {}
        lr = np.float32(self.lr)
        mom = np.float32(self.momentum)
        decay = np.float32(1.0 - self.lr * self.weight_decay) \
            if self.weight_decay else None
        for name, w, g in prepared:
            if name not in self.state:
                self.state[name] = {"m": np.zeros_like(w, dtype=np.float32)}
            m = self.state[name]["m"]
            if decay is not None:
                w *= decay
            m *= mom
            m += g
            w -= lr * m
            if collect_norms:
                norms[name] = float(np.sqrt(np.mean(m * m, dtype=np.float64)))
        self.n_updates += 1
        return norms

    def state_dict(self):
        return {name: {"m": state["m"].copy()}
                for name, state in self.state.items()}

    def load_state_dict(self, state, factored_flags=None, copy=True):
        self.state = {}
        for name, entry in state.items():
            if "m" not in entry:
                raise ValueError(f"DustSGD state for {name!r} has no momentum buffer")
            m = np.asarray(entry["m"], dtype=np.float32)
            if not np.isfinite(m).all():
                raise FloatingPointError(
                    f"non-finite DustSGD momentum for {name!r}")
            self.state[name] = {"m": m.copy() if copy else m}
