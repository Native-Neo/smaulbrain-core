"""NumPy SmaulBRAIN: byte-level recurrent linear attention + shared MoE pool."""
from __future__ import annotations

import math

import numpy as np

from . import config as _cfg, moe as _moe, ops as _ops, quant as _quant
from . import npgrad as ng
from .npgrad import Parameter, Tensor, as_tensor


def build_params(spec, generator=None, n_experts: int = -1, device="cpu", mtp_depth: int = 0):
    if generator is None:
        rng = np.random.default_rng(0)
    elif isinstance(generator, np.random.Generator):
        rng = generator
    elif isinstance(generator, int):
        rng = np.random.default_rng(generator)
    elif hasattr(generator, "integers"):
        rng = np.random.default_rng(int(generator.integers(0, 2**63 - 1)))
    else:
        raise TypeError("generator must be a numpy Generator, integer seed, or legacy generator")
    d = spec.d_model
    p = {}
    s = 1.0 / math.sqrt(d)

    def rand(shape, scale=s):
        return Parameter((rng.standard_normal(shape) * scale).astype(np.float32))

    p["embed"] = rand((spec.vocab_size, d))
    p["final_norm"] = Parameter(np.ones(d, dtype=np.float32))
    for i in range(spec.n_layers):
        px = f"L{i}."
        for key in ("Wr", "Wk", "Wv", "Wo"):
            p[px + "attn." + key] = rand((d, d))
        p[px + "attn.w"] = rand((d,), 0.1)
        p[px + "attn.u"] = Parameter(np.zeros(d, dtype=np.float32))
        p[px + "n1"] = Parameter(np.ones(d, dtype=np.float32))
        p[px + "n2"] = Parameter(np.ones(d, dtype=np.float32))
        if spec.kind == "dense":
            h = spec.ffn_hidden
            p[px + "ffn.Wa"] = rand((d, h))
            p[px + "ffn.Wb"] = rand((d, h))
            p[px + "ffn.Wc"] = rand((h, d), 1.0 / math.sqrt(h))
    if spec.kind == "moe":
        n = spec.n_experts_init if n_experts < 0 else n_experts
        p["router"] = rand((d, n))
        for eid in range(n):
            p[f"expert.{eid}.Win"] = rand((d, spec.expert_hidden))
            p[f"expert.{eid}.Wout"] = rand((spec.expert_hidden, d), 1.0 / math.sqrt(spec.expert_hidden))
    for k in range(1, max(0, int(mtp_depth)) + 1):
        px = f"mtp.{k}."
        p[px + "hn"] = Parameter(np.ones(d, dtype=np.float32))
        p[px + "en"] = Parameter(np.ones(d, dtype=np.float32))
        p[px + "proj"] = rand((2 * d, d), 1.0 / math.sqrt(2 * d))
        p[px + "n1"] = Parameter(np.ones(d, dtype=np.float32))
        p[px + "on"] = Parameter(np.ones(d, dtype=np.float32))
        for key in ("Wr", "Wk", "Wv", "Wo"):
            p[px + "attn." + key] = rand((d, d))
        p[px + "attn.w"] = rand((d,), 0.1)
        p[px + "attn.u"] = Parameter(np.zeros(d, dtype=np.float32))
    return p


def count_params_dict(p):
    return int(sum(v.numel() if isinstance(v, Tensor) else np.asarray(v).size for v in p.values()))


def expert_ids(p):
    return sorted({int(k.split(".")[1]) for k in p if k.startswith("expert.")})


class PackedCache:
    def __init__(self, max_entries=256):
        self.max_entries = max_entries
        self.d = {}
        self.reuse_hits = 0
        self.builds = 0

    def get(self, key, W):
        entry = self.d.get(key)
        if entry is not None and entry["ref"] is W:
            self.reuse_hits += 1
            return entry["packed"], entry["scale"], entry["shape"]
        packed, scale, shape, _ = _quant.pack_ternary_tiled(W.data)
        if key not in self.d and self.max_entries > 0 and len(self.d) >= self.max_entries:
            self.d.pop(next(iter(self.d)))
        self.d[key] = {"packed": packed, "scale": scale, "shape": shape, "ref": W}
        self.builds += 1
        return packed, scale, shape

    def invalidate(self, keys=None):
        if keys is None:
            self.d.clear()
        else:
            for key in keys:
                self.d.pop(key, None)


class FP8Cache:
    def __init__(self, max_entries=256):
        self.max_entries = max_entries
        self.d = {}
        self.builds = 0
        self.reuse_hits = 0

    def get(self, key, W):
        entry = self.d.get(key)
        if entry is not None and entry["ref"] is W:
            self.reuse_hits += 1
            return entry["weight"], entry["scale"]
        Wq, scale = _quant.fp8_e4m3_quantize(W.data)
        if key not in self.d and self.max_entries > 0 and len(self.d) >= self.max_entries:
            self.d.pop(next(iter(self.d)))
        self.d[key] = {"weight": Wq, "scale": scale, "ref": W}
        self.builds += 1
        return Wq, scale

    def invalidate(self, keys=None):
        if keys is None:
            self.d.clear()
        else:
            for key in keys:
                self.d.pop(key, None)


def _weight_grad(X, dY):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if dY.ndim == 1:
        dY = dY.reshape(1, -1)
    return X.data.T @ dY.data


def _quantized_linear(X, W, Wq, kind, backward_mode, packed, scale, shape):
    X = as_tensor(X)
    W = as_tensor(W)
    if kind == "fp32":
        return X @ W
    if kind == "fp8":
        out = X.data @ Wq
    else:
        out = _quant.packed_matmat(X.data, packed, shape, scale)

    def bw(g):
        if kind == "fp8":
            if backward_mode == "fp8":
                dYq, _ = _quant.fp8_e4m3_quantize(g)
                Xq, _ = _quant.fp8_e4m3_quantize(X.data)
                return dYq @ Wq.T, _weight_grad(Xq, dYq)
            return g @ Wq.T, _weight_grad(X, g)
        dX = _quant.packed_matmat_transpose(g, packed, shape, scale)
        if backward_mode == "fp8":
            dYq, _ = _quant.fp8_e4m3_quantize(g)
            Xq, _ = _quant.fp8_e4m3_quantize(X.data)
            dW = _weight_grad(Xq, dYq)
        else:
            dW = _weight_grad(X, g)
        return dX, dW

    return ng.make_op(out, (X, W), bw)


class _QuantLinear:
    @staticmethod
    def apply(X, W, Wq, kind, backward_mode, packed, scale, shape):
        return _quantized_linear(X, W, Wq, kind, backward_mode, packed, scale, shape)


class Model:
    def __init__(self, spec, params: dict, forward_quant="fp32", backward_mode="fp32",
                 paging_mode="R", paging_store=None, loss_smooth_beta=0.99,
                 capacity_factor: float = 1.25, device="cpu",
                 mtp_depth: int = 0, mtp_loss_weight: float = 0.1):
        _cfg.validate_quant_combo(forward_quant, backward_mode)
        self.spec = spec
        self.device = str(device)
        self.param_names = []
        self.params = []
        self._pd = None
        self.p = params
        self.mtp_depth = int(mtp_depth)
        self.mtp_loss_weight = float(mtp_loss_weight)
        if self.mtp_depth < 0:
            raise ValueError("mtp_depth must be >= 0")
        if not (self.mtp_loss_weight >= 0) or self.mtp_loss_weight != self.mtp_loss_weight \
                or self.mtp_loss_weight in (float("inf"), float("-inf")):
            raise ValueError("mtp_loss_weight must be a finite non-negative number")
        self.forward_quant = forward_quant
        self.backward_mode = backward_mode
        self.paging_mode = paging_mode
        self.store = paging_store
        self.packed = PackedCache()
        self.fp8 = FP8Cache()
        self.loss_smooth_beta = float(loss_smooth_beta)
        self.capacity_factor = float(capacity_factor)
        self._last_aux = {"overflow_dropped": 0, "balance_aux": 0.0}
        self._paged_leaves = {}
        if forward_quant == "ternary":
            _quant.prepare_ternary_backend()
        self._assert_counts()

    # -- parameter plumbing -------------------------------------------------
    @property
    def p(self):
        if self._pd is None:
            self._pd = {k: v for k, v in zip(self.param_names, self.params)}
        return self._pd

    @p.setter
    def p(self, values):
        names = []
        vals = []
        for k, v in values.items():
            data = v.data if isinstance(v, Tensor) else np.asarray(v)
            data = np.ascontiguousarray(data, dtype=np.float32)
            vals.append(Parameter(data))
            names.append(k)
        self.param_names = names
        self.params = vals
        self._pd = None

    def _assert_counts(self):
        p = self.p
        if self.spec.kind == "dense":
            n = count_params_dict(p)
            exp = _cfg.EXPECTED_DENSE_TOTAL[self.spec.name] + _cfg.mtp_param_count(self.spec, self.mtp_depth)
            if n != exp:
                raise ValueError(f"preset {self.spec.name}: got {n} params, expected {exp}")
        else:
            for eid in expert_ids(p):
                n = p[f"expert.{eid}.Win"].numel() + p[f"expert.{eid}.Wout"].numel()
                exp = _cfg.EXPECTED_EXPERT_PARAMS[self.spec.name]
                if n != exp:
                    raise ValueError(f"preset {self.spec.name} expert {eid}: got {n}, expected {exp}")
        self._assert_mtp_counts()

    def _assert_mtp_counts(self):
        have = sorted({int(k.split(".")[1]) for k in self.p if k.startswith("mtp.")})
        want = list(range(1, self.mtp_depth + 1))
        if have != want:
            raise ValueError(f"MTP params {have or '[]'} do not match mtp_depth={self.mtp_depth}")
        n = sum(v.numel() for k, v in self.p.items() if k.startswith("mtp."))
        exp = _cfg.mtp_param_count(self.spec, self.mtp_depth)
        if n != exp:
            raise ValueError(f"MTP params: got {n}, expected {exp}")

    def report(self):
        n = len(expert_ids(self.p)) if self.spec.kind == "moe" else 0
        rep = _cfg.arch_report(self.spec, n, mtp_depth=self.mtp_depth)
        rep.update(forward_quant=self.forward_quant, backward_mode=self.backward_mode,
                   paging_mode=self.paging_mode)
        return rep

    def zero_grad(self, set_to_none=True):
        for value in self.p.values():
            if set_to_none:
                value.grad = None
            elif value.grad is not None:
                value.grad[...] = 0

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    # -- primitives ---------------------------------------------------------
    def _linear(self, X, W, key):
        kind = self.forward_quant
        if kind == "fp32":
            return X @ W
        Wq = np.empty(0, dtype=np.float32)
        packed = None
        scale = 1.0
        shape = tuple(W.shape)
        if kind == "fp8":
            Wq, scale = self.fp8.get(key, W)
        else:
            packed, scale, shape = self.packed.get(key, W)
        return _QuantLinear.apply(X, W, Wq, kind, self.backward_mode, packed, scale, shape)

    def _expert_weights(self, eid):
        if self.store is None or self.store.mode in ("R", "R2V", "V"):
            return {"Win": self.p[f"expert.{eid}.Win"], "Wout": self.p[f"expert.{eid}.Wout"]}
        loaded = self.store.get(eid)
        if ng.is_grad_enabled():
            out = {}
            for suffix in ("Win", "Wout"):
                key = (eid, suffix)
                if key not in self._paged_leaves:
                    self._paged_leaves[key] = Parameter(np.array(loaded[suffix], dtype=np.float32, copy=True))
                out[suffix] = self._paged_leaves[key]
            return out
        return {k: Parameter(np.array(v, dtype=np.float32, copy=True)) for k, v in loaded.items()}

    def _attn(self, X, px, s0, capture=False):
        p = self.p
        R = self._linear(X, p[px + "attn.Wr"], px + "attn.Wr")
        K = self._linear(X, p[px + "attn.Wk"], px + "attn.Wk")
        V = self._linear(X, p[px + "attn.Wv"], px + "attn.Wv")
        alpha = ng.exp(-ng.softplus(p[px + "attn.w"]))
        initial = None if s0 is None else as_tensor(s0).float()
        kv = K * V
        if X.shape[0] == 0:
            S = Tensor(np.zeros((0, X.shape[-1]), dtype=np.float32))
            state = Tensor(np.zeros_like(alpha.data))
        else:
            S = ng.scan(kv, alpha, initial)
            state = S[-1]
        Y = ng.sigmoid(R) * (S + ng.mul(p[px + "attn.u"], kv))
        out = self._linear(Y, p[px + "attn.Wo"], px + "attn.Wo")
        if capture:
            return out, state, S
        return out, state

    def _ffn(self, X, px):
        p = self.p
        A = self._linear(X, p[px + "ffn.Wa"], px + "ffn.Wa")
        B = self._linear(X, p[px + "ffn.Wb"], px + "ffn.Wb")
        return self._linear(ng.silu(A) * B, p[px + "ffn.Wc"], px + "ffn.Wc")

    def _expert(self, X, eid):
        W = self._expert_weights(eid)
        return self._linear(ng.gelu(self._linear(X, W["Win"], f"expert.{eid}.Win")),
                            W["Wout"], f"expert.{eid}.Wout")

    def _route(self, X, top_k, capacity_factor, disabled):
        W = self.p["router"]
        scores = X @ W
        select_scores = scores.data
        disabled = sorted(set(int(i) for i in disabled))
        if disabled:
            select_scores = select_scores.copy()
            select_scores[:, disabled] = -np.inf
        idx = np.argsort(-select_scores, axis=1, kind="stable")[:, :top_k]
        top_scores = ng.gather_last(scores, idx)
        raw_weights = ng.softmax(top_scores, axis=1)
        T = X.shape[0]
        flat = idx.reshape(-1)
        cap = flat.size if capacity_factor is None else max(
            1, math.ceil(capacity_factor * T * top_k / W.shape[1]))
        order = np.argsort(flat, kind="stable")
        sorted_flat = flat[order]
        counts = np.bincount(sorted_flat, minlength=W.shape[1])
        starts = np.cumsum(counts) - counts
        rank = np.arange(flat.size) - starts[sorted_flat]
        keep = np.ones(flat.shape, dtype=np.float32)
        keep[order[rank >= cap]] = 0.0
        keep = keep.reshape(idx.shape)
        kept = raw_weights * keep
        denom = kept.sum(axis=1, keepdim=True)
        weights = kept / ng.clamp(denom, 1e-12, None)
        load = np.minimum(counts, cap)
        info = {"overflow_dropped": int(flat.size - load.sum()), "cap_per_expert": cap,
                "load": load.tolist()}
        return idx, weights, info

    def _moe(self, X, disabled=(), no_drop=False):
        p = self.p
        top_k = self.spec.top_k
        idx, weights, info = self._route(
            X, top_k, None if no_drop else self.capacity_factor, disabled)
        Y = Tensor(np.zeros_like(X.data))
        flat = idx.reshape(-1)
        order = np.argsort(flat, kind="stable")
        counts = np.bincount(flat[order], minlength=p["router"].shape[1])
        start = 0
        for eid, cnt in enumerate(counts.tolist()):
            if cnt == 0:
                continue
            pos = order[start:start + cnt]
            start += cnt
            rows = pos // top_k
            cols = pos % top_k
            if self.store is not None:
                self.store.pin(eid)
            try:
                yb = self._expert(X[rows], eid)
                wt = weights[rows, cols].reshape(-1, 1)
                Y = ng.index_add_rows(Y, rows, wt * yb)
            finally:
                if self.store is not None:
                    self.store.unpin(eid)
        return Y, info

    @staticmethod
    def _rmsnorm(X, gain):
        return X * ng.rsqrt(X.square().mean(dim=-1, keepdim=True) + _ops.EPS) * gain

    def _mtp_module(self, k, H, tok_in, state=None):
        p = self.p
        H = as_tensor(H)
        px = f"mtp.{k}."
        hn = self._rmsnorm(H, p[px + "hn"])
        en = self._rmsnorm(p["embed"][np.asarray(tok_in, dtype=np.int64)], p[px + "en"])
        U = self._linear(ng.concat([hn, en], axis=-1), p[px + "proj"], px + "proj")
        A = self._rmsnorm(U, p[px + "n1"])
        O, s = self._attn(A, px, state)
        Y = U + O
        logits = self._rmsnorm(Y, p[px + "on"]) @ ng.transpose(p["embed"])
        return logits, Y, s

    def _mtp_losses(self, X, tok):
        losses = []
        H = X
        for k in range(1, self.mtp_depth + 1):
            L = tok.size - 1 - k
            if L <= 0:
                break
            logits, Y, _s = self._mtp_module(k, H[:L], tok[k:k + L])
            losses.append(ng.cross_entropy(logits, tok[k + 1:k + 1 + L]))
            H = Y
        return losses

    def _forward_impl(self, tokens, states=None, disabled=(), return_hidden=False,
                      capture_states=False, exact_routing=False):
        p = self.p
        tok = np.asarray(tokens.data if isinstance(tokens, Tensor) else tokens, dtype=np.int64).reshape(-1)
        X = p["embed"][tok]
        new_states = [] if states is None else states
        aux = {"overflow_dropped": 0, "balance_aux": 0.0}
        hist = [] if capture_states else None
        for i in range(self.spec.n_layers):
            px = f"L{i}."
            s0 = None if states is None else states[i]
            h = self._rmsnorm(X, p[px + "n1"])
            if capture_states:
                o, s1, S = self._attn(h, px, s0, capture=True)
                hist.append(S.data.copy())
            else:
                o, s1 = self._attn(h, px, s0)
            X = X + o
            h2 = self._rmsnorm(X, p[px + "n2"])
            if self.spec.kind == "dense":
                f = self._ffn(h2, px)
            else:
                f, info = self._moe(h2, disabled, no_drop=exact_routing)
                aux["overflow_dropped"] += info["overflow_dropped"]
            X = X + f
            if states is None:
                new_states.append(s1.detach().data)
            else:
                states[i] = s1.detach().data
        H = self._rmsnorm(X, p["final_norm"])
        logits = H @ ng.transpose(p["embed"])
        self._last_aux = aux
        out_states = states if states is not None else new_states
        if return_hidden or capture_states:
            return logits, out_states, {"hidden": X, "state_hist": hist}
        return logits, out_states

    def forward(self, tokens, states=None, train=False, disabled=()):
        if train:
            return self._forward_impl(tokens, states, disabled)[0], self._last_aux
        with ng.no_grad():
            return self._forward_impl(tokens, states, disabled)

    def loss_and_backward(self, tokens):
        tok = np.asarray(tokens.data if isinstance(tokens, Tensor) else tokens, dtype=np.int64).reshape(-1)
        if tok.size < 2:
            raise ValueError("training sequence needs at least two tokens")
        self.zero_grad(set_to_none=True)
        self._paged_leaves = {}
        mtp_losses = []
        with ng.enable_grad():
            logits, _, det = self._forward_impl(tok[:-1], return_hidden=True)
            loss_main = ng.cross_entropy(logits, tok[1:])
            loss = loss_main
            if self.mtp_depth > 0:
                for lk in self._mtp_losses(det["hidden"], tok):
                    mtp_losses.append(float(lk.detach().item()))
                    loss = loss + self.mtp_loss_weight * lk
            loss.backward()
        p = self.p
        for (eid, suffix), leaf in self._paged_leaves.items():
            target = p[f"expert.{eid}.{suffix}"]
            if leaf.grad is not None:
                if target.grad is None:
                    target.grad = leaf.grad.copy()
                else:
                    target.grad += leaf.grad
        grads = {k: v.grad.copy() for k, v in p.items() if v.grad is not None}
        out = float(loss.detach().item())
        aux = dict(self._last_aux)
        aux["loss_main"] = float(loss_main.detach().item())
        aux["mtp_losses"] = mtp_losses
        del logits, det, loss, loss_main
        self._paged_leaves = {}
        self.zero_grad(set_to_none=True)
        return out, grads, aux

    def init_states(self):
        return [np.zeros(self.spec.d_model, dtype=np.float32) for _ in range(self.spec.n_layers)]

    @staticmethod
    def serialize_states(states):
        return {str(i): np.asarray(s).tolist() for i, s in enumerate(states)}

    @staticmethod
    def restore_states(blob):
        return [np.asarray(blob[str(i)], dtype=np.float32) for i in range(len(blob))]

    def step_token(self, tok, states):
        logits, _ = self.forward([tok], states=states, train=False)
        return logits[0]

    def hidden_forward(self, tokens, states=None):
        with ng.no_grad():
            logits, _, det = self._forward_impl(tokens, states=states, return_hidden=True)
        return logits, det["hidden"]

    def step_details(self, tok, states):
        logits, hidden = self.hidden_forward([tok], states)
        return logits[0], hidden[0]

    def verify_tokens(self, tokens, states):
        with ng.no_grad():
            _logits, _, det = self._forward_impl(tokens, states=states, return_hidden=True,
                                                 capture_states=True, exact_routing=True)
            hidden = det["hidden"]
            emb = self.p["embed"]
            H = self._rmsnorm(hidden, self.p["final_norm"])
            rows = [H[j] @ ng.transpose(emb) for j in range(H.shape[0])]
            logits = ng.stack(rows) if rows else Tensor(np.zeros((0, emb.shape[0]), dtype=np.float32))
            return logits, hidden, det["state_hist"]

    def mtp_step(self, k, h_prev, tok_id, state):
        tok_in = np.asarray([int(tok_id)], dtype=np.int64)
        with ng.no_grad():
            logits, Y, s = self._mtp_module(k, as_tensor(h_prev).reshape(1, -1), tok_in, state=state)
        return logits[0], Y[0], s
