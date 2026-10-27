"""Inference engine using NumPy arrays and persistent recurrent state.

State seam: ``prefill(tokens)`` consumes the prompt and leaves the pending
distribution for the next token in ``_pending_logits``; ``generate_stream``
consumes that pending distribution instead of re-stepping the prompt's last
token. ``prefill(p) + generate([])`` is therefore token-identical to
``generate(p)`` for both plain and speculative decoding. Only the purely
degenerate case (fresh engine, empty prompt, no history) keeps the historical
byte-0 seed. Speculative decoding (opt-in via ``speculative=K``) uses the
model's MTP modules to draft tokens after the normally sampled one, verifies
them with a single batched main-model forward, and accepts the matching
prefix. Sampling and acceptance math are unchanged from the PyTorch
implementation.
"""
from __future__ import annotations

import numpy as np

from .npgrad import Tensor


def _arr(value):
    return np.asarray(value.data if isinstance(value, Tensor) else value)


class InferenceEngine:
    _WINDOW = 256   # max positions processed per batched MTP catch-up pass
    _RING = 2048    # per-depth ring buffer size (hidden/state history)

    def __init__(self, model, seed=0, temperature=1.0, top_k=0, top_p=1.0, speculative=0):
        self.model = model
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        if not 0 < self.top_p <= 1:
            raise ValueError("--top-p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("--top-k must be >= 0")
        if not self.temperature >= 0:
            raise ValueError("--temperature must be >= 0")
        self.mtp_depth = int(getattr(model, "mtp_depth", 0) or 0)
        self.speculative = int(speculative)
        if self.speculative < 0:
            raise ValueError("speculative must be >= 0")
        if self.speculative > self.mtp_depth:
            raise ValueError(f"speculative={self.speculative} exceeds model mtp_depth={self.mtp_depth}")
        self.draft_depth = min(self.speculative, self.mtp_depth)
        self.generator = np.random.default_rng(int(seed))
        self.states = model.init_states()
        self._reset_mtp()
        self._pending_logits = None
        self._last_plain_token = None

    def reset(self):
        self.states = self.model.init_states()
        self._reset_mtp()
        self._pending_logits = None
        self._last_plain_token = None

    # -- persistent state (main recurrent layers + optional MTP pipeline) --
    def export_states(self):
        blob = self.model.serialize_states(self.states)
        if self._mtp_ready:
            base = max(0, (min(self._mtp_pos) if self._mtp_pos else 0) - 1)
            lo = max(0, base - self._tok_base)
            blob["mtp"] = {"base": int(base),
                           "tokens": list(self._toks[lo:]),
                           "h0": [[int(p), _arr(v).tolist()] for p, v in self._h0 if p >= base],
                           "h": [[[int(p), _arr(v).tolist()] for p, v in ring if p >= base]
                                 for ring in self._h],
                           "states": [None if s is None else _arr(s).tolist() for s in self._mtp_states],
                           "pos": list(self._mtp_pos),
                           "unconsumed": bool(self._unconsumed)}
        return blob

    def import_states(self, blob):
        main = {k: v for k, v in blob.items() if k != "mtp"}
        self.states = [np.asarray(s, dtype=np.float32) for s in self.model.restore_states(main)]
        self._reset_mtp()
        self._pending_logits = None
        self._last_plain_token = None
        mtp = blob.get("mtp")
        if mtp is None or not self._mtp_ready:
            self._mtp_ready = False
            return
        self._tok_base = int(mtp["base"])
        self._toks = [int(t) for t in mtp["tokens"]]
        self._h0 = [(int(p), np.asarray(v, dtype=np.float32)) for p, v in mtp["h0"]]
        self._h = [[(int(p), np.asarray(v, dtype=np.float32)) for p, v in ring] for ring in mtp["h"]]
        self._mtp_states = [None if s is None else np.asarray(s, dtype=np.float32) for s in mtp["states"]]
        self._mtp_pos = [int(p) for p in mtp["pos"]]
        pad = self._mtp_pos[-1] if self._mtp_pos else 0
        k = self.draft_depth
        self._mtp_states = (self._mtp_states + [None] * k)[:k]
        self._h = (self._h + [[] for _ in range(k)])[:k]
        self._mtp_pos = (self._mtp_pos + [pad] * k)[:k]
        self._unconsumed = bool(mtp.get("unconsumed", True))

    def _reset_mtp(self):
        k = self.draft_depth
        self._mtp_ready = bool(k > 0)
        self._mtp_K = k
        self._mtp_states = [None] * k
        self._mtp_pos = [0] * k
        self._tok_base = 0
        self._toks = []
        self._h0 = []
        self._h = [[] for _ in range(k)]
        self._unconsumed = False

    def _ring_put(self, ring, pos, vec):
        pos = int(pos)
        value = _arr(vec)
        if ring and ring[-1][0] == pos:
            ring[-1] = (pos, value)
        else:
            ring.append((pos, value))
        while len(ring) > self._RING:
            ring.pop(0)

    @staticmethod
    def _ring_get(ring, pos):
        for p, v in ring:
            if p == pos:
                return v
        return None

    def _tok_at(self, pos):
        i = pos - self._tok_base
        return self._toks[i] if 0 <= i < len(self._toks) else None

    def _trim(self):
        if not self._mtp_pos:
            return
        keep = max(0, min(self._mtp_pos) - 1)
        if keep > self._tok_base:
            drop = min(keep - self._tok_base, len(self._toks))
            if drop > 0:
                del self._toks[:drop]
                self._tok_base += drop
        for ring in [self._h0] + self._h:
            while ring and ring[0][0] < keep:
                ring.pop(0)

    def _mtp_catch_up(self, upto=None):
        if not self._mtp_ready:
            return
        if upto is None:
            upto = self._tok_base + len(self._toks)
        while True:
            moved = False
            for k in range(1, self._mtp_K + 1):
                src = self._h0 if k == 1 else self._h[k - 2]
                pos = self._mtp_pos[k - 1]
                end = min(upto, pos + self._WINDOW)
                span = []
                j = pos
                while j < end:
                    if self._tok_at(j + k) is None or self._ring_get(src, j) is None:
                        break
                    span.append(j)
                    j += 1
                if not span:
                    continue
                H = np.stack([self._ring_get(src, x) for x in span])
                tids = np.asarray([self._tok_at(x + k) for x in span], dtype=np.int64)
                _logits, Y, s = self.model._mtp_module(k, H, tids, state=self._mtp_states[k - 1])
                self._mtp_states[k - 1] = s
                self._mtp_pos[k - 1] = span[-1] + 1
                for x, row in zip(span, Y):
                    self._ring_put(self._h[k - 1], x, row)
                moved = True
            if not moved:
                break
        self._trim()

    def prefill(self, tokens, chunk=0):
        toks = list(tokens)
        if not toks:
            # Pure no-op: states are untouched, so a pending distribution left
            # by earlier consumption stays valid.
            return self.states
        self._pending_logits = None
        if not self._mtp_ready and self.draft_depth > 0:
            self._reset_mtp()
        if not self._mtp_ready:
            if chunk and chunk > 0:
                for i in range(0, len(toks), chunk):
                    logits, _ = self.model.forward(toks[i:i + chunk], states=self.states)
                    self._pending_logits = _arr(logits[-1])
            elif toks:
                logits, _ = self.model.forward(toks, states=self.states)
                self._pending_logits = _arr(logits[-1])
            return self.states
        step = chunk if chunk and chunk > 0 else 512
        for i in range(0, len(toks), step):
            seg = toks[i:i + step]
            base = self._tok_base + len(self._toks)
            logits, hidden = self.model.hidden_forward(seg, self.states)
            self._pending_logits = _arr(logits[-1])
            for j in range(hidden.shape[0]):
                self._ring_put(self._h0, base + j, hidden[j])
            self._toks.extend(int(x) for x in seg)
            self._mtp_catch_up()
        self._unconsumed = False
        return self.states

    # -- sampling (kept identical to plain decoding) --
    def _probs(self, logits):
        logits = _arr(logits)
        if self.temperature == 0:
            p = np.zeros_like(logits)
            p[int(np.argmax(logits))] = 1.0
            return p
        l = logits.astype(np.float32) / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, l.size)
            threshold = np.sort(l)[-k]
            l = np.where(l < threshold, -np.inf, l)
        if self.top_p < 1:
            order = np.argsort(-l, kind="stable")
            sorted_l = l[order]
            probs = np.exp(sorted_l - np.max(sorted_l))
            probs /= probs.sum()
            cumulative = np.cumsum(probs)
            remove = cumulative - probs >= self.top_p
            sorted_l = np.where(remove, -np.inf, sorted_l)
            l = np.full_like(l, -np.inf)
            l[order] = sorted_l
        shifted = l - np.max(l)
        probs = np.exp(shifted)
        return probs / probs.sum()

    def _draw(self, probs):
        if self.temperature == 0:
            return int(np.argmax(probs))
        return int(self.generator.choice(probs.size, p=probs))

    def _sample(self, logits):
        return self._draw(self._probs(logits))

    def _accept(self, p, q, draft):
        if self.temperature == 0:
            best = int(np.argmax(p))
            return best == int(draft), (None if best == int(draft) else best)
        pd = float(p[draft])
        qd = float(q[draft])
        if qd <= 0:
            return False, self._draw(p)
        ratio = pd / qd
        if ratio >= 1.0:
            return True, None
        if float(self.generator.random()) < ratio:
            return True, None
        resid = np.maximum(p - q, 0.0)
        total = float(resid.sum())
        if total <= 0:
            return False, self._draw(p)
        return False, self._draw(resid / total)

    def generate(self, prompt_tokens, max_new_tokens=64, stop_ids=()):
        return list(self.generate_stream(prompt_tokens, max_new_tokens, stop_ids))

    def _consume_pending_logits(self):
        pending = self._pending_logits
        self._pending_logits = None
        return pending

    def generate_stream(self, prompt_tokens, max_new_tokens=64, stop_ids=()):
        prompt = list(prompt_tokens)
        self.prefill(prompt)
        stop = set(stop_ids)
        if not (self._mtp_ready and self.speculative > 0):
            # ``prefill`` already consumed the prompt and left the
            # distribution for the next token in ``_pending_logits``.  Start
            # there instead of re-stepping the prompt's last token (historical
            # behavior, which double-consumed it) or a literal seed byte.
            pending = self._consume_pending_logits()
            if pending is None:
                if self._last_plain_token is not None:
                    pending = self.model.step_token(self._last_plain_token, self.states)
                else:
                    # Fresh engine without any context keeps the historical
                    # byte-0 seed for that degenerate case.
                    pending = self.model.step_token(0, self.states)
            for _ in range(max_new_tokens):
                nxt = self._sample(pending)
                self._last_plain_token = nxt
                yield nxt
                if nxt in stop:
                    self._pending_logits = None
                    return
                self._pending_logits = pending = self.model.step_token(nxt, self.states)
            self._last_plain_token = nxt
            return
        pending = self._consume_pending_logits()
        if pending is None:
            # No fresh distribution (fresh engine, or restored states): keep
            # the legacy behavior of stepping the tracked last token.
            last = self._toks[-1] if self._toks else 0
            if not self._unconsumed:
                self._toks.append(int(last))
            self._unconsumed = True
            yield from self._generate_speculative(max_new_tokens, stop)
            return
        # After prefill the last prompt token is already consumed and its
        # hidden vector is on the _h0 ring; step straight from the pending
        # distribution.
        pos = self._tok_base + len(self._toks) - 1
        self._unconsumed = True
        yield from self._generate_speculative(
            max_new_tokens, stop, pending=pending, seed_pos=pos)

    def _mtp_draft(self, pos):
        drafts = []
        qs = []
        K = self._mtp_K
        if K <= 0:
            return drafts, qs
        pending = {}
        out = [dict() for _ in range(K)]

        def tok_tent(p):
            return pending[p] if p in pending else self._tok_at(p)

        def hprev_tent(k, j):
            if k == 1:
                return self._ring_get(self._h0, j)
            return out[k - 2].get(j, self._ring_get(self._h[k - 2], j))

        for k in range(1, K + 1):
            rows = list(range(self._mtp_pos[k - 1], pos + 1))
            if not rows:
                break
            H = []
            T = []
            ready = True
            for j in rows:
                hp = hprev_tent(k, j)
                tv = tok_tent(j + k)
                if hp is None or tv is None:
                    ready = False
                    break
                H.append(hp)
                T.append(tv)
            if not ready or not H:
                break
            logits, Y, _s = self.model._mtp_module(
                k, np.stack(H), np.asarray(T, dtype=np.int64), state=self._mtp_states[k - 1])
            for j, row in zip(rows, Y):
                out[k - 1][j] = _arr(row)
            probs = self._probs(logits[-1])
            d = self._draw(probs)
            drafts.append(d)
            qs.append(probs)
            pending[pos + k + 1] = d
        return drafts, qs

    def _rollback(self, hist, idx):
        self.states = [hist[l][idx].copy() for l in range(len(self.states))]

    def _generate_speculative(self, max_new_tokens, stop, pending=None, seed_pos=None):
        # ``pending`` marks an already-consumed seed: the token at ``seed_pos``
        # was consumed by ``prefill`` (its hidden lives on the _h0 ring) and
        # ``pending`` describes what follows it. The loop must not re-step it;
        # later iterations revert to the legacy ``last``-consumed path.
        last = self._toks[-1] if self._toks else 0
        remaining = int(max_new_tokens)
        step_from_pending = pending is not None
        while remaining > 0:
            if step_from_pending:
                logits = pending
                pos = seed_pos
                step_from_pending = False
            else:
                logits, h0 = self.model.step_details(last, self.states)
                self._unconsumed = False
                pos = self._tok_base + len(self._toks) - 1
                self._ring_put(self._h0, pos, h0)
            t1 = self._sample(logits)
            self._toks.append(int(t1))
            self._unconsumed = True
            self._mtp_catch_up(pos)
            drafts, qs = self._mtp_draft(pos)
            remaining -= 1
            yield t1
            if t1 in stop or remaining == 0:
                return
            if not drafts:
                last = t1
                continue
            K = len(drafts)
            seq = [t1] + drafts
            vlogits, hidden, hist = self.model.verify_tokens(seq, self.states)
            accepted = 0
            repl = None
            for j, d in enumerate(drafts):
                ok, r = self._accept(self._probs(vlogits[j]), qs[j], d)
                if ok:
                    accepted += 1
                else:
                    repl = r
                    break
            if accepted == K:
                repl = self._draw(self._probs(vlogits[K]))
            self._rollback(hist, accepted)
            for j in range(accepted):
                self._toks.append(int(drafts[j]))
            for j in range(accepted + 1):
                self._ring_put(self._h0, pos + 1 + j, hidden[j])
            self._toks.append(int(repl))
            for m, d in enumerate(drafts[:accepted]):
                remaining -= 1
                yield d
                if d in stop or remaining == 0:
                    self._rollback(hist, m)
                    del self._toks[max(0, pos + 3 + m - self._tok_base):]
                    return
            remaining -= 1
            yield repl
            if repl in stop or remaining == 0:
                return
            last = repl
