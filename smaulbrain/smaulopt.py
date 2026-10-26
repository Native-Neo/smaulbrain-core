"""NumPy implementation of SmaulOPT's factored second-moment optimizer."""
from __future__ import annotations

import numpy as np

from . import _native_opt
from .npgrad import Tensor


def _arr(value, dtype=np.float32):
    if isinstance(value, Tensor):
        value = value.data
    return np.asarray(value, dtype=dtype)


class SmaulOPT:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.0):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.state = {}
        self.n_updates = 0

    def _init_state(self, name, w):
        w = _arr(w)
        if w.ndim == 2:
            self.state[name] = {"m": np.zeros_like(w), "R": np.zeros(w.shape[0], dtype=np.float32),
                                "C": np.zeros(w.shape[1], dtype=np.float32), "factored": True}
        else:
            self.state[name] = {"m": np.zeros_like(w), "v": np.zeros_like(w), "factored": False}

    def register(self, params):
        for k, w in params.items():
            if k not in self.state:
                self._init_state(k, w)

    def drop(self, names):
        for k in names:
            self.state.pop(k, None)

    def state_memory_elements(self):
        factored = full = 0
        for s in self.state.values():
            if s["factored"]:
                factored += s["m"].size + s["R"].size + s["C"].size
            else:
                full += 2 * s["m"].size
        return {"factored_elements": factored, "full_elements": full}

    def step(self, params, grads, collect_norms=True):
        norms = {}
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        omb1 = 1.0 - b1
        omb2 = 1.0 - b2
        wd = 1.0 - self.lr * self.weight_decay if self.weight_decay else None
        factored = []
        for name, value in params.items():
            w = value.data if isinstance(value, Tensor) else value
            g = grads.get(name)
            if g is None:
                continue
            g = np.ascontiguousarray(_arr(g))
            if name not in self.state:
                self._init_state(name, w)
            st = self.state[name]
            if st["factored"]:
                # Defer matrix updates so the native backend can consume the
                # whole list with one FFI crossing. Validation is all-or-none
                # before the call, so the fallback below cannot apply twice.
                factored.append((name, w, g, st))
                continue
            if wd is not None:
                w *= wd
            st["m"] *= b1
            st["m"] += omb1 * g
            g2 = g * g
            st["v"] *= b2
            st["v"] += omb2 * g2
            del g2
            vhat = np.maximum(st["v"], 0.0)
            upd = st["m"] / (np.sqrt(vhat, out=vhat) + eps)
            if collect_norms:
                norms[name] = float(np.sqrt(np.mean(upd * upd, dtype=np.float64)))
            w -= self.lr * upd

        if factored:
            native = _native_opt.factored_step_many(
                self, factored, collect_norms) if _native_opt is not None else None
            if native is not None:
                if collect_norms:
                    for (name, _w, _g, _st), norm in zip(factored, native):
                        norms[name] = norm
            else:
                for name, w, g, st in factored:
                    native_one = _native_opt.factored_step(
                        self, w, g, st, collect_norms)
                    if native_one is not None:
                        if collect_norms:
                            norms[name] = native_one
                        continue
                    if wd is not None:
                        w *= wd
                    st["m"] *= b1
                    st["m"] += omb1 * g
                    g2 = g * g
                    st["R"] *= b2
                    st["R"] += omb2 * g2.mean(axis=1)
                    st["C"] *= b2
                    st["C"] += omb2 * g2.mean(axis=0)
                    del g2
                    denom_r = max(float(st["R"].mean()), 1e-30)
                    vhat = (st["R"][:, None] * st["C"][None, :]) / denom_r
                    np.maximum(vhat, 0.0, out=vhat)
                    upd = st["m"] / (np.sqrt(vhat, out=vhat) + eps)
                    if collect_norms:
                        norms[name] = float(np.sqrt(
                            np.mean(upd * upd, dtype=np.float64)))
                    w -= self.lr * upd
        self.n_updates += 1
        return norms

    def state_dict(self):
        return {k: {sk: (sv.copy() if isinstance(sv, np.ndarray) else sv)
                    for sk, sv in s.items() if sk != "factored"}
                for k, s in self.state.items()}

    def load_state_dict(self, sd, factored_flags=None, copy=True):
        flags = factored_flags or {}
        self.state = {}
        for k, s in sd.items():
            entry = {}
            for sk, sv in s.items():
                arr = _arr(sv)
                entry[sk] = arr.copy() if copy else arr
            factored = "R" in entry
            if k in flags and bool(flags[k]) != factored:
                raise ValueError(
                    f"load_state_dict: factored_flags[{k!r}]={bool(flags[k])} contradicts "
                    f"the stored state layout ({'R/C' if factored else 'v'} moments)")
            entry["factored"] = factored
            self.state[k] = entry
