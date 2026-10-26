"""NumPy implementation of SmaulOPT's factored second-moment optimizer."""
from __future__ import annotations

import numpy as np

from . import _native_opt
from .npgrad import Tensor


def _arr(value, dtype=np.float32):
    if isinstance(value, Tensor):
        value = value.data
    return np.asarray(value, dtype=dtype)


def _parameter_data(value):
    """Return the writable float array behind a parameter without copying."""
    data = value.data if isinstance(value, Tensor) else value
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"SmaulOPT parameters must be NumPy arrays or Tensors, got {type(value).__name__}")
    if not np.issubdtype(data.dtype, np.floating):
        raise TypeError(f"SmaulOPT parameters must be floating point, got {data.dtype}")
    if not data.flags.writeable:
        raise ValueError("SmaulOPT parameter storage is read-only")
    return data


def _state_array(state, key, shape, name):
    value = state.get(key)
    if value is None:
        raise ValueError(f"optimizer state for {name!r} is missing {key!r}")
    if isinstance(value, Tensor):
        value = value.data
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"optimizer state {name!r}.{key} must be a NumPy array, "
            f"got {type(value).__name__}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"optimizer state {name!r}.{key} must be floating point")
    if value.shape != shape:
        raise ValueError(
            f"optimizer state {name!r}.{key} has shape {value.shape}, expected {shape}")
    if not value.flags.writeable:
        raise ValueError(f"optimizer state {name!r}.{key} is read-only")
    return value


def _validate_state(name, w, state):
    if not isinstance(state, dict):
        raise TypeError(f"optimizer state for {name!r} must be a dict")
    factored = state.get("factored")
    if not isinstance(factored, (bool, np.bool_)):
        raise ValueError(f"optimizer state for {name!r} has no valid 'factored' flag")
    if factored:
        if w.ndim != 2:
            raise ValueError(
                f"factored optimizer state for {name!r} requires a 2-D parameter, "
                f"got shape {w.shape}")
        _state_array(state, "m", w.shape, name)
        _state_array(state, "R", (w.shape[0],), name)
        _state_array(state, "C", (w.shape[1],), name)
    else:
        _state_array(state, "m", w.shape, name)
        _state_array(state, "v", w.shape, name)


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
        w = _parameter_data(w)
        if w.ndim == 2:
            self.state[name] = {"m": np.zeros_like(w), "R": np.zeros(w.shape[0], dtype=np.float32),
                                "C": np.zeros(w.shape[1], dtype=np.float32), "factored": True}
        else:
            self.state[name] = {"m": np.zeros_like(w), "v": np.zeros_like(w), "factored": False}

    def register(self, params):
        for k, w in params.items():
            w = _parameter_data(w)
            if k not in self.state:
                self._init_state(k, w)
            else:
                _validate_state(k, w, self.state[k])

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

    def step(self, params, grads, collect_norms=True, check_finite=True):
        """Apply one update.

        Every parameter, gradient, and state buffer is validated before the
        first mutation. A bad gradient therefore cannot leave the model half
        updated, and an unknown gradient name is an error instead of a silent
        no-op. ``check_finite=False`` skips the full-gradient finite scan for
        callers that have already validated their gradients.
        """
        if not isinstance(grads, dict):
            raise TypeError("grads must be a dict of parameter-name -> gradient")
        unknown = [name for name, grad in grads.items()
                   if grad is not None and name not in params]
        if unknown:
            raise ValueError(
                "gradient supplied for unknown SmaulOPT parameter(s): "
                + ", ".join(sorted(repr(name) for name in unknown)))

        prepared = []
        for name, value in params.items():
            g = grads.get(name)
            if g is None:
                continue
            w = _parameter_data(value)
            if name not in self.state:
                self._init_state(name, w)
            st = self.state[name]
            _validate_state(name, w, st)
            g = _arr(g)
            if g.shape != w.shape:
                raise ValueError(
                    f"SmaulOPT gradient for {name!r} has shape {g.shape}, "
                    f"expected {w.shape}")
            if check_finite and g.size and not np.isfinite(g).all():
                raise FloatingPointError(
                    f"non-finite SmaulOPT gradient for {name!r}; update rejected")
            if not g.flags.c_contiguous:
                g = np.ascontiguousarray(g)
            prepared.append((name, w, g, st, bool(st["factored"])))

        norms = {}
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        omb1 = 1.0 - b1
        omb2 = 1.0 - b2
        wd = 1.0 - self.lr * self.weight_decay if self.weight_decay else None
        factored = []
        for name, w, g, st, is_factored in prepared:
            if is_factored:
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

    def load_state_dict(self, sd, factored_flags=None, copy=True, check_finite=True):
        if not isinstance(sd, dict):
            raise TypeError("SmaulOPT state must be a dict")
        flags = factored_flags or {}
        if not isinstance(flags, dict):
            raise TypeError("factored_flags must be a dict")
        checked = {}
        for k, s in sd.items():
            if not isinstance(k, str) or not isinstance(s, dict):
                raise TypeError("SmaulOPT state must map parameter names to dicts")
            if "m" not in s:
                raise ValueError(f"optimizer state for {k!r} is missing 'm'")
            has_r, has_c = "R" in s, "C" in s
            if has_r != has_c:
                raise ValueError(
                    f"optimizer state for {k!r} must contain both 'R' and 'C'")
            factored = has_r
            if factored and "v" in s:
                raise ValueError(
                    f"optimizer state for {k!r} mixes factored and full moments")
            if not factored and "v" not in s:
                raise ValueError(
                    f"optimizer state for {k!r} has no 'v' second moment")
            if k in flags and bool(flags[k]) != factored:
                raise ValueError(
                    f"load_state_dict: factored_flags[{k!r}]={bool(flags[k])} contradicts "
                    f"the stored state layout ({'R/C' if factored else 'v'} moments)")
            entry = {}
            m = _arr(s["m"])
            entry["m"] = m.copy() if copy or not m.flags.writeable else m
            if factored:
                for key in ("R", "C"):
                    arr = _arr(s[key])
                    entry[key] = arr.copy() if copy or not arr.flags.writeable else arr
                expected = [(entry["m"], m.shape),
                            (entry["R"], (m.shape[0],)),
                            (entry["C"], (m.shape[1],))]
            else:
                v = _arr(s["v"])
                entry["v"] = v.copy() if copy or not v.flags.writeable else v
                expected = [(entry["m"], m.shape), (entry["v"], m.shape)]
            for arr, shape in expected:
                if arr.shape != shape:
                    raise ValueError(
                        f"optimizer state for {k!r} has shape {arr.shape}, expected {shape}")
                if check_finite and arr.size and not np.isfinite(arr).all():
                    raise FloatingPointError(
                        f"non-finite optimizer state for {k!r}; load rejected")
            entry["factored"] = factored
            checked[k] = entry
        self.state = checked
