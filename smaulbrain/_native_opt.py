"""Optional CPU C++ fused SmaulOPT update reached through NumPy buffers."""
from __future__ import annotations

import numpy as np

from . import _native_opt_numpy as _backend
from .npgrad import Tensor


def factored_step(opt, w, g, st, collect_norms):
    if isinstance(w, Tensor):
        w = w.data
    w = np.asarray(w)
    if not w.flags.c_contiguous or w.ndim != 2 or w.size < 1024:
        return None
    for key in ("m", "R", "C"):
        value = st.get(key)
        if isinstance(value, Tensor):
            value = value.data
        if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
            return None
        st[key] = value
    if st["m"].shape != w.shape or st["R"].shape != (w.shape[0],) or st["C"].shape != (w.shape[1],):
        return None
    if isinstance(g, Tensor):
        g = g.data
    g = np.asarray(g)
    return _backend.factored_step(opt, w, g, {k: st[k] for k in ("m", "R", "C")}, collect_norms)
