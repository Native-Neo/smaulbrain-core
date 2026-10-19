"""Optimizer + operator math tests (critic-math audit).

Covers:
  * SmaulOPT.step vs an independent plain-PyTorch reference for factored
    (2-D) and full (1-D) params, with and without weight decay;
  * the optional fused C++ update (`_native_opt`/`_native_opt_numpy`) vs the
    pure-Python fallback on the same inputs, including monkeypatching the
    native entry point to None and `SMAULBRAIN_NATIVE_OPT=0`;
  * optimizer state_dict/load_state_dict round-trips and validation;
  * ops.rmsnorm_fwd/rmsnorm_bwd vs math_ref and torch.autograd (fp64),
    gelu_tanh_grad, sigmoid/silu/softplus;
  * RecurrentAttention vs math_ref.recurrent_forward_ref, T=1..5, d=3..8,
    including chunked state continuation (the inference contract).

Tolerances (documented, measured on this baseline):
  * PyTorch reference python path: exact (max|dW| == 0.0 observed); asserted
    <= 1e-6 * max|W| (abs 1e-9 floor).
  * Fused native path vs reference: observed <= 5.9e-8 relative per step
    (2.4e-7 after 4 steps); asserted <= 1e-5 relative.
  * native vs python fallback (same inputs): observed |dW| <= 3.6e-7,
    |dR|,|dC| <= 4.7e-10, |dnorm| <= 7.7e-8; asserted <= 1e-5 / 1e-6.
  * ops vs fp64 math_ref/autograd: observed <= 5e-7, asserted <= 2e-5.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from smaulbrain import math_ref, ops
from smaulbrain import _native_opt as native_opt
from smaulbrain import _native_opt_numpy as native_backend
from smaulbrain.smaulopt import SmaulOPT

_HAVE_NATIVE = native_backend._load() is not None
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# independent reference
# --------------------------------------------------------------------------
def ref_step(params, grads, state, lr, b1, b2, eps, wd):
    """Plain-PyTorch SmaulOPT reference (no native path, own state ordering).

    Returns the updated params dict (out-of-place update).
    """
    out = {}
    for name, W in params.items():
        g = grads[name].detach().float()
        st = state.setdefault(name, {})
        if W.ndim == 2:
            if "m" not in st:
                st["m"] = torch.zeros_like(W)
                st["R"] = torch.zeros(W.shape[0])
                st["C"] = torch.zeros(W.shape[1])
            m, R, C = st["m"], st["R"], st["C"]
            m = b1 * m + (1.0 - b1) * g
            R = b2 * R + (1.0 - b2) * g.square().mean(dim=1)
            C = b2 * C + (1.0 - b2) * g.square().mean(dim=0)
            vhat = (R[:, None] * C[None, :] / R.mean()).clamp_min(0)
            upd = m / (vhat.sqrt() + eps)
            st["m"], st["R"], st["C"] = m, R, C
        else:
            if "m" not in st:
                st["m"] = torch.zeros_like(W)
                st["v"] = torch.zeros_like(W)
            m, v = st["m"], st["v"]
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * g.square()
            upd = m / (v.clamp_min(0).sqrt() + eps)
            st["m"], st["v"] = m, v
        if wd:
            W = W * (1.0 - lr * wd)
        out[name] = W - lr * upd
    return out


def run_steps(shape, wd, steps=4, lr=1e-3, b1=0.9, b2=0.98, eps=1e-8, seed=1234):
    """Run SmaulOPT and the reference on identical data; return final tensors."""
    torch.manual_seed(seed)
    W0 = torch.randn(*shape) * 0.5
    Wopt, Wref = W0.clone(), W0.clone()
    opt = SmaulOPT(lr=lr, beta1=b1, beta2=b2, eps=eps, weight_decay=wd)
    ref_state = {}
    for s in range(steps):
        g = torch.randn(*shape) * (0.1 + 0.05 * s)
        opt.step({"W": Wopt}, {"W": g}, collect_norms=False)
        Wref = ref_step({"W": Wref}, {"W": g}, ref_state, lr, b1, b2, eps, wd)["W"]
    return Wopt, Wref, opt


@pytest.mark.parametrize("shape,wd", [((8, 8), 0.0), ((8, 8), 0.1), ((16,), 0.0),
                                      ((16,), 0.01), ((40, 30), 0.0), ((40, 30), 0.1)])
def test_smaulopt_step_matches_python_reference(shape, wd):
    """Claim 1: factored R/C, decay before update, eps after sqrt, 3+ steps."""
    Wopt, Wref, _ = run_steps(shape, wd)
    scale = max(1e-9, Wref.abs().max().item())
    err = (Wopt - Wref).abs().max().item()
    if Wopt.ndim == 2 and Wopt.numel() >= 1024:
        # fused C++ path only (see test_native_matches_python_fallback)
        assert err <= 1e-5 * scale, (shape, wd, err)
    else:
        assert err <= 1e-6 * scale, (shape, wd, err)


def test_smaulopt_reference_independent_of_hyperparameters():
    """Claim 1: hyperparameter fuzz (beta1/beta2/eps/wd/lr, shapes, 1-D+2-D)."""
    for shape in ((1,), (7,), (3, 4), (40, 30), (1, 1024)):
        for b1, b2, eps, wd, lr in [(0.0, 0.0, 1e-8, 0.0, 1e-3),
                                    (0.5, 0.5, 1e-6, 0.1, 0.1),
                                    (0.9, 0.98, 1e-8, 0.1, 1e-3)]:
            torch.manual_seed(hash((shape, b1, b2, eps, wd, lr)) % 2**31)
            W0 = torch.randn(*shape) * 0.5
            Wo, Wr = W0.clone(), W0.clone()
            opt = SmaulOPT(lr=lr, beta1=b1, beta2=b2, eps=eps, weight_decay=wd)
            state = {}
            for s in range(3):
                g = torch.randn(*shape) * (0.05 + 0.2 * s)
                opt.step({"W": Wo}, {"W": g}, collect_norms=False)
                Wr = ref_step({"W": Wr}, {"W": g}, state, lr, b1, b2, eps, wd)["W"]
            scale = max(1e-9, Wr.abs().max().item())
            assert (Wo - Wr).abs().max().item() <= 1e-5 * scale, (shape, b1, b2, eps, wd, lr)


def test_smaulopt_native_path_used_for_large_matrices():
    """A >=1024-element 2-D param goes through the fused update when available."""
    if not _HAVE_NATIVE:
        pytest.skip("native optimizer backend unavailable")
    opt = SmaulOPT()
    w = torch.randn(64, 64)
    st = {"m": torch.zeros_like(w), "R": torch.zeros(64), "C": torch.zeros(64), "factored": True}
    assert native_opt.factored_step(opt, w, torch.randn_like(w), st, True) is not None
    small = torch.randn(8, 8)
    st_small = {"m": torch.zeros_like(small), "R": torch.zeros(8), "C": torch.zeros(8), "factored": True}
    assert native_opt.factored_step(opt, small, torch.randn_like(small), st_small, True) is None


@pytest.mark.skipif(not _HAVE_NATIVE, reason="native optimizer backend unavailable")
def test_native_matches_python_fallback_same_state(monkeypatch):
    """Claim 2: the fused update vs the pure-python fallback on identical
    inputs, through SmaulOPT.step (>=1024-element matrices use the fused C++).

    Forcing `_native_opt.factored_step` to None exercises the python fallback.
    Compares W, m, R, C and the returned update RMS.
    """
    for shape in [(32, 32), (16, 64), (17, 61), (300, 300)]:
        def run(force_python):
            torch.manual_seed(7)
            w = torch.randn(*shape) * 0.5
            opt = SmaulOPT(lr=2e-3, beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.1)
            norms = []
            for s in range(3):
                g = torch.randn(*shape) * (0.1 + 0.1 * s)
                if force_python:
                    monkeypatch.setattr(native_opt, "factored_step", lambda *a, **k: None)
                norms.append(opt.step({"W": w}, {"W": g})["W"])
                if force_python:
                    monkeypatch.undo()
            st = opt.state["W"]
            return w.clone(), st["m"].clone(), st["R"].clone(), st["C"].clone(), norms

        a, b = run(False), run(True)
        assert (a[0] - b[0]).abs().max().item() <= 1e-5
        assert torch.equal(a[1], b[1])
        assert (a[2] - b[2]).abs().max().item() <= 1e-6
        assert (a[3] - b[3]).abs().max().item() <= 1e-6
        assert (a[0].shape, a[1].shape, a[2].shape, a[3].shape) == \
               (b[0].shape, b[1].shape, b[2].shape, b[3].shape)
        for x, y in zip(a[4], b[4]):
            assert abs(x - y) <= 1e-5


@pytest.mark.skipif(not _HAVE_NATIVE, reason="native optimizer backend unavailable")
def test_native_backend_norm_and_side_effects(monkeypatch):
    """`_native_opt_numpy.factored_step` RMS over the whole matrix == python path."""
    torch.manual_seed(3)
    w = torch.randn(64, 64) * 0.5
    g = torch.randn(64, 64) * 0.1
    opt = SmaulOPT(lr=1e-3, weight_decay=0.1)
    wn = w.clone()
    stn = {"m": torch.zeros_like(w), "R": torch.zeros(64), "C": torch.zeros(64), "factored": True}
    got = native_backend.factored_step(opt, wn.detach().numpy(),
                                       g.detach().contiguous().numpy(),
                                       {k: stn[k].detach().numpy() for k in ("m", "R", "C")}, True)
    assert isinstance(got, float) and got > 0
    # python fallback on the same inputs
    monkeypatch.setattr(native_opt, "factored_step", lambda *a, **k: None)
    wp = w.clone()
    ref = opt.step({"W": wp}, {"W": g})["W"]
    assert abs(got - ref) <= 1e-5
    assert (wn - wp).abs().max().item() <= 1e-5
    # collect_norms=False returns 0.0 and does not touch the norm buffer
    stn2 = {k: stn[k].clone() for k in ("m", "R", "C")}
    got0 = native_backend.factored_step(opt, wn.detach().numpy(),
                                        g.detach().contiguous().numpy(),
                                        {k: stn2[k].detach().numpy() for k in ("m", "R", "C")}, False)
    assert got0 == 0.0


def test_native_returns_none_for_ineligible_tensors():
    """The fused path refuses and the caller falls back; no crash."""
    opt = SmaulOPT()
    m = torch.randn(40, 40)
    zero = torch.zeros(40, 40)
    rc = {"m": zero, "R": torch.zeros(40), "C": torch.zeros(40)}
    assert native_opt.factored_step(opt, m.t(), torch.randn(40, 40), rc, True) is None  # non-contiguous
    assert native_opt.factored_step(opt, m.double(), torch.randn(40, 40), rc, True) is None
    assert native_opt.factored_step(opt, m, torch.randn(40, 40),
                                    {"m": zero, "R": torch.zeros(41), "C": torch.zeros(40)}, True) is None
    assert native_opt.factored_step(opt, torch.randn(16), torch.randn(16),
                                    {"m": torch.zeros(16), "R": torch.zeros(4), "C": torch.zeros(4)}, True) is None


@pytest.mark.skipif(not _HAVE_NATIVE, reason="native optimizer backend unavailable")
def test_native_env_gate_disables_backend():
    """SMAULBRAIN_NATIVE_OPT=0 makes a fresh process fall back to python."""
    code = ("from smaulbrain import _native_opt_numpy as b\n"
            "print('NATIVE' if b._load() is not None else 'FALLBACK')\n")
    env = dict(os.environ, PYTHONPATH=_ROOT, SMAULBRAIN_NATIVE_OPT="0")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "FALLBACK", out.stderr
    env["SMAULBRAIN_NATIVE_OPT"] = "1"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "NATIVE", out.stderr


@pytest.mark.skipif(not _HAVE_NATIVE, reason="native optimizer backend unavailable")
def test_native_results_independent_of_thread_count():
    """Deterministic reductions: SMAULBRAIN_THREADS=1 and 4 give identical bytes."""
    code = ("import hashlib, torch\n"
            "from smaulbrain.smaulopt import SmaulOPT\n"
            "from smaulbrain import _native_opt as nat\n"
            "torch.manual_seed(0)\n"
            "w = torch.randn(512, 512) * 0.5\n"
            "g = torch.randn(512, 512) * 0.1\n"
            "opt = SmaulOPT(lr=1e-3, weight_decay=0.1)\n"
            "st = {'m': torch.zeros_like(w), 'R': torch.zeros(512), 'C': torch.zeros(512), 'factored': True}\n"
            "for s in range(3):\n"
            "    r = nat.factored_step(opt, w, g * (1 + s), st, True)\n"
            "    assert r is not None\n"
            "print(hashlib.sha256(w.numpy().tobytes()).hexdigest()[:16],\n"
            "      hashlib.sha256(st['m'].numpy().tobytes()).hexdigest()[:16])\n")
    outs = []
    for threads in ("1", "4"):
        env = dict(os.environ, PYTHONPATH=_ROOT, SMAULBRAIN_THREADS=threads)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] and outs[0]


def test_smaulopt_norms_and_no_grad_flag():
    """collect_norms=False records nothing; missing grads are skipped."""
    torch.manual_seed(0)
    w = torch.randn(6, 6)
    g = torch.randn(6, 6)
    opt = SmaulOPT()
    assert opt.step({"W": w}, {"W": g}, collect_norms=False) == {}
    n = opt.step({"W": w}, {"W": g}, collect_norms=True)
    assert set(n) == {"W"} and n["W"] > 0
    assert opt.n_updates == 2
    assert opt.step({"W": w}, {}, collect_norms=False) == {}


def test_smaulopt_state_dict_roundtrip():
    """Claim 3: factored + full state, drop, state_memory_elements."""
    torch.manual_seed(0)
    A = torch.randn(40, 30)
    b = torch.randn(10)
    opt = SmaulOPT(lr=0.01, weight_decay=0.1)
    opt.step({"A": A, "b": b}, {"A": torch.randn(40, 30) * 0.1, "b": torch.randn(10) * 0.1})
    sd = opt.state_dict()
    assert set(sd) == {"A", "b"}
    assert "factored" not in sd["A"] and "factored" not in sd["b"]
    assert sd["A"]["m"].data_ptr() != opt.state["A"]["m"].data_ptr()  # cloned
    # R/C survive a second serialization round-trip
    assert set(opt.state_dict()["A"]) == {"m", "R", "C"}
    assert set(opt.state_dict()["b"]) == {"m", "v"}

    opt2 = SmaulOPT(lr=0.01, weight_decay=0.1)
    opt2.load_state_dict(sd)
    assert opt2.state["A"]["factored"] is True
    assert opt2.state["b"]["factored"] is False
    for k, sub in (("A", ("m", "R", "C")), ("b", ("m", "v"))):
        for sk in sub:
            assert torch.equal(opt2.state[k][sk], opt.state[k][sk])

    mem = opt2.state_memory_elements()
    assert mem == {"factored_elements": 40 * 30 + 40 + 30, "full_elements": 2 * 10}

    # continuing training from a restored state matches the original state
    gA = torch.randn(40, 30) * 0.05
    gb = torch.randn(10) * 0.05
    Wa, Wb = A.clone(), b.clone()
    ob = SmaulOPT(lr=0.01, weight_decay=0.1)
    ob.load_state_dict(sd)
    Wa2, Wb2 = A.clone(), b.clone()
    ob2 = SmaulOPT(lr=0.01, weight_decay=0.1)
    ob2.load_state_dict(sd)
    ob.step({"A": Wa, "b": Wb}, {"A": gA, "b": gb})
    ob2.step({"A": Wa2, "b": Wb2}, {"A": gA, "b": gb})
    assert torch.equal(Wa, Wa2) and torch.equal(Wb, Wb2)

    # list payloads are converted to tensors
    o3 = SmaulOPT()
    o3.load_state_dict({"z": {"m": [[1.0, 2.0]], "v": [[3.0, 4.0]]}})
    assert isinstance(o3.state["z"]["m"], torch.Tensor)
    assert o3.state["z"]["factored"] is False

    # legacy checkpoints without flags infer the layout from R/C
    o4 = SmaulOPT()
    o4.load_state_dict(sd, factored_flags={})
    assert o4.state["A"]["factored"] is True and o4.state["b"]["factored"] is False

    # drop
    o3.drop(["z", "missing"])
    assert o3.state == {}


def test_smaulopt_load_state_dict_rejects_inconsistent_flags():
    """Claim 3 (regression): an override that contradicts the stored tensors
    used to build a broken state (KeyError('v') on the next step)."""
    opt = SmaulOPT()
    # state says factored (R/C present) but the override says full -> no 'v'
    with pytest.raises(ValueError):
        opt.load_state_dict({"A": {"m": torch.zeros(3, 4), "R": torch.zeros(3),
                                   "C": torch.zeros(4)}}, factored_flags={"A": False})
    # state says full (no R/C) but the override says factored -> no R/C
    with pytest.raises(ValueError):
        opt.load_state_dict({"B": {"m": torch.zeros(3, 4), "v": torch.zeros(3, 4)}},
                            factored_flags={"B": True})
    # consistent forms still load
    o = SmaulOPT()
    o.load_state_dict({"A": {"m": torch.zeros(3, 4), "R": torch.zeros(3), "C": torch.zeros(4)}})
    assert o.state["A"]["factored"] is True
    o = SmaulOPT()
    o.load_state_dict({"B": {"m": torch.zeros(3, 4), "v": torch.zeros(3, 4)}},
                      factored_flags={"B": False})
    assert o.state["B"]["factored"] is False
    o.step({"B": torch.zeros(3, 4)}, {"B": torch.ones(3, 4)})  # usable


# --------------------------------------------------------------------------
# ops
# --------------------------------------------------------------------------
@pytest.mark.parametrize("d,B", [(1, 1), (5, 1), (8, 2), (3, 4)])
def test_rmsnorm_vs_autograd_and_math_ref(d, B):
    """Claim 4: rmsnorm_fwd/bwd vs fp64 autograd and math_ref."""
    torch.manual_seed(3)
    shape = (d,) if B == 1 else (B, d)
    x = torch.randn(*shape, dtype=torch.float64)
    g = torch.randn(d, dtype=torch.float64)
    y, cache = ops.rmsnorm_fwd(x, g, eps=1e-6)

    xr = x.tolist()
    yr = [math_ref.rmsnorm_ref(row, g.tolist()) for row in (xr if B > 1 else [xr])]
    yl = y.tolist() if B > 1 else [y.tolist()]
    for i in range(B):
        for j in range(d):
            assert abs(yl[i][j] - yr[i][j]) <= 2e-6

    xa = x.clone().requires_grad_(True)
    ga = g.clone().requires_grad_(True)
    ya = xa * torch.rsqrt(xa.square().mean(-1, keepdim=True) + 1e-6) * ga
    dy = torch.randn_like(ya)
    dxa, dga = torch.autograd.grad(ya, (xa, ga), dy)
    dx, dg = ops.rmsnorm_bwd(dy, cache)
    assert (dx - dxa).abs().max().item() <= 2e-5
    assert (dg - dga).abs().max().item() <= 2e-5
    assert dx.shape == x.shape and dg.shape == g.shape
    assert cache[2].shape == x.shape  # xn cached without gain


def test_gelu_tanh_grad_matches_autograd():
    """Claim 4: gelu_tanh_grad == d/dx F.gelu(x, approximate='tanh')."""
    for scale in (0.5, 3.0):
        x = torch.randn(9, dtype=torch.float64) * scale
        xr = x.clone().requires_grad_(True)
        y = torch.nn.functional.gelu(xr, approximate="tanh")
        ag = torch.autograd.grad(y, xr, torch.ones_like(y))[0]
        og = ops.gelu_tanh_grad(x)
        assert (ag - og).abs().max().item() <= 1e-6
        assert (ops.gelu_tanh(x) - torch.nn.functional.gelu(x, approximate="tanh")).abs().max().item() == 0.0


def test_sigmoid_silu_softplus_spot_checks():
    """Claim 4: activations agree with math_ref and torch."""
    xs = torch.linspace(-8, 8, 33)
    for f, ref, tf in [(ops.sigmoid, math_ref.sigmoid, torch.sigmoid),
                       (ops.silu, math_ref.silu, lambda t: t * torch.sigmoid(t)),
                       (ops.softplus, math_ref.softplus, torch.nn.functional.softplus)]:
        got = f(xs)
        assert max(abs(got[i].item() - ref(xs[i].item())) for i in range(len(xs))) <= 1e-6
        assert (got - tf(xs)).abs().max().item() <= 1e-6


# --------------------------------------------------------------------------
# RecurrentAttention
# --------------------------------------------------------------------------
@pytest.mark.parametrize("d", [3, 5, 8])
def test_recurrent_attention_matches_math_ref(d):
    """Claim 5: outputs and final state for T=1..5 against the scalar ref."""
    ra = ops.RecurrentAttention(d, rng=np.random.default_rng(7))
    for T in (1, 2, 3, 5):
        X = torch.randn(T, d)
        Y, (s, _extra) = ra.forward(X)
        Yr, sr = math_ref.recurrent_forward_ref(
            X.tolist(), ra.Wr.tolist(), ra.Wk.tolist(), ra.Wv.tolist(), ra.Wo.tolist(),
            ra.w.tolist(), ra.u.tolist())
        for t in range(T):
            for j in range(d):
                assert abs(Y[t, j].item() - Yr[t][j]) <= 3e-5
        for j in range(d):
            assert abs(s[j].item() - sr[j]) <= 3e-5


@pytest.mark.parametrize("d", [3, 8])
def test_recurrent_attention_chunked_continuation(d):
    """Claim 5: T=5 in one call == 2+3 with the returned state (and nonzero s0)."""
    ra = ops.RecurrentAttention(d, rng=np.random.default_rng(11))
    X = torch.randn(5, d)
    Y1, (s1, _) = ra.forward(X)
    Ya, (sa, _) = ra.forward(X[:2])
    Yb, (sb, _) = ra.forward(X[2:], s0=sa)
    assert torch.equal(torch.cat([Ya, Yb]), Y1)
    assert torch.equal(sb, s1)

    s0 = torch.randn(d) * 0.3
    Yf, (sf, _) = ra.forward(X, s0=s0)
    Y1a, (s1a, _) = ra.forward(X[:1], s0=s0)
    Y1b, (s1b, _) = ra.forward(X[1:], s0=s1a)
    assert torch.equal(torch.cat([Y1a, Y1b]), Yf)
    assert torch.equal(s1b, sf)
