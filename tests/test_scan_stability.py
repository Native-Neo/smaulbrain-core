"""Regression tests for the recurrent-scan stability fallback.

The chunked closed form overflows to NaN once a channel's alpha**chunk
underflows the fp32 normal range; the exact sequential recurrence cannot.
These tests pin that boundary and the equivalence of both paths inside the
closed form's safe range.
"""
import numpy as np

from smaulbrain import npgrad as ng


def _alpha_from_w(w):
    """alpha = exp(-softplus(w)) evaluated the way the model does it."""
    w = np.asarray(w, dtype=np.float32)
    return (np.exp(-np.log1p(np.exp(-np.abs(w))) - np.maximum(w, 0.0))).astype(np.float32)


def _sequential_ref(values, alpha, s0):
    state = np.array(s0, dtype=np.float32, copy=True)
    out = np.empty_like(values)
    for t in range(values.shape[0]):
        state = alpha * state + values[t]
        out[t] = state
    return out


def test_scan_matches_sequential_reference_across_alphas():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(8):
        d = int(rng.integers(1, 8))
        t = int(rng.integers(1, 64))
        w = rng.uniform(-2.0, 30.0, size=d).astype(np.float32)
        alpha = _alpha_from_w(w)
        values = rng.standard_normal((t, d)).astype(np.float32)
        s0 = rng.standard_normal(d).astype(np.float32)
        got = ng.scan(ng.Tensor(values), ng.Tensor(alpha), s0=ng.Tensor(s0)).data
        ref = _sequential_ref(values, alpha, s0)
        assert np.isfinite(got).all()
        rel = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-6))
        worst = max(worst, rel)
    assert worst < 1e-5


def test_scan_stays_finite_where_closed_form_produced_nan():
    # Carried-state forwards went non-finite from w>=45 before the fallback;
    # alpha == 0 (full fp32 underflow at w~200+) underflowed outright.
    one = np.ones(8, dtype=np.float32)
    for w in (0.0, 20, 30, 44, 45, 46, 60, 90, 200, 500):
        alpha = _alpha_from_w(np.full(8, w))
        values = np.ones((16, 8), dtype=np.float32)
        got = ng.scan(ng.Tensor(values), ng.Tensor(alpha), s0=ng.Tensor(one)).data
        ref = _sequential_ref(values, alpha, one)
        assert np.isfinite(got).all(), f"non-finite scan output at w={w}"
        assert np.allclose(got, ref, rtol=1e-4, atol=1e-5), f"mismatch at w={w}"


def test_scan_backward_matches_finite_differences_at_degenerate_alpha():
    # Through alpha = exp(-softplus(w)) so the autograd dalpha/dw chain is
    # validated end to end in a regime where the closed form is unstable.
    d, t = 4, 6
    x = np.full((t, d), 0.7, dtype=np.float32)
    scale = np.cos(np.arange(d, dtype=np.float32))
    w0 = np.array([46.0, 30.0, 5.0, 0.5], dtype=np.float32)
    h = 0.05
    numeric = np.zeros(d)
    for i in range(d):
        for sign in (1.0, -1.0):
            w = w0.astype(np.float64)
            w[i] += sign * h
            alpha = _alpha_from_w(w)
            out = ng.scan(ng.Tensor(x), ng.Tensor(alpha), s0=ng.Tensor(np.ones(d, np.float32))).data
            numeric[i] += sign * (out * scale).sum() / (2 * h)
    wv = ng.Tensor(w0.copy(), requires_grad=True)
    out = ng.scan(ng.Tensor(x), ng.exp(-ng.softplus(wv)), s0=ng.Tensor(np.ones(d, np.float32)))
    out.backward(ng.Tensor(np.broadcast_to(scale, (t, d)).copy()))
    analytic = wv.grad
    assert np.isfinite(analytic).all()
    rel = np.abs(numeric - analytic) / (np.abs(numeric) + 1e-3)
    assert np.max(rel) < 5e-3
