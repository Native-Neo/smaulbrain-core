"""MoE router + expert aggregation tests (critic-math audit, claim 6).

Semantics under test (smaulbrain/moe.py, Model._moe):
  * top-k selection is a stable descending argsort of X@W (lower expert index
    wins ties), softmax over the selected logits; disabled experts are forced
    to -inf and never selected;
  * per-expert capacity cap = max(1, ceil(cf*T*k/n)) (cf=None -> no cap);
    slots over the cap are dropped in token order, kept weights are
    renormalized, fully dropped rows are exact zeros (NaN-safe);
  * info counts load/overflow consistently;
  * Router.backward is the Jacobian of the normalized weights w.r.t. X and W
    (drops are constants; dropped slots have zero gradient);
  * Model._moe's grouped gather equals an explicit per-slot weighted-expert
    sum, both forward and gradients;
  * balance_aux_loss / expert_output_stats / softmax_rows helpers.
"""
import math

import pytest
import torch

from smaulbrain import config as cfg
from smaulbrain import model as M
from smaulbrain import moe
from smaulbrain.moe import Router


# --------------------------------------------------------------------------
# slow pure-python reference (documented semantics)
# --------------------------------------------------------------------------
def _slow_route(X, W, top_k, capacity_factor=1.25, disabled=()):
    Xl, Wl = X.tolist(), W.tolist()
    T, d, n = len(Xl), len(Wl), len(Wl[0])
    disabled = sorted(set(int(i) for i in disabled))
    scores = [[sum(Xl[t][j] * Wl[j][e] for j in range(d)) for e in range(n)]
              for t in range(T)]
    for t in range(T):
        for e in disabled:
            scores[t][e] = float("-inf")
    idx = [sorted(range(n), key=lambda e: (-scores[t][e], e))[:top_k] for t in range(T)]
    raw = []
    for t in range(T):
        z = [scores[t][e] for e in idx[t]]
        m = max(z)
        ex = [math.exp(v - m) for v in z]
        s = sum(ex)
        raw.append([v / s for v in ex])
    flat = [e for row in idx for e in row]
    cap = len(flat) if capacity_factor is None else \
        max(1, math.ceil(capacity_factor * T * top_k / n))
    seen = [0] * n
    keep = [True] * len(flat)
    for i, e in enumerate(flat):
        if seen[e] >= cap:
            keep[i] = False
        seen[e] += 1
    weights = []
    for t in range(T):
        kept = [raw[t][j] if keep[t * top_k + j] else 0.0 for j in range(top_k)]
        denom = sum(kept)
        if denom > 0:
            weights.append([v / max(denom, 1e-12) for v in kept])
        else:
            weights.append([0.0] * top_k)
    counts = [sum(1 for e in flat if e == x) for x in range(n)]
    load = [min(c, cap) for c in counts]
    info = {"overflow_dropped": len(flat) - sum(load), "cap_per_expert": cap, "load": load}
    return idx, weights, info


def _router(W):
    r = Router.__new__(Router)
    r.W = W
    return r


def _maxerr(a, b, tol=2e-5):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    scale = max(1.0, a.abs().max().item(), b.abs().max().item())
    return (a - b).abs().max().item() <= tol * scale


# --------------------------------------------------------------------------
# route vs the slow reference
# --------------------------------------------------------------------------
_ROUTE_CASES = [
    # T, d, n, top_k, capacity_factor, disabled
    (8, 5, 7, 3, 1.25, ()),
    (4, 3, 4, 2, 0.5, ()),
    (6, 4, 3, 2, 0.5, ()),
    (5, 3, 3, 1, 0.2, ()),
    (3, 4, 5, 3, 2.0, (1,)),
    (2, 3, 4, 2, None, ()),
    (9, 6, 8, 4, 0.75, (0, 7)),
    (1, 3, 3, 2, 1.0, ()),
    (7, 4, 6, 6, 0.1, ()),   # top_k == n with an aggressive cap
]


@pytest.mark.parametrize("T,d,n,top_k,cf,disabled", _ROUTE_CASES)
def test_route_matches_slow_reference(T, d, n, top_k, cf, disabled):
    torch.manual_seed(1000 + 31 * T + 7 * n + top_k)
    W = torch.randn(d, n)
    X = torch.randn(T, d) * 1.5
    idx, weights, info, _cache = _router(W).route(X, top_k, cf, disabled)
    sidx, sw, sinfo = _slow_route(X, W, top_k, cf, disabled)
    assert idx.tolist() == sidx
    assert _maxerr(weights, torch.tensor(sw), tol=1e-6)
    assert info["overflow_dropped"] == sinfo["overflow_dropped"]
    assert info["cap_per_expert"] == sinfo["cap_per_expert"]
    assert info["load"] == sinfo["load"]
    assert info["overflow_dropped"] == T * top_k - sum(info["load"])
    assert idx.shape == (T, top_k) and idx.dtype == torch.long
    assert weights.shape == (T, top_k)
    # every row with at least one kept slot sums to one (renormalization)
    sums = weights.sum(dim=1)
    assert torch.allclose(sums[sums > 0], torch.ones_like(sums[sums > 0]), atol=1e-6)
    assert ((sums > 0) | (weights.abs().sum(1) == 0)).all()


@pytest.mark.parametrize("T,d,n,top_k,cf,disabled", [
    (4, 3, 4, 2, 1.0, ()),
    (3, 2, 3, 2, 0.5, (1,)),
    (5, 4, 5, 3, 0.25, ()),
])
def test_route_integer_ties_match_slow_reference(T, d, n, top_k, cf, disabled):
    torch.manual_seed(5)
    W = torch.randint(-2, 3, (d, n)).float()
    X = torch.randint(-2, 3, (T, d)).float()
    idx, weights, info, _ = _router(W).route(X, top_k, cf, disabled)
    sidx, sw, sinfo = _slow_route(X, W, top_k, cf, disabled)
    assert idx.tolist() == sidx
    assert _maxerr(weights, torch.tensor(sw), tol=1e-6)
    assert info["load"] == sinfo["load"] and info["overflow_dropped"] == sinfo["overflow_dropped"]


def test_route_tie_break_prefers_lower_expert_index():
    r = _router(torch.eye(4))
    X = torch.ones(2, 4)  # all four scores tie at 1.0
    idx, weights, _info, _ = r.route(X, 2, None)
    assert idx.tolist() == [[0, 1], [0, 1]]
    assert torch.allclose(weights, torch.full((2, 2), 0.5))
    # disabling the winner moves the tie to the next-lowest enabled expert
    idx2, _w2, _info2, _ = r.route(X, 1, None, disabled=(0,))
    assert idx2.tolist() == [[1], [1]]


# --------------------------------------------------------------------------
# capacity limiting
# --------------------------------------------------------------------------
def test_capacity_drops_latest_tokens_per_expert():
    # column 0 wins for every token; columns 1,2 tie for second (lower index wins)
    W = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    X = torch.tensor([[1.0, 0.0]] * 4)
    r = _router(W)
    idx, weights, info, _ = r.route(X, 2, 0.5)  # cap = max(1, ceil(0.5*4*2/3)) = 2
    assert idx.tolist() == [[0, 1]] * 4
    assert info["cap_per_expert"] == 2
    assert info["load"] == [2, 2, 0]
    assert info["overflow_dropped"] == 4
    raw = torch.softmax(torch.tensor([2.0, 0.0]), 0)
    # only the two earliest tokens per expert are kept; the rest are zero rows
    assert torch.allclose(weights[:2], raw.expand(2, 2), atol=1e-6)
    assert weights[2:].abs().max().item() == 0.0
    # without a cap the same routing keeps (and normalizes) every token
    idx_n, w_n, info_n, _ = r.route(X, 2, None)
    assert idx_n.tolist() == idx.tolist()
    assert info_n["overflow_dropped"] == 0 and info_n["cap_per_expert"] == 8
    assert torch.allclose(w_n, raw.expand(4, 2), atol=1e-6)


def test_capacity_factor_none_disables_limiting():
    torch.manual_seed(3)
    W = torch.randn(5, 6)
    X = torch.randn(12, 5)
    r = _router(W)
    idx, weights, info, _ = r.route(X, 6, None)
    assert info["overflow_dropped"] == 0
    assert info["cap_per_expert"] == 12 * 6
    # no drops => plain softmax over the top-k logits
    top = (X @ W).gather(1, idx)
    assert torch.allclose(weights, torch.softmax(top, 1), atol=1e-6)
    assert torch.allclose(weights.sum(1), torch.ones(12), atol=1e-6)


def test_all_dropped_row_is_zero_and_backward_finite():
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    X = torch.tensor([[10.0, 0.0], [5.0, 0.0], [4.0, 0.0]])
    r = _router(W)
    idx, weights, info, cache = r.route(X, 1, 1.0)  # cap = 2 -> row 2 fully dropped
    assert idx.tolist() == [[0], [0], [0]]
    assert weights.tolist() == [[1.0], [1.0], [0.0]]
    assert info["overflow_dropped"] == 1 and info["load"] == [2, 0]
    dX, dW = r.backward(torch.ones(3, 1), cache)
    assert torch.isfinite(dX).all().item() and torch.isfinite(dW).all().item()
    assert dX[2].abs().max().item() == 0.0  # dropped row contributes no gradient
    assert dW[:, 1].abs().max().item() == 0.0  # expert 1 was never selected


# --------------------------------------------------------------------------
# disabled experts + validation
# --------------------------------------------------------------------------
def test_disabled_experts_are_excluded():
    r = _router(torch.eye(4))
    X = torch.ones(3, 4)  # all scores tie at 1.0
    idx, _w, _info, _ = r.route(X, 2, None, disabled=(0, 2))
    assert idx.tolist() == [[1, 3]] * 3
    # disabled experts keep their column in the capacity denominator
    _idx, _w2, info2, _ = r.route(X, 2, 1.0, disabled=(0, 2))  # cap = ceil(3*2/4) = 2
    assert info2["cap_per_expert"] == 2
    # fewer enabled experts than top_k is rejected
    with pytest.raises(ValueError):
        r.route(X, 3, None, disabled=(0, 2))


def test_route_validation_errors():
    r = _router(torch.randn(4, 5))
    X = torch.randn(3, 4)
    for bad in (lambda: r.route(X, 0), lambda: r.route(X, 6),
                lambda: r.route(X, 5, disabled=(1,)),
                lambda: r.route(torch.tensor([[float("nan")] * 4]), 1),
                lambda: r.route(X, 1, disabled=(9,)),
                lambda: r.route(X, 1, disabled=(-1,))):
        with pytest.raises(ValueError):
            bad()
    inf = _router(torch.full((4, 5), float("inf")))
    with pytest.raises(ValueError):
        inf.route(X, 1)


# --------------------------------------------------------------------------
# backward vs autograd through the route forward
# --------------------------------------------------------------------------
_GRAD_CASES = [
    # T, d, n, top_k, capacity_factor, disabled, integer scores (ties)
    (8, 5, 7, 3, 1.25, (), False),
    (4, 3, 4, 2, 0.5, (), False),
    (6, 4, 3, 2, 0.5, (), False),
    (5, 3, 3, 1, 0.2, (), False),
    (5, 3, 3, 1, 0.2, (), True),
    (3, 4, 5, 3, 2.0, (1,), False),
    (2, 3, 4, 2, None, (), False),
    (9, 6, 8, 4, 0.75, (0, 7), False),
]


@pytest.mark.parametrize("T,d,n,top_k,cf,disabled,ties", _GRAD_CASES)
def test_router_backward_matches_autograd(T, d, n, top_k, cf, disabled, ties):
    torch.manual_seed(11 + 13 * T + n)
    if ties:
        W0 = torch.randint(-2, 3, (d, n)).float() * 1.5
        X0 = torch.randint(-2, 3, (T, d)).float()
    else:
        W0 = torch.randn(d, n) * 1.5
        X0 = torch.randn(T, d)
    Wg = W0.clone().requires_grad_(True)
    Xg = X0.clone().requires_grad_(True)
    r_auto = _router(Wg)
    idx, weights, info, _cache = r_auto.route(Xg, top_k, cf, disabled)
    dWeights = torch.randn(T, top_k)
    loss = (weights * dWeights).sum()
    loss.backward()
    assert Xg.grad is not None and Wg.grad is not None

    r = _router(W0)
    idx2, w2, info2, cache2 = r.route(X0, top_k, cf, disabled)
    assert idx2.tolist() == idx.tolist()
    assert _maxerr(w2, weights.detach(), tol=1e-6)
    assert info == info2
    dX, dW = r.backward(dWeights, cache2)
    assert _maxerr(dX, Xg.grad), (dX - Xg.grad).abs().max().item()
    assert _maxerr(dW, Wg.grad), (dW - Wg.grad).abs().max().item()


# --------------------------------------------------------------------------
# Model._moe aggregation vs an explicit per-slot loop
# --------------------------------------------------------------------------
@pytest.mark.parametrize("disabled,no_drop", [((), True), ((), False), ((0, 3), True)])
def test_model_moe_matches_explicit_slot_loop(disabled, no_drop):
    spec = cfg.PRESETS["moe-tiny"]
    model = M.Model(spec, M.build_params(spec, generator=torch.Generator().manual_seed(8)))
    torch.manual_seed(0)
    X = torch.randn(5, spec.d_model) * 0.7
    Xg = X.clone().requires_grad_(True)
    Y, info = model._moe(Xg, disabled, no_drop)

    r = _router(model.p["router"])
    cf = None if no_drop else model.capacity_factor
    idx, weights, info2, _ = r.route(Xg, spec.top_k, cf, disabled)
    assert info == info2
    Y2 = torch.zeros_like(Xg)
    for t in range(X.shape[0]):
        for j in range(spec.top_k):
            yb = model._expert(Xg[t:t + 1], int(idx[t, j]))
            Y2 = Y2.index_add(0, torch.tensor([t]), weights[t, j] * yb)
    assert torch.allclose(Y, Y2, atol=1e-5)

    used = sorted(set(int(e) for e in idx.reshape(-1)))
    params = [Xg, model.p["router"]] + \
        [model.p[f"expert.{e}.Win"] for e in used] + \
        [model.p[f"expert.{e}.Wout"] for e in used]
    R = torch.randn_like(Y)
    g1 = torch.autograd.grad((Y * R).sum(), params, allow_unused=True)
    g2 = torch.autograd.grad((Y2 * R).sum(), params, allow_unused=True)
    for p_, a, b in zip(params, g1, g2):
        assert (a is None) == (b is None)
        if a is not None:
            assert _maxerr(a, b, tol=1e-5), p_


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def test_router_constructor_deterministic():
    a, b = Router(4, 3, rng=0), Router(4, 3, rng=0)
    assert torch.equal(a.W, b.W)
    g = torch.Generator().manual_seed(7)
    c = Router(4, 3, rng=g)
    ref = torch.randn(4, 3, generator=torch.Generator().manual_seed(7)) / 2.0
    assert torch.allclose(c.W, ref, atol=0)
    assert a.n_experts == 3 and a.W.shape == (4, 3)


def test_softmax_rows_and_loss_helpers():
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    assert torch.allclose(moe.softmax_rows(x), torch.softmax(x, dim=1))
    assert torch.allclose(moe.softmax_rows(x).sum(1), torch.ones(4), atol=1e-6)

    n, T, k = 5, 9, 2
    idx = torch.randint(0, n, (T, k))
    w = torch.softmax(torch.randn(T, k), 1)
    got = moe.balance_aux_loss(w, idx, n)
    counts = torch.bincount(idx.reshape(-1), minlength=n).float()
    sums = torch.zeros(n).index_add_(0, idx.reshape(-1), w.reshape(-1))
    want = n * ((counts / T) * (sums / T)).sum().item()
    assert abs(got - want) <= 1e-7
    # balanced routing of one token per expert => aux loss 1.0
    idxb = torch.arange(n).unsqueeze(1)
    wb = torch.ones(n, 1)
    assert abs(moe.balance_aux_loss(wb, idxb, n) - 1.0) <= 1e-7
    assert moe.balance_aux_loss(torch.zeros(0, 2), torch.zeros(0, 2, dtype=torch.long), n) == 0.0

    st = moe.expert_output_stats(idx, w, n)
    assert st["tokens_total"] == T
    assert st["per_expert_counts"] == counts.tolist()
    assert abs(st["mean_load"] - counts.mean().item()) <= 1e-9
    assert st["min_load"] == int(counts.min()) and st["max_load"] == int(counts.max())
    assert st["unused_experts"] == [e for e in range(n) if counts[e].item() == 0]
