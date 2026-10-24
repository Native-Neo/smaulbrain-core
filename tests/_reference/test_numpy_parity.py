"""Development-only PyTorch parity checks for the core NumPy primitives.

Run explicitly with:
    PYTHONPATH=. pytest -q tests/_reference/test_numpy_parity.py
"""
import numpy as np
import pytest
import torch

from smaulbrain import npgrad as ng
from smaulbrain import ops


def test_linear_forward_backward_parity():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((2, 5, 7)).astype(np.float32)
    w_np = rng.standard_normal((4, 7)).astype(np.float32)
    b_np = rng.standard_normal(4).astype(np.float32)
    g_np = rng.standard_normal((2, 5, 4)).astype(np.float32)

    x_t = torch.from_numpy(x_np.copy()).requires_grad_(True)
    w_t = torch.from_numpy(w_np.copy()).requires_grad_(True)
    b_t = torch.from_numpy(b_np.copy()).requires_grad_(True)
    y_t = torch.nn.functional.linear(x_t, w_t, b_t)
    (y_t * torch.from_numpy(g_np)).sum().backward()

    x_n = ng.Tensor(x_np.copy(), True)
    w_n = ng.Tensor(w_np.copy(), True)
    b_n = ng.Tensor(b_np.copy(), True)
    y_n = ng.linear(x_n, w_n, b_n)
    ng.sum(y_n * ng.Tensor(g_np)).backward()

    np.testing.assert_allclose(y_n.data, y_t.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x_n.grad, x_t.grad.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(w_n.grad, w_t.grad.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b_n.grad, b_t.grad.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("length", [1, 5, 130])
def test_scan_forward_backward_parity(length):
    rng = np.random.default_rng(length)
    v_np = rng.standard_normal((2, length, 6)).astype(np.float32)
    a_np = rng.uniform(0.1, 0.99, size=6).astype(np.float32)
    g_np = rng.standard_normal((2, length, 6)).astype(np.float32)

    v_t = torch.from_numpy(v_np.copy()).requires_grad_(True)
    a_t = torch.from_numpy(a_np.copy()).requires_grad_(True)
    state = torch.zeros(2, 6)
    rows = []
    for t in range(length):
        state = a_t * state + v_t[:, t]
        rows.append(state)
    y_t = torch.stack(rows, dim=1)
    (y_t * torch.from_numpy(g_np)).sum().backward()

    v_n = ng.Tensor(v_np.copy(), True)
    a_n = ng.Tensor(a_np.copy(), True)
    y_n = ng.scan(v_n, a_n)
    ng.sum(y_n * ng.Tensor(g_np)).backward()

    np.testing.assert_allclose(y_n.data, y_t.detach().numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(v_n.grad, v_t.grad.numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)


def test_rmsnorm_parity():
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((2, 7)).astype(np.float32)
    g_np = rng.standard_normal(7).astype(np.float32)
    dy_np = rng.standard_normal((2, 7)).astype(np.float32)
    y_t = torch.nn.functional.rms_norm(
        torch.from_numpy(x_np), (7,), torch.from_numpy(g_np), eps=1e-6
    )
    y_n, cache = ops.rmsnorm_fwd(x_np, g_np)
    np.testing.assert_allclose(y_n.data, y_t.numpy(), rtol=1e-5, atol=1e-5)
    dx_n, dg_n = ops.rmsnorm_bwd(dy_np, cache)
    x_t = torch.from_numpy(x_np.copy()).requires_grad_(True)
    g_t = torch.from_numpy(g_np.copy()).requires_grad_(True)
    y_t = torch.nn.functional.rms_norm(x_t, (7,), g_t, eps=1e-6)
    (y_t * torch.from_numpy(dy_np)).sum().backward()
    np.testing.assert_allclose(dx_n.data, x_t.grad.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dg_n.data, g_t.grad.numpy(), rtol=1e-5, atol=1e-5)
