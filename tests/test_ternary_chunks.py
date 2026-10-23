import numpy as np
import pytest
import torch

from smaulbrain import _quant_native
from smaulbrain.quant import sample_ternary_gradient_update_chunk


def test_chunk_sampler_preserves_global_importance_and_maps_indices(monkeypatch):
    calls = []
    expected_local_indices = np.array([0, 3, 5], dtype=np.int32)
    expected_directions = np.array([-1, 1, -1], dtype=np.int8)

    monkeypatch.setattr(_quant_native, "prepare_ternary_backend", lambda: True)

    def fake_native_sampler(gradient, rate, seed):
        calls.append((gradient.copy(), rate, seed))
        return expected_local_indices, expected_directions

    monkeypatch.setattr(
        _quant_native, "sample_ternary_gradient_updates", fake_native_sampler
    )
    gradient = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
    )

    indices, directions = sample_ternary_gradient_update_chunk(
        gradient,
        rate=0.2,
        seed=17,
        global_mean_abs=7.0,
        full_cols=5,
        column_offset=2,
    )

    # Local flat indices 0, 3, 5 map to (row, col) coordinates (0,0),
    # (1,1), (2,1) in the two-column chunk, then to global columns 2, 3, 3.
    assert indices.tolist() == [2, 8, 13]
    assert directions.tolist() == [-1, 1, -1]
    assert len(calls) == 1
    assert calls[0][1] == pytest.approx(0.1)  # .2 * chunk_mean(3.5) / global_mean(7)
    assert calls[0][2] == 17


def test_chunk_sampler_returns_empty_for_zero_global_mean(monkeypatch):
    monkeypatch.setattr(
        _quant_native,
        "sample_ternary_gradient_updates",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not sample")),
    )
    indices, directions = sample_ternary_gradient_update_chunk(
        torch.zeros(3, 2),
        rate=0.2,
        seed=1,
        global_mean_abs=0.0,
        full_cols=5,
        column_offset=2,
    )
    assert indices.dtype == torch.int32
    assert directions.dtype == torch.int8
    assert indices.numel() == directions.numel() == 0


@pytest.mark.parametrize(
    "full_cols,column_offset",
    [(1, 0), (5, -1), (5, 4)],
)
def test_chunk_sampler_rejects_invalid_global_layout(full_cols, column_offset):
    with pytest.raises(ValueError):
        sample_ternary_gradient_update_chunk(
            torch.ones(3, 2),
            rate=0.1,
            seed=1,
            global_mean_abs=1.0,
            full_cols=full_cols,
            column_offset=column_offset,
        )
