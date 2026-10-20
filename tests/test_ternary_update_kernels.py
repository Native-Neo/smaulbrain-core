import numpy as np
import pytest

from smaulbrain import _quant_native as native


@pytest.fixture(scope="module", autouse=True)
def require_native_backend():
    if not native.prepare_ternary_backend():
        pytest.skip("native C++ ternary backend is unavailable")


def test_native_sampler_handles_transposed_gradient_without_copy_contract():
    base = np.arange(1, 49, dtype=np.float32).reshape(6, 8)
    gradient = base.T  # Non-contiguous logical (8, 6) gradient.
    indices1, directions1 = native.sample_ternary_gradient_updates(
        gradient, rate=0.25, seed=123456
    )
    indices2, directions2 = native.sample_ternary_gradient_updates(
        gradient, rate=0.25, seed=123456
    )

    assert np.array_equal(indices1, indices2)
    assert np.array_equal(directions1, directions2)
    assert indices1.dtype == np.int32
    assert directions1.dtype == np.int8
    assert len(indices1) > 0
    assert np.all(indices1 >= 0)
    assert np.all(indices1 < gradient.size)
    selected = gradient.reshape(-1)[indices1]
    assert np.array_equal(directions1, -np.sign(selected).astype(np.int8))


def test_native_apply_ternary_update_segments_combines_conflicting_votes():
    packed, _, shape, _ = native.pack_ternary(np.zeros((1, 12), dtype=np.float32))
    assert shape == (1, 12)
    before = packed.copy()

    changed = native.apply_ternary_update_segments(
        packed,
        n_weights=12,
        index_segments=[
            np.array([0, 1, 4, 7], dtype=np.int32),
            np.array([1, 4, 7], dtype=np.int32),
        ],
        direction_segments=[
            np.array([1, 1, -1, 1], dtype=np.int8),
            np.array([-1, -1, -1], dtype=np.int8),
        ],
    )

    # Index 0 moves 0->+1. Votes for index 1 cancel. The two negative
    # votes for index 4 combine into one move to -1; index 7 cancels.
    codes = native.unpack_codes(packed, shape).reshape(-1)
    assert changed == 2
    assert codes[0] == 1
    assert codes[1] == 0
    assert codes[4] == -1
    assert codes[7] == 0
    assert not np.array_equal(packed, before)


def test_native_apply_ternary_update_rejects_oversized_declared_shape():
    packed = np.array([0x55], dtype=np.uint8)
    before = packed.copy()
    with pytest.raises(ValueError, match="packed buffer is too short"):
        native.apply_ternary_update_segments(
            packed, 5,
            [np.array([4], dtype=np.int32)],
            [np.array([1], dtype=np.int8)],
        )
    assert np.array_equal(packed, before)


def test_native_apply_ternary_update_rejects_invalid_index():
    packed = np.array([0x55], dtype=np.uint8)
    with pytest.raises(RuntimeError, match="status 2"):
        native.apply_ternary_update_segments(
            packed, 4,
            [np.array([4], dtype=np.int32)],
            [np.array([1], dtype=np.int8)],
        )
