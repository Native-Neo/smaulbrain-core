"""Regression tests for the minor-audit fixes integrated in the 3.3.0 release.

Each test below reproduces a defect that was fixed in this batch and fails on
the pre-fix baseline.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from smaulbrain import config as cfg, inference as I, model as M, kernels
from smaulbrain import _quant_native as qn


def _dense_tiny():
    spec = cfg.PRESETS["dense-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(0), 0, mtp_depth=2)
    return M.Model(spec, params, mtp_depth=2)


def test_engine_rejects_nan_temperature():
    model = _dense_tiny()
    with pytest.raises(ValueError):
        I.InferenceEngine(model, temperature=float("nan"))


def test_engine_rejects_negative_temperature():
    model = _dense_tiny()
    with pytest.raises(ValueError):
        I.InferenceEngine(model, temperature=-0.1)


def test_kernels_try_load_ignores_partial_builds(tmp_path, monkeypatch):
    # A truncated build artifact must never shadow a complete library.
    import glob
    pkg = Path(os.path.dirname(kernels.__file__))
    # Build a real complete library first.
    kernels.build_here(build_dir=str(tmp_path), verbose=False)
    # Drop a partial artifact in the package dir.
    partial = pkg / "smaul_kernels-mavx-9999999999999999.tmp.so"
    partial.write_bytes(b"short")
    try:
        lib, info = kernels.try_load()
        # try_load is memoized on a missing-library result; clear and re-load.
        kernels.try_load.cache_clear()
        lib, info = kernels.try_load()
        assert info.get("available") is True, info
    finally:
        partial.unlink(missing_ok=True)


def test_kernels_build_dir_other_filesystem(tmp_path, monkeypatch):
    # Build into a dir that is not the package dir; must not crash with
    # cross-device rename errors.
    import glob
    pkg = Path(os.path.dirname(kernels.__file__))
    for old in glob.glob(str(pkg / "smaul_kernels*.so")):
        os.remove(old)
    kernels.try_load.cache_clear()
    target = tmp_path / "build"
    target.mkdir()
    path = kernels.build_here(build_dir=str(target), verbose=False)
    assert os.path.isfile(path)
    assert path.endswith(".so")
