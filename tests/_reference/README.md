# tests/_reference — quarantined archive

This directory is excluded from the default suite (`norecursedirs` in
`pytest.ini`).  It holds the port-era reference suites written while the
runtime still used PyTorch tensors.  Run it explicitly with
`python -m pytest -q tests/_reference` when touching the archived paths.

## Current state (measured 2026-10-28, this tree vs the `373b2c5` baseline)

- 263 passed / 223 failed — **identical to the baseline rot set**, so none of
  it is recent breakage.  One failure that *was* introduced this session
  (`test_kernels_build_missing_source`) was fixed in `a752226`.
- The passing tests are live canaries: they compare the NumPy runtime against
  the archived PyTorch implementation (torch is installed) and stay green.

## Failure taxonomy (pre-existing rot, ~4 root causes)

1. `TypeError: generator must be a numpy Generator, integer seed, or legacy
   generator` (84 tests) — fixtures pass torch generators to APIs that now
   require NumPy RNGs.
2. torch↔numpy boundary idioms (~70): `ndarray.numpy()`, `Tensor.detach()`
   on ndarrays, `ndarray @ Tensor`, `torch.equal` on mixed tensors.
3. `SmaulOPT parameters must be NumPy arrays or Tensors` (~12) — fixtures
   build `torch.nn` models and hand them to the NumPy optimizer.
4. stale expectations (~30): legacy `.pt`/`.npz` checkpoint layouts, dtype
   asserts (`dtype == torch.float32`), retired APIs.

## Disposition

- Semantics worth keeping are pinned by the live suite (`tests/`, 121 tests);
  portable gaps found during triage were ported there.
- The remainder documents the torch-era behaviour of retired APIs and stays
  as-is: rewriting 223 fixtures would buy no new coverage of the live
  runtime.  Do not "fix" these tests to make them pass — they encode an API
  that no longer exists.
