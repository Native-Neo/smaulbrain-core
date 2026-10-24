# Port notes — smaulbrain-core

The runtime is pure NumPy plus the Python standard library. PyTorch remains only
under `tests/_reference/` as a development oracle for parity checks.

## Numeric deviations

- **float32 runtime.** Parameters, activations, gradients, and optimizer state
  are `float32`. Reductions that the original computed in `float64` (RMSNorm
  statistics, gradient norms, loss normalization, optimizer norm tracking) are
  still accumulated in `float64`. BLAS and NumPy may sum in a different order
  than the original kernels, so results can differ in the last few bits.
- **Autograd implementation.** `smaulbrain.npgrad` records a lightweight tape
  with explicit backward closures. Broadcasting gradients are reduced over all
  leading dimensions; floating-point dtype casts remain differentiable; Python
  scalar constants are kept at float32. Primitive parity tests cover linear,
  recurrent scan, and RMSNorm forward/backward at `rtol` 1e-5 to 5e-5.
- **Recurrent scan.** The original time loop / native scan is replaced by a
  chunked closed-form scan (`npgrad.scan`). The chunk size adapts to the decay
  range, so summation order differs from a token-by-token loop. The math is the
  same, and the scan is parity-tested against PyTorch.
- **RNG streams.** NumPy `Generator` (PCG64) replaces the PyTorch RNG. Existing
  checkpoints load weights exactly, but freshly seeded runs do not reproduce
  PyTorch's random stream. Checkpoints record and restore the NumPy bit
  generator state, so NumPy runs resume exactly.
- **FP8.** E4M3 is software-emulated with explicit scaling, subnormals,
  saturation at 448, and NaN/Inf sentinels. It is not hardware FP8 arithmetic.
- **Ternary.** Weights use the original rule: threshold
  `0.7 * mean(abs(W))`, codes in {-1, 0, +1}, scale `mean(abs(W))` over
  nonzeros, packed four codes per byte. Packed matmul avoids reconstructing a
  dense weight matrix. The optional C++ kernel and the portable NumPy path use
  the same layout; the fallback is used when no compiler is available.
- **SmaulOPT.** The factored optimizer keeps the original first moment and
  row/column second-moment structure. The optional fused C++ update is reached
  through NumPy buffers; without a compiler or for non-qualifying tensors the
  NumPy reference update runs instead. Rounding can differ from the original
  fused kernel.
- **Interpolation.** `F.interpolate(..., mode="linear", align_corners=False)`
  and `area` / `adaptive_avg_pool1d` are reimplemented and parity-tested; edge
  sampling follows PyTorch's half-pixel coordinates.

## Checkpoint and file-format changes

- New checkpoints are directories containing `manifest.json`,
  `params.safetensors`, and `optim.safetensors`. The version string is
  `smaulbrain-numpy-safetensors-v1:1`. The manifest keeps checksums, model and
  optimizer metadata, data cursor, and NumPy RNG state.
- Legacy `smaulbrain-v2:1` `params.npz` / `optim.npz` checkpoints remain
  readable; the next save writes the safetensors format.
- Legacy PyTorch `.pt` checkpoints are not read at runtime. Convert them with
  `tests/_reference/convert_torch_checkpoint.py` (development-only; it imports
  PyTorch).
- Paged expert files are `expert_<id>.safetensors`. Legacy per-expert `.npz`
  files are still read; legacy `.pt` expert files must be converted.

## Device and performance behavior

- The build is CPU-only. `--device cuda` is accepted for CLI compatibility but
  executes on CPU; there is no CUDA backend. `paging.accelerator_available()`
  always reports no accelerator, so `V`, `D2V`, and `R2V` paging raise
  `PagingError`.
- Optional C++ kernels are compiled on first use with SSE2/AVX flags appropriate
  for the host. No AVX2-only path is required. If compilation fails, the NumPy
  paths remain functional.
- Full presets require substantial RAM. Use the tiny presets for CPU
  development and benchmark the actual machine before selecting larger models.

## Unsupported items

- `--pagingmthd`, adaptive halting depth, and `--trunk-lr-mult` are not present
  in this repository; they were not invented during the port.
- CUDA execution, CUDA paging, and hardware FP8 are not supported.
- PyTorch RNG continuation is not reproduced; see the RNG note above.
