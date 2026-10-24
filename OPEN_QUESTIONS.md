# Open questions

Items below are intentionally not implemented rather than guessed at. See
`PORT_NOTES.md` for the numeric and checkpoint deviations.

1. `--pagingmthd`, adaptive halting depth, and `--trunk-lr-mult` do not exist in
   this repository's source or CLI. No behavior was invented for them.
2. CUDA execution, CUDA/VRAM paging (`V`, `D2V`, `R2V`), and hardware FP8 are
   unavailable in the CPU-only NumPy build. Those paging modes raise
   `PagingError`.
3. PyTorch RNG streams and bit-exact fused-kernel arithmetic are not reproduced.
   Existing weights load exactly, but freshly seeded runs differ.
4. Full-preset convergence, memory use, and throughput on the target Ivy Bridge
   CPU have not been measured. The port is validated with tiny presets and
   parity tests.
5. Legacy PyTorch `.pt` checkpoints require the development-only converter in
   `tests/_reference/`; runtime loading intentionally supports NumPy/safetensors
   and legacy `.npz` only.
