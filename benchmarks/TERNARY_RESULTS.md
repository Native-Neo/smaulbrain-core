# Packed Ternary Forward/Backward: Results

This report benchmarks the packed 2-bit ternary path against the all-FP32
path on the `dense-tiny` preset. It measures model forward and backward
separately through `Model.loss_and_backward(profile=True)`.

The keyword here is **packed**: ternary weights are consumed directly from
their tiled 2-bit buffer. No dense FP32/BF16 weight matrix is reconstructed,
and the same portable fallback remains available when the C++ backend cannot
be compiled.

## Summary

On the dense-tiny model at sequence 512, with 15 measured steps after three
warm-up steps and three mode-order-rotated runs:

- **Forward:** 127.8 ms FP32 vs 70.8 ms ternary. Ternary forward is **1.81x
  faster**.
- **Backward:** 60.9 ms FP32 vs 65.6 ms ternary. The packed input-gradient
  kernel is offset by the unchanged FP32 straight-through weight GEMM, so
  backward is **7.7% slower**.
- **Full optimizer step:** 197.6 ms FP32 vs 146.1 ms ternary. Including the
  optimizer, gradient extraction, and invalidation/re-packing of changed
  ternary weights, the step is **1.35x faster**.

At sequence 64, forward is 1.23x faster, backward is 1.02x faster, and the
full step is 1.12x faster. The fixed cost of re-packing every changed weight
is a larger fraction of the shorter step.

With weights frozen after warm-up (pure forward/backward arithmetic and a warm
packed cache), sequence 512 measures 1.86x faster forward, 0.95x backward, and
1.40x for the complete step. Sequence 64 measures 1.34x, 0.91x, and 1.08x
respectively.

**Interpretation:** ternary is now faster than FP32 for complete training
steps on the measured dense workload, driven by a substantially faster
forward. The remaining backward cost is the STE weight gradient, which is a
normal FP32 GEMM in both modes; packed storage cannot accelerate it.

## Measured model timings

`benchmarks/ternary_benchmark.py`, `dense-tiny`, 15 steps, 3 warm-ups, median
of 3 runs with mode order rotated between runs. Training mode runs the
optimizer and invalidates the packed cache before each next step; frozen mode
skips both.

### Training step (includes optimizer and re-packing)

| sequence | mode | forward | backward | step | forward speedup | backward speedup | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | FP32 | 24.7 ms | 21.5 ms | 56.7 ms | 1.00x | 1.00x | 1.00x |
| 64 | ternary | 20.1 ms | 21.1 ms | 50.5 ms | 1.23x | 1.02x | 1.12x |
| 512 | FP32 | 127.8 ms | 60.9 ms | 197.6 ms | 1.00x | 1.00x | 1.00x |
| 512 | ternary | 70.8 ms | 65.6 ms | 146.1 ms | 1.81x | 0.93x | 1.35x |

### Frozen weights (pure forward/backward, warm cache)

| sequence | mode | forward | backward | step | forward speedup | backward speedup | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | FP32 | 25.5 ms | 21.6 ms | 49.5 ms | 1.00x | 1.00x | 1.00x |
| 64 | ternary | 19.1 ms | 23.6 ms | 45.9 ms | 1.34x | 0.91x | 1.08x |
| 512 | FP32 | 145.1 ms | 74.9 ms | 221.8 ms | 1.00x | 1.00x | 1.00x |
| 512 | ternary | 77.9 ms | 79.1 ms | 158.6 ms | 1.86x | 0.95x | 1.40x |

The `step` column also includes copying gradients out of the graph and, in
training mode, the optimizer and cache invalidation. The profiled
`forward`/`backward` columns are the model's own phase timers.

## Packed-kernel changes

The gains come from the native tiled kernels and their Python bridge, not from
weakening the numerical contract:

- Work is partitioned over both output tiles and batch blocks, so small
  matrices use more than one worker. The old kernels only split by output tile
  (forward) or input-row range (backward).
- The low-work threshold for the native pool is 65,536 MACs for tiled
  matmuls, so small attention matrices at short sequences also use workers.
- Backward reads `dY` directly when `dout` is a multiple of the 16-column
  tile width. The old path allocated and copied a padded `batch * ntiles * 16`
  buffer even for aligned model shapes.
- The first input-gradient tile stores directly into `dx`; later tiles
  accumulate. This removes a full zero-fill pass and one read of `dx`.
- The hot tiled/packing entry points bind raw `c_void_p` addresses instead of
  creating a `ctypes` pointer object for every tensor and call.
- The quantized linear computes the FP32 STE weight gradient before launching
  the packed input-gradient kernel, so native workers do not spin against the
  subsequent OpenBLAS GEMM on the same cores.

### Kernel-only A/B

Direct C calls with a warm packed cache and preallocated output buffers.
Batch 512 is the model's training sequence. Times are microseconds, median of
four repeated batches of 1,200 calls; lower is better.

| shape | fwd old | fwd new | fwd speedup | dX old | dX new | dX speedup |
|---|---:|---:|---:|---:|---:|---:|
| 32x32 | 52.0 | 32.5 | 1.60x | 67.1 | 37.4 | 1.79x |
| 32x128 | 116.0 | 103.7 | 1.12x | 313.7 | 113.7 | 2.76x |
| 128x32 | 86.2 | 85.0 | 1.01x | 178.7 | 130.8 | 1.37x |

At batch 64, the input-gradient speedups are 1.08x (32x32), 1.64x (32x128),
and 1.19x (128x32); forward speedups are 1.12x, 1.32x, and 1.45x. "Old" is
the parent commit's kernel; "new" is this checkout.

## Reproduce

The measurements were taken on an Intel i3-3220 (2 cores / 4 threads, AVX1,
no usable FMA), with both the native pool and OpenBLAS/OMP limited to two
threads and the process pinned to four logical CPUs:

```bash
PYTHONPATH=$PWD SMAULBRAIN_THREADS=2 OPENBLAS_NUM_THREADS=2 \
OMP_NUM_THREADS=2 taskset -c 0-3 \
python benchmarks/ternary_benchmark.py --steps 15 --warmup 3 --runs 3 \
    --seq-lens 64 512 --modes fp32 ternary \
    --json-out benchmarks/ternary_results.json
```

Add `--frozen-weights` to measure pure forward/backward arithmetic without
parameter updates or re-packing:

```bash
PYTHONPATH=$PWD SMAULBRAIN_THREADS=2 OPENBLAS_NUM_THREADS=2 \
OMP_NUM_THREADS=2 taskset -c 0-3 \
python benchmarks/ternary_benchmark.py --steps 15 --warmup 3 --runs 3 \
    --seq-lens 64 512 --modes fp32 ternary --frozen-weights \
    --json-out benchmarks/ternary_results_frozen.json
```

Raw medians and per-step samples are in
[`ternary_results.json`](ternary_results.json) and
[`ternary_results_frozen.json`](ternary_results_frozen.json).

## Caveats

- This is a CPU-only NumPy implementation. The native kernel uses AVX1 when
  available and falls back to SSE2/scalar code; it does not use AVX2, AVX-512,
  or hardware FMA. The host CPU reports `fma` in `/proc/cpuinfo` but faults on
  an actual FMA instruction, so the build deliberately uses
  `-ffp-contract=off` and no FMA.
- Absolute times move with background load and CPU frequency. The recorded
  load average after the final run was about 1.9-2.2 on a four-thread desktop;
  the ratio between configurations is more stable than the absolute numbers.
- Dense FP32 backward is a highly optimized OpenBLAS GEMM. Ternary dX is
  faster than the old packed kernel but still not universally faster than
  dense BLAS, so backward speedup depends on matrix shape and batch size.
- No quality or convergence claim is made here. Ternarizing weights changes
  the function being trained; this report measures only execution time.
- `backward_mode=fp32` and `backward_mode=ternary` currently share the same
  packed input-gradient and FP32 STE weight-gradient path. Only `fp8` selects
  a distinct backward implementation. The benchmark uses `ternary`.
