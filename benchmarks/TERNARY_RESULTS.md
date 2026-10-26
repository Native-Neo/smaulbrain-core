# Packed Ternary Forward/Backward: Results

This report benchmarks the packed 2-bit ternary path against the all-FP32
path on the `dense-tiny` preset. It measures model forward and backward
separately through `Model.loss_and_backward(profile=True)`.

The keyword here is **packed**: ternary weights are consumed directly from
their tiled 2-bit buffer. No dense FP32/BF16 weight matrix is reconstructed,
and the same portable fallback remains available when the C++ backend cannot
be compiled. The benchmark imports and drives the production `Model`,
`npgrad`, and `SmaulOPT` path, including the training-loop invalidation order;
only the kernel-only A/B table below calls the C entry points directly.

> **Revision note.** The clean-load end-to-end tables below are the original
> packed-kernel run (load average 1.9-2.2). Since then the cached packing path
> was batched and SiLU stopped allocating full-size temporaries; both changes
> are measured in their own sections. A fresh end-to-end run was taken with a
> concurrent Java workload on the host (load average 2.8-4.9), so its absolute
> times are inflated and are reported only as a directional check. Those
> reruns are stored as `*_loaded.json` and `ternary_phase_results.json`. They
> predate the final removal of a redundant single-matrix copy; that cleanup
> only lowers packing cost. The component A/B measurements are the useful
> result for this revision.

## Summary

On the dense-tiny model at sequence 512, in the clean-load run with 15
measured steps after three warm-up steps and three mode-order-rotated runs:

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

**Interpretation:** ternary is faster than FP32 for complete training steps on
the measured dense workload, driven by a substantially faster forward. The
remaining backward cost is the STE weight gradient, which is a normal FP32
GEMM in both modes; packed storage cannot accelerate it. The clean-load
numbers are conservative for this revision because both batching and in-place
SiLU reduce the ternary and FP32 forward paths further.

## Measured model timings

`benchmarks/ternary_benchmark.py`, `dense-tiny`, 15 steps, 3 warm-ups, median
of 3 runs with mode order rotated between runs. Training mode runs the
optimizer and invalidates the packed cache before each next step; frozen mode
skips both. These are the original clean-load measurements; see the current
loaded-host check below.

### Clean-load training step (includes optimizer and re-packing)

| sequence | mode | forward | backward | step | forward speedup | backward speedup | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | FP32 | 24.7 ms | 21.5 ms | 56.7 ms | 1.00x | 1.00x | 1.00x |
| 64 | ternary | 20.1 ms | 21.1 ms | 50.5 ms | 1.23x | 1.02x | 1.12x |
| 512 | FP32 | 127.8 ms | 60.9 ms | 197.6 ms | 1.00x | 1.00x | 1.00x |
| 512 | ternary | 70.8 ms | 65.6 ms | 146.1 ms | 1.81x | 0.93x | 1.35x |

### Clean-load frozen weights (pure forward/backward, warm cache)

| sequence | mode | forward | backward | step | forward speedup | backward speedup | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | FP32 | 25.5 ms | 21.6 ms | 49.5 ms | 1.00x | 1.00x | 1.00x |
| 64 | ternary | 19.1 ms | 23.6 ms | 45.9 ms | 1.34x | 0.91x | 1.08x |
| 512 | FP32 | 145.1 ms | 74.9 ms | 221.8 ms | 1.00x | 1.00x | 1.00x |
| 512 | ternary | 77.9 ms | 79.1 ms | 158.6 ms | 1.86x | 0.95x | 1.40x |

The `step` column also includes copying gradients out of the graph and, in
training mode, the optimizer and cache invalidation. The profiled
`forward`/`backward` columns are the model's own phase timers.

### Current loaded-host check

The same training command was repeated after the batching and in-place SiLU
changes, while a Java workload was using the host (recorded load average 4.17
/ 3.61 / 2.80). Absolute times are much larger than the clean-load run, so the
two tables must not be compared row by row.

| sequence | mode | forward | backward | step | forward speedup | backward speedup | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | FP32 | 33.7 ms | 31.3 ms | 79.1 ms | 1.00x | 1.00x | 1.00x |
| 64 | ternary | 30.6 ms | 35.9 ms | 79.7 ms | 1.10x | 0.87x | 0.99x |
| 512 | FP32 | 187.8 ms | 116.3 ms | 320.5 ms | 1.00x | 1.00x | 1.00x |
| 512 | ternary | 113.3 ms | 128.0 ms | 251.2 ms | 1.66x | 0.91x | 1.28x |

The current frozen run (load 4.86 / 3.77 / 2.99) measured 1.11x / 0.87x /
1.00x at sequence 64 and 1.52x / 0.86x / 1.20x at sequence 512 for
forward/backward/step. The contention penalizes the native worker pool more
than OpenBLAS, which is why the short-sequence step ratio falls to parity.
The component A/B measurements below isolate the actual code changes. These
two training/frozen reruns are in
[`ternary_results_loaded.json`](ternary_results_loaded.json) and
[`ternary_results_frozen_loaded.json`](ternary_results_frozen_loaded.json).

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

## Batched cache refill

Training invalidates the packed representation of every weight touched by the
optimizer. The old next forward rebuilt those entries with 105 separate
ctypes calls to `pack_ternary_tiled`, then copied each returned buffer. The new
path records the invalidated references in `PackedCache.stale`, then
`Model.loss_and_backward` calls `prefetch()` before the forward graph is built.
`pack_ternary_tiled_many` packs all matrices into one uint8 allocation in one
native call and returns views; no second copy is needed. Entries for dynamic
MoE experts are excluded and packed lazily on demand.

Invalid entries are never usable as hits: `get()` still rebuilds on a ref
mismatch, a full `invalidate()` clears the stale set, and a failed batch raises
instead of silently serving stale weights. Tests cover batch/individual byte
parity, cache invalidation, prefetch parity, expert exclusion, and the actual
ternary training path.

Direct pack of the 105 dense-tiny matrices (CPU time, representative median
of 30 calls; repeated runs varied by about 0.2 ms):

| variant | median | minimum |
|---|---:|---:|
| per-matrix calls | 2.24 ms | 1.99 ms |
| batched native call | 1.17 ms | 0.91 ms |

The single-matrix path also no longer copies the freshly packed buffer, so
even lazy expert packing dropped from roughly 2.9 ms to 2.2-2.6 ms for this
set. The end-to-end effect is smaller because forward work dominates. In a
same-process A/B of cold forwards after invalidation (12 alternating pairs,
sequence 512), lazy packing took 156.9 ms median versus 153.8 ms with
prefetch. The two paths are logically identical; the difference is the ~1.5 ms
of packing and wrapper work.

Phase instrumentation from the loaded current run:

| sequence | forward | of which pack | cache lookup + prefetch | optimizer | invalidation | step |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 35.02 ms | 2.06 ms | 2.43 ms | 11.84 ms | 0.21 ms | 88.54 ms |
| 512 | 125.19 ms | 1.59 ms | 2.09 ms | 11.45 ms | 0.18 ms | 272.20 ms |

The earlier phase run measured 3.35 ms of packing at sequence 64 and
3.9-4.4 ms at sequence 512, at lower background load. Compare the direct CPU
A/B above, not those absolute values.

## In-place SiLU

`silu` previously allocated a float64 copy, the sigmoid, the product, the
output cast, and (in full-precision storage) the held derivative. It now runs
the same ufuncs in place on the float64 copy, in the same operand order, and
uses the original fp32 activation for the final multiply. The reduced-storage
path negates into a fresh buffer so the graph activation is not mutated.
Full-precision results are bitwise identical; tests compare fp32, bf16, and
int8 storage paths against the allocating formula.

On a representative FFN activation (`512x128`, fp32 activation storage,
median of 100 calls):

| version | time |
|---|---:|
| allocating | 3.52 ms |
| in-place | 2.95 ms |

That is a 1.19x improvement, or roughly 8 ms per sequence-512 forward across
15 FFN blocks. The profile below still lists SiLU first because the remaining
cost is the `exp` and elementwise work itself, not allocation.

## Why the STE weight gradient still uses OpenBLAS

A temporary AVX1 outer-product kernel was implemented and measured against
`x.T @ dy` with preallocated outputs. It was correct (maximum relative error
about 6e-4 versus BLAS, from summation order) but lost on the training shape.
Batch 512, two native threads, median of repeated runs:

| dW shape | OpenBLAS | native prototype | result |
|---|---:|---:|---|
| 32x32 | 49.5 us | 58.6 us | BLAS 1.18x faster |
| 32x128 | 133.0 us | 234.6 us | BLAS 1.76x faster |
| 128x32 | 142.5 us | 214.7 us | BLAS 1.51x faster |

At batch 64 the prototype reached parity only for 128x32 (38.3 us versus
39.9 us); the saving would be about 0.02 ms per step, so a second code path is
not justified. Transposing the operands (`(dy.T @ x).T`) and contiguous-copy
variants were also slower. Concatenating the attention and FFN weight
gradients into one GEMM lost the benefit to the copy/concatenation overhead
(122.9 us fused versus 93.2 us separate for attention; 217.8 us versus
214.2 us for FFN in a noisy run).

The shipped path therefore remains `x.T @ dy`, letting OpenBLAS choose its
own threading and packing. This is a deliberate negative result, not an
unimplemented optimization.

## Full-model profile

A cProfile run of one ternary training step at sequence 512 (current code,
loaded host) recorded 281 ms of profiled time; cProfile inflates Python work,
so use the phase table for step accounting. Selected cumulative/tottime:

| component | time | share / note |
|---|---:|---|
| SiLU (15 calls) | 43 ms cumulative | largest activation cost |
| packed forward + dX (210 calls) | 48 ms cumulative | NCM matrix work |
| recurrent scans (forward + backward) | 40 ms cumulative | chunked closed-form scan |
| STE dW (105 calls) | 16 ms tottime | FP32 OpenBLAS floor |
| RMSNorm (31 calls) | 14 ms cumulative | about 5% |
| native optimizer | 13 ms cumulative | about 5% |
| sigmoid (15 calls) | 10 ms cumulative | attention gate |
| batched pack | about 2 ms | after this revision |

There is no MoE routing in `dense-tiny`, so routing is not represented in this
table. The NCM-specific work is the packed matmul pair; it is already the
optimized native path, and the backward half is close to BLAS. Normalization
is not dominant. The next largest targets after this revision are the
recurrent scan math and the remaining transcendentals in SiLU, followed by
batching optimizer launches if optimizer time grows with model size.

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

Add `--profile-phases` to wrap cache lookups and prefetch, and to record
optimizer, invalidation, gradient-extraction, and packing time separately:

```bash
PYTHONPATH=$PWD SMAULBRAIN_THREADS=2 OPENBLAS_NUM_THREADS=2 \
OMP_NUM_THREADS=2 taskset -c 0-3 \
python benchmarks/ternary_benchmark.py --steps 15 --warmup 3 --runs 3 \
    --seq-lens 64 512 --modes fp32 ternary --profile-phases \
    --json-out benchmarks/ternary_phase_results.json
```

The wrapper adds a small Python overhead and should not be used for the
headline end-to-end table. Raw medians, per-step samples, and the recorded
post-run load averages are in [`ternary_results.json`](ternary_results.json),
[`ternary_results_frozen.json`](ternary_results_frozen.json),
[`ternary_results_loaded.json`](ternary_results_loaded.json),
[`ternary_results_frozen_loaded.json`](ternary_results_frozen_loaded.json),
and [`ternary_phase_results.json`](ternary_phase_results.json).

## Caveats

- This is a CPU-only NumPy implementation. The native kernel uses AVX1 when
  available and falls back to SSE2/scalar code; it does not use AVX2, AVX-512,
  or hardware FMA. The host CPU reports `fma` in `/proc/cpuinfo` but faults on
  an actual FMA instruction, so the build deliberately uses
  `-ffp-contract=off` and no FMA.
- Absolute times move with background load and CPU frequency. The clean-load
  tables were recorded at load average 1.9-2.2; the current check and phase run
  were recorded at 2.8-4.9 while a Java workload shared the four logical CPUs.
  The ratio between configurations is more stable than the absolute numbers,
  but the native pool is penalized more than OpenBLAS under contention.
- Dense FP32 backward is a highly optimized OpenBLAS GEMM. Ternary dX is
  faster than the old packed kernel but still not universally faster than
  dense BLAS, so backward speedup depends on matrix shape and batch size.
- No quality or convergence claim is made here. Ternarizing weights changes
  the function being trained; this report measures only execution time.
- `backward_mode=fp32` and `backward_mode=ternary` currently share the same
  packed input-gradient and FP32 STE weight-gradient path. Only `fp8` selects
  a distinct backward implementation. The benchmark uses `ternary`.
