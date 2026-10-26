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

> **Revision note.** The end-to-end tables below are a fresh low-load rerun
> of the current checkout after the concurrent Java workload stopped. The
> earlier pre-change baseline remains in git history; runs taken while Java
> was active are retained as `*_loaded.json` but are excluded from the
> comparison because contention distorts the native/OpenBLAS balance. The
> component A/B sections use the same current code unless stated otherwise.

## Summary

Latest low-load rerun of the current checkout, after the concurrent Java
workload stopped (15 steps, three warm-ups, three mode-rotated runs, 1-minute
load average 1.26-1.73; a browser still shared the CPU):

- **Forward:** 142.8 ms FP32 vs 83.3 ms ternary. Ternary forward is **1.72x
  faster**.
- **Backward:** 73.6 ms FP32 vs 84.4 ms ternary. The packed input-gradient
  kernel is offset by the unchanged FP32 straight-through weight GEMM, so
  backward is **0.87x**.
- **Full optimizer step:** 229.0 ms FP32 vs 181.8 ms ternary. Including the
  optimizer, gradient extraction, and invalidation/re-packing, the step is
  **1.26x faster**.

At sequence 64, forward is 1.25x faster, backward is 0.93x, and the full step
is 1.03x faster. The fixed step overhead and optimizer dominate the shorter
sequence.

With weights frozen after warm-up (pure forward/backward arithmetic and a warm
packed cache), sequence 512 measures 1.71x faster forward, 0.90x backward, and
1.32x for the complete step. Sequence 64 measures 1.27x, 0.85x, and 1.04x.
These frozen numbers use the same host state as the training table.

**Interpretation:** ternary is faster for complete training steps on the
measured dense workload, driven by a substantially faster forward. The
remaining backward cost is the STE weight gradient, which is a normal FP32
GEMM in both modes; packed storage cannot accelerate it. Training-step absolute
times are roughly 10-25% slower than the earlier clean-load baseline in git
history even though the load average is lower, so compare configurations
within one host state rather than across sessions.

## Measured model timings

`benchmarks/ternary_benchmark.py`, `dense-tiny`, 15 steps, 3 warm-ups, median
of 3 runs with mode order rotated between runs. Training mode runs the
optimizer and invalidates the packed cache before each next step; frozen mode
skips both.

### Training step (includes optimizer and re-packing)

| sequence | mode | forward | backward | step | forward speedup | backward speedup | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | FP32 | 27.1 ms | 22.6 ms | 60.1 ms | 1.00x | 1.00x | 1.00x |
| 64 | ternary | 21.8 ms | 24.4 ms | 58.1 ms | 1.25x | 0.93x | 1.03x |
| 512 | FP32 | 142.8 ms | 73.6 ms | 229.0 ms | 1.00x | 1.00x | 1.00x |
| 512 | ternary | 83.3 ms | 84.4 ms | 181.8 ms | 1.72x | 0.87x | 1.26x |

### Frozen weights (pure forward/backward, warm cache)

| sequence | mode | forward | backward | step | forward speedup | backward speedup | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | FP32 | 27.4 ms | 22.5 ms | 52.0 ms | 1.00x | 1.00x | 1.00x |
| 64 | ternary | 21.5 ms | 26.6 ms | 49.9 ms | 1.27x | 0.85x | 1.04x |
| 512 | FP32 | 140.1 ms | 75.6 ms | 220.5 ms | 1.00x | 1.00x | 1.00x |
| 512 | ternary | 82.0 ms | 84.2 ms | 166.8 ms | 1.71x | 0.90x | 1.32x |

The `step` column also includes copying gradients out of the graph and, in
training mode, the optimizer and cache invalidation. The profiled
`forward`/`backward` columns are the model's own phase timers.

### Host-load sensitivity

Earlier runs with a concurrent Java workload are retained as
[`ternary_results_loaded.json`](ternary_results_loaded.json) and
[`ternary_results_frozen_loaded.json`](ternary_results_frozen_loaded.json).
They are not used for the summary or comparison tables because the native
worker pool is penalized more than OpenBLAS under contention. The current
clean runs are [`ternary_results.json`](ternary_results.json) and
[`ternary_results_frozen.json`](ternary_results_frozen.json).

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

Direct pack of the 105 dense-tiny matrices (CPU time, median of 30 calls):

| variant | median | minimum |
|---|---:|---:|
| per-matrix calls | 2.05 ms | 1.94 ms |
| batched native call | 0.94 ms | 0.88 ms |

The single-matrix path also no longer copies the freshly packed buffer, so
its measured cost is now about 2.0 ms for this set. The end-to-end effect is
smaller because forward work dominates. In a same-process A/B of cold forwards
after invalidation (12 alternating pairs, sequence 512), lazy packing took
82.75 ms median versus 79.24 ms with prefetch. The two paths are logically
identical; the difference is the packing and wrapper work.

Phase instrumentation from the clean current run:

| sequence | forward | of which pack | cache lookup + prefetch | optimizer | invalidation | step |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 22.58 ms | 1.28 ms | 1.55 ms | 9.48 ms | 0.13 ms | 59.37 ms |
| 512 | 83.09 ms | 1.23 ms | 1.49 ms | 8.58 ms | 0.14 ms | 174.93 ms |

These are clean runs under the same host state as the summary; the phase and
main runs are separate, so compare component shares rather than the step
totals. The profile wrapper adds a small overhead. Earlier instrumented runs
measured 3.35 ms of packing at sequence 64 and 3.9-4.4 ms at sequence 512;
compare the direct CPU A/B above rather than those older absolute values.

## In-place SiLU

`silu` previously allocated a float64 copy, the sigmoid, the product, the
output cast, and (in full-precision storage) the held derivative. It now runs
the same ufuncs in place on the float64 copy, in the same operand order, and
uses the original fp32 activation for the final multiply. The reduced-storage
path negates into a fresh buffer so the graph activation is not mutated.
Full-precision results are bitwise identical; tests compare fp32, bf16, and
int8 storage paths against the allocating formula.

On a representative FFN activation (`512x128`, fp32 activation storage,
median of 200 calls across two alternating rounds):

| version | time |
|---|---:|
| allocating | 2.23-2.42 ms |
| in-place | 2.06-2.08 ms |

That is a 1.07-1.18x improvement, or roughly 2-5 ms per sequence-512 forward
across 15 FFN blocks. The profile below still lists SiLU first because the
remaining cost is the `exp` and elementwise work itself, not allocation.

## Why the STE weight gradient still uses OpenBLAS

A temporary AVX1 outer-product kernel was implemented and measured against
`x.T @ dy` with preallocated outputs. It was correct (maximum relative error
about 6e-4 versus BLAS, from summation order) but lost on the training shape.
Batch 512, two native threads, best of repeated medians:

| dW shape | OpenBLAS | native prototype | result |
|---|---:|---:|---|
| 32x32 | 30.2 us | 35.7 us | BLAS 1.18x faster |
| 32x128 | 74.4 us | 132.0 us | BLAS 1.77x faster |
| 128x32 | 76.9 us | 129.3 us | BLAS 1.68x faster |

At batch 64 the native prototype was still slower on every shape (native vs
OpenBLAS: 12.7 vs 8.6 us, 22.4 vs 20.8 us, and 22.9 vs 20.9 us), so there was
no saving to recover. Transposing the operands (`(dy.T @ x).T`) and contiguous-copy
variants were also slower. Concatenating the attention and FFN weight
gradients into one GEMM lost the benefit to the copy/concatenation overhead
(122.9 us fused versus 93.2 us separate for attention; 217.8 us versus
214.2 us for FFN in a noisy run).

The shipped path therefore remains `x.T @ dy`, letting OpenBLAS choose its
own threading and packing. This is a deliberate negative result, not an
unimplemented optimization.

## Full-model profile

A cProfile run of one ternary training step at sequence 512 (current code,
low-load host, 1-minute load about 1.3) recorded 193 ms of profiled time;
cProfile inflates Python work, so use the phase table for step accounting.
Selected cumulative/tottime:

| component | time | share / note |
|---|---:|---|
| SiLU (15 calls) | 29 ms cumulative | largest activation cost |
| packed forward + dX (210 calls) | 27 ms cumulative | NCM matrix work |
| recurrent scans (forward + backward) | 32 ms cumulative | chunked closed-form scan |
| STE dW (105 calls) | 10 ms tottime | FP32 OpenBLAS floor |
| RMSNorm (31 calls) | 10 ms cumulative | about 5% |
| native optimizer | 10 ms cumulative | about 5% |
| sigmoid (15 calls) | 8 ms cumulative | attention gate |
| batched pack | 1-1.3 ms | after this revision |

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
- Absolute times move with background load and CPU frequency. The current
  tables were recorded at 1-minute load average 1.26-1.73 with a browser
  sharing the four logical CPUs; runs taken during a Java workload are retained
  separately.
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
