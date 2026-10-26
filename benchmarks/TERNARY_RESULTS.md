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

> **Revision note.** This revision adds a native sequential recurrent scan, a
> fused RMSNorm node, exact in-place SiLU/sigmoid, and batched native SmaulOPT
> updates. The end-to-end tables below are a fresh 15-step, three-warm-up,
> three-mode-rotated rerun of the current checkout at 1-minute load average
> 1.20-1.41; a browser still shared the four logical CPUs. Earlier low-load
> artifacts remain in git history and the contention runs remain
> `*_loaded.json`. Compare configurations within one host state rather than
> across sessions.

## Summary

Current checkout, `dense-tiny`, sequence 512:

- **Forward:** 120.9 ms FP32 vs 59.9 ms ternary. Ternary forward is **2.02x
  faster**.
- **Backward:** 46.4 ms FP32 vs 52.0 ms ternary. The packed input-gradient
  kernel is offset by the unchanged FP32 straight-through weight GEMM, so
  backward is **0.89x**.
- **Full optimizer step:** 175.3 ms FP32 vs 117.9 ms ternary. Including the
  optimizer, gradient extraction, and invalidation/re-packing, the step is
  **1.49x faster**.
- **Optimizer alone:** 4.74 ms FP32 vs 4.65 ms ternary, down from about
  8.5-8.8 ms in the previous committed artifact.
- **Frozen weights, sequence 512:** 2.09x faster forward, 0.89x backward, and
  1.53x for the complete step.

At sequence 64, training forward is 1.32x faster, backward is 0.92x, and the
full step is 1.08x faster. Frozen weights measure 1.40x, 0.96x, and 1.16x.
Fixed step overhead and the FP32 weight-gradient floor dominate the shorter
sequence.

### Compared with the previous committed artifact

The previous committed low-load artifact measured the same preset and command
but in a different session. It is indicative, not a controlled A/B: load
averages and browser activity differed.

| sequence 512 | previous | current | change |
|---|---:|---:|---:|
| FP32 forward | 142.8 ms | 120.9 ms | 1.18x lower |
| ternary forward | 83.3 ms | 59.9 ms | 1.39x lower |
| FP32 backward | 73.6 ms | 46.4 ms | 1.59x lower |
| ternary backward | 84.4 ms | 52.0 ms | 1.63x lower |
| FP32 full step | 229.0 ms | 175.3 ms | 1.31x lower |
| ternary full step | 181.8 ms | 117.9 ms | 1.54x lower |
| step speedup | 1.26x | 1.49x | +0.23x |

Frozen sequence 512 moved from 82.0/84.2/166.8 ms (ternary forward/backward/
step) to 53.4/46.2/100.9 ms, and from 140.1/75.6/220.5 ms to
111.6/41.0/154.4 ms for FP32.

**Interpretation:** ternary remains faster for complete training steps on the
measured dense workload, driven by a substantially faster forward. The
remaining backward cost is the STE weight gradient, which is a normal FP32
GEMM in both modes; packed storage cannot accelerate it. The common scan,
RMSNorm, nonlinear, and optimizer improvements also speed up the FP32
baseline, so compare the ternary/FP32 ratios from this table rather than the
absolute reduction alone.

## Measured model timings

`benchmarks/ternary_benchmark.py`, `dense-tiny`, 15 steps, 3 warm-ups, median
of 3 runs with mode order rotated between runs. Training mode runs the
optimizer and invalidates the packed cache before each next step; frozen mode
skips both.

### Training step (includes optimizer and re-packing)

| sequence | mode | forward | backward | optimizer | step | forward speedup | backward speedup | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | FP32 | 22.3 ms | 16.7 ms | 4.77 ms | 45.5 ms | 1.00x | 1.00x | 1.00x |
| 64 | ternary | 16.9 ms | 18.1 ms | 4.71 ms | 42.1 ms | 1.32x | 0.92x | 1.08x |
| 512 | FP32 | 120.9 ms | 46.4 ms | 4.74 ms | 175.3 ms | 1.00x | 1.00x | 1.00x |
| 512 | ternary | 59.9 ms | 52.0 ms | 4.65 ms | 117.9 ms | 2.02x | 0.89x | 1.49x |

### Frozen weights (pure forward/backward, warm cache)

| sequence | mode | forward | backward | step | forward speedup | backward speedup | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | FP32 | 19.8 ms | 15.4 ms | 36.4 ms | 1.00x | 1.00x | 1.00x |
| 64 | ternary | 14.1 ms | 16.1 ms | 31.4 ms | 1.40x | 0.96x | 1.16x |
| 512 | FP32 | 111.6 ms | 41.0 ms | 154.4 ms | 1.00x | 1.00x | 1.00x |
| 512 | ternary | 53.4 ms | 46.2 ms | 100.9 ms | 2.09x | 0.89x | 1.53x |

The `step` column also includes copying gradients out of the graph and, in
training mode, the optimizer and cache invalidation. The profiled
`forward`/`backward` columns are the model's own phase timers.

### Host-load sensitivity

Earlier runs with a concurrent Java workload are retained as
[`ternary_results_loaded.json`](ternary_results_loaded.json) and
[`ternary_results_frozen_loaded.json`](ternary_results_frozen_loaded.json).
They are not used for the summary or comparison tables because the native
worker pool is penalized more than OpenBLAS under contention. The current
runs are [`ternary_results.json`](ternary_results.json),
[`ternary_results_frozen.json`](ternary_results_frozen.json), and
[`ternary_phase_results.json`](ternary_phase_results.json).

## Native recurrent scan

`npgrad.scan` now uses the native sequential recurrence for the common
training case: 2-D contiguous float32 values and decay, with no explicit
initial state. The chunked closed form remains the reference for batched/3-D
values, explicit `s0` (whose `ds0` term the native kernel does not return),
other dtypes/layouts, and systems without a compiler.

The backend is loaded lazily from `ng.scan` itself. This makes scan behavior
independent of whether a ternary model happened to run first, and gives the
FP32 baseline the same scan arithmetic in the benchmark. `Dust` uses the same
native 2-D path for its clean forward so that its parity check stays on the
same scan path as `Model.forward`.

Microbenchmark, `T=511`, `D=32`, full autograd scan forward plus backward,
including Tensor construction (median of 40 calls, two alternating rounds):

| implementation | median | minimum |
|---|---:|---:|
| chunked closed form | 2.28-2.30 ms | 2.23 ms |
| native sequential | 0.154-0.155 ms | 0.147 ms |

That is roughly a **14.8x** improvement for the scan itself. The sequential
and chunked forms differ in the last few float32 bits; the integration test
allows `2e-6` absolute/relative and finite-difference parity tests exercise the
backward scan. `test_native_scan_fast_path_matches_closed_form` also forces the
portable path and compares outputs and both gradients.

## Fused RMSNorm

`Model._rmsnorm` now calls a single `npgrad.rmsnorm` node. Its forward and
backward deliberately repeat the operation order of the previous
`square -> mean -> add -> rsqrt -> multiply` graph, so outputs and both input
gradients are bitwise identical. The node keeps the float32 statistic
arithmetic and leaves its result uncompressed, matching the old explicit
`activation_storage("fp32")` region, but does not retain the intermediate tape
tensors.

Microbenchmark on `512x32`, full forward plus backward, median of 500 calls:

| implementation | median |
|---|---:|
| explicit autograd graph | 0.89 ms |
| fused node | 0.39 ms |

That is about **2.3x** for the operation and removes 31 sets of intermediate
nodes per sequence-512 step. `test_fused_rmsnorm_matches_explicit_graph`
covers 2-D and 3-D inputs with `assert_array_equal` on output, `dX`, and
`dgain`.

## Exact in-place SiLU and sigmoid

`silu` runs the same float64 negation, `exp`, add, and reciprocal ufuncs in
place, then writes the fp32 output once through an explicit float64 ufunc.
`sigmoid` does the same for the `0.5*(1+tanh(0.5x))` formula. Full-precision
results and held derivatives are bitwise identical to the allocating form;
reduced-storage paths use fresh buffers and do not mutate the graph activation.

Fresh direct microbenchmarks, full-precision formula including the held
derivative where applicable, median of 200 calls across two alternating
rounds:

| case | allocating | in-place | change |
|---|---:|---:|---:|
| SiLU `512x128` | 2.09 ms | 1.88 ms | 1.11x |
| sigmoid `512x32` | 0.547 ms | 0.501 ms | 1.09x |

At sequence 512 the model runs SiLU 15 times and the attention sigmoid 15
times, so the combined saving is roughly 3-4 ms per forward. The current
cProfile still lists SiLU first because the remaining `exp` and elementwise
work is the actual cost, not the removed temporaries.

## Batched native SmaulOPT update

`SmaulOPT.step` now collects all factored matrices and dispatches one
`smaulbrain_smaulopt_factored_many` C call instead of one ctypes crossing per
matrix. The C wrapper calls the existing fused update sequentially for each
matrix, so the per-element operation order and results are unchanged; it only
removes Python argument conversion and FFI overhead. If any entry is ineligible, validation
returns `None` before mutation and the old per-matrix/NumPy fallback runs. The
batching test compares batched and forced-per-matrix updates bitwise,
including optimizer state and reported norms.

Dense-tiny matrix set (105 factored matrices), optimizer-only step with
`collect_norms=False`, median of 300 steps:

| implementation | median | minimum |
|---|---:|---:|
| per-matrix calls | 8.48 ms | 7.83 ms |
| batched call | 6.14 ms | 5.17 ms |

That is about **1.38x** for the optimizer phase. The end-to-end phase timer
shows the same effect without the microbenchmark setup.

## Packed-kernel changes

The earlier packed-kernel work remains the basis of the forward gain. In
summary:

- Work is partitioned over both output tiles and batch blocks, so small
  matrices use more than one worker.
- The low-work threshold for the native pool is 65,536 MACs for tiled
  matmuls, so small attention matrices at short sequences also use workers.
- Backward reads `dY` directly when `dout` is a multiple of the 16-column
  tile width, avoiding a padded copy.
- The first input-gradient tile stores directly into `dx`; later tiles
  accumulate, removing a zero-fill pass and one read of `dx`.
- Hot tiled/packing entry points pass raw addresses to avoid per-call
  `ctypes` pointer objects.
- Quantized linear computes the FP32 STE weight gradient before launching the
  packed input-gradient kernel, so native workers do not spin against the
  following OpenBLAS GEMM.

This revision also raises the forward batch block from 64 to 512 and uses
64-row input-gradient blocks only for 32x32 matrices, 32 rows otherwise.
Isolated kernel probes showed a low-single-digit forward improvement and mixed
changes in `dX`; the full-model runs above show no regression. Those constants
are scheduling only and do not change packed layout or arithmetic.

### Earlier kernel-only A/B

Direct C calls with a warm packed cache and preallocated output buffers.
Batch 512 is the model's training sequence. Times are microseconds, median of
four repeated batches of 1,200 calls; lower is better. "Old" is the parent
commit's kernel; "new" is the earlier tiled-kernel revision that this report's
code still contains.

| shape | fwd old | fwd new | fwd speedup | dX old | dX new | dX speedup |
|---|---:|---:|---:|---:|---:|---:|
| 32x32 | 52.0 | 32.5 | 1.60x | 67.1 | 37.4 | 1.79x |
| 32x128 | 116.0 | 103.7 | 1.12x | 313.7 | 113.7 | 2.76x |
| 128x32 | 86.2 | 85.0 | 1.01x | 178.7 | 130.8 | 1.37x |

At batch 64, the input-gradient speedups are 1.08x (32x32), 1.64x (32x128),
and 1.19x (128x32); forward speedups are 1.12x, 1.32x, and 1.45x.

## Batched cache refill

Training invalidates the packed representation of every weight touched by the
optimizer. `PackedCache.stale` records the invalidated references;
`Model.loss_and_backward` calls `prefetch()` before the forward graph is
built. `pack_ternary_tiled_many` packs all matrices into one uint8 allocation
in one native call and returns views; no second copy is needed. Entries for
dynamic MoE experts are excluded and packed lazily on demand.

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

The single-matrix path also no longer copies the freshly packed buffer. The
end-to-end effect is smaller because forward work dominates. In an earlier
same-process A/B of cold forwards after invalidation (12 alternating pairs,
sequence 512), lazy packing took 82.75 ms median versus 79.24 ms with prefetch;
the two paths are logically identical, and the difference is packing/wrapper
work.

Phase instrumentation from the current run (ternary mode):

| sequence | forward | of which pack | cache lookup + prefetch | optimizer | invalidation | step |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 17.37 ms | 1.20 ms | 1.37 ms | 4.77 ms | 0.13 ms | 43.40 ms |
| 512 | 63.30 ms | 1.17 ms | 1.46 ms | 4.78 ms | 0.12 ms | 127.05 ms |

These are profile-instrumented runs and add Python overhead relative to the
headline table; compare component shares rather than the step total. Pack and
cache lookup are stable at about 1.2 and 1.4-1.5 ms.

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
1-minute load about 1.05) recorded 132 ms of profiled time; cProfile inflates
Python work, so use the phase table for step accounting. Selected
cumulative/tottime:

| component | time | share / note |
|---|---:|---|
| SiLU (15 calls) | 24 ms cumulative | largest activation cost |
| packed forward + dX (210 calls) | 24 ms cumulative | 11 ms forward, 13 ms dX |
| recurrent scans (forward + backward) | 2 ms cumulative | native sequential path |
| STE dW (105 calls) | 9 ms tottime | FP32 OpenBLAS floor |
| sigmoid (15 calls) | 8 ms cumulative | attention gate |
| fused RMSNorm (31 calls) | 5 ms cumulative | bitwise-equivalent node |
| batched native optimizer | 7 ms cumulative | 4 ms native call dispatch |
| batched pack | 1-2 ms | after invalidation |
| cross-entropy | 2 ms | final logits |

There is no MoE routing in `dense-tiny`, so routing is not represented in this
table. The NCM-specific work is the packed matmul pair; the backward half is
close to a BLAS floor plus the separate FP32 dW. The next remaining targets
are the SiLU transcendentals and the packed-kernel Python bridge; the scan,
RMSNorm, and optimizer launches are no longer dominant.

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
  tables were recorded at 1-minute load average 1.20-1.41 with a browser
  sharing the four logical CPUs; runs taken during a Java workload are
  retained separately. The ratio between configurations is more stable than
  the absolute numbers, but mode order is still rotated to reduce drift.
- The benchmark uses `profile=True` to split forward/backward and therefore
  includes a small graph-storage statistics pass in the step wall time. The
  comparison is internally consistent; a production training loop without
  profiling does not pay that pass.
- The native scan's last-bit differences are covered by finite-difference and
  closed-form parity tests. It is not used when gradients with respect to an
  initial state are required; those inputs keep the closed-form implementation.
- The batched optimizer calls the same per-matrix kernel in sequence; it does
  not change update order or arithmetic.
