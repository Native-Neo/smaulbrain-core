# Reduced-Precision Activation and Gradient Storage: Results

This report measures the opt-in storage formats documented in the README. The
default stays `fp32/fp32`; reduced-precision storage changes only the lifetime
format of cached activations and completed parameter gradients. Architecture,
loss definitions, checkpoints, optimizer state, and master parameters are
unchanged.

## Summary

- **Activation cache (dense-tiny, sequence 512, one profiled step):**
  33.72 MiB in FP32, 20.59 MiB in BF16, and 14.03 MiB in INT8. That is a
  38.9% and 58.4% reduction respectively.
- **Dense parameter gradients:** 1,024,000 bytes in FP32 versus exactly
  512,000 bytes with `--gradient-storage bf16` (50% smaller).
- **Peak RSS (20 steps, sequence 512, median of three runs):**
  102.44 MiB FP32, 101.71 MiB with BF16 gradients only, 84.42 MiB BF16
  activations, 77.02 MiB INT8 activations, and 82.84 MiB BF16 activations plus
  BF16 gradients.
- **CPU time per step (20 steps, sequence 512, median of three runs):**
  0.2604 s FP32, 0.3073 s with BF16 gradients only, 0.2291 s BF16 activations,
  0.2189 s INT8 activations, and 0.2873 s BF16+BF16. Quantized activations are
  now 12.0% (BF16) and 15.9% (INT8) faster than the FP32 activation cache on
  this preset and sequence length.
- **Short sequences:** at sequence 64 the same modes are within noise of one
  another (medians 0.0771 s FP32, 0.0776 s BF16, 0.0764 s INT8), because codec
  work is comparable to the smaller cache. The memory reduction remains.
- **Quality:** on a 60-step held-out byte corpus, final eval loss was 4.2303
  (FP32), 4.2344 (BF16), 4.2276 (INT8), and 4.2359 (BF16+BF16). All long
  stability runs converged to within 0.013 nats of FP32.
- **Previous implementation:** the prior report measured 0.521 s/step BF16 and
  0.483 s/step INT8 on the same 20-step sequence-512 setup (FP32 was 0.260).
  The current codecs and reduced-mode nonlinear paths make those modes roughly
  2.2x faster than that earlier implementation.
- **Traversal fix:** an unrelated reference cycle in the reverse-mode
  traversal is removed. Current FP32 peak RSS was 106 MiB after 40 steps,
  versus 225 MiB for the pristine original checkout, with bit-identical
  losses.

## Environment and method

- Python 3.14.7, NumPy 2.4.6, 4 CPU cores, 7.4 GiB RAM. Background desktop
  applications were running; timings were collected with CPU affinity and
  repeated to reduce scheduler noise.
- BLAS was pinned to one thread and the benchmark process to CPU 0 for every
  measurement:

  ```bash
  taskset -c 0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 FLEXBLAS_NUM_THREADS=1 \
      python benchmarks/precision_benchmark.py ...
  ```

- The 20-step and 30-step tables use `--warmup 2` or `--warmup 3` and no
  evaluation. The sequence-512 table is the median of three full benchmark
  processes; the sequence-64 table is also the median of three processes.
- `benchmarks/precision_benchmark.py` builds the same model initialization,
  token stream, optimizer, learning rate, clipping, and evaluation data for
  every configuration. The default seed is `1234`.
- `max_rss_kb` is the kernel's exact process high-water mark
  (`ru_maxrss`). `peak_rss_kb` is a 2 ms sampler and is retained for
  comparison. Medians are reported where runs were repeated because wall
  time, shared-page residency, and the allocator vary under external load.
- `mean_step_cpu_s` uses `time.process_time()`, so it excludes time the
  process spent descheduled. It still includes all storage conversion work.
  `backward s/step` is the benchmark's wall-clock backward plus
  gradient-extraction phase, so it can reflect scheduling as well as CPU work.
- Evaluation is deliberately full precision: `Model.forward(..., train=False)`
  does not enter an activation-storage context. Reported quality differences
  therefore come from the trained parameters, not from quantizing the eval
  forward pass.

## Storage formats and correctness

### BF16

BF16 values are stored as raw IEEE bfloat16 bit patterns in `uint16`, not as
float32 arrays that were briefly cast. Encoding uses round-to-nearest-even on
the 16 discarded mantissa bits; decoding widens the bits with a shift and
view. Signed zero, infinities, and NaNs are preserved, including the high
payload bits of NaN.

The codec was checked against `ml_dtypes` 0.6.0 as a development-only oracle
(not a runtime dependency) on 200,000 random 32-bit patterns:

- all finite values, infinities, and signed zeros encode to identical bits;
- decoding is exactly `bits << 16`;
- 730 patterns differ, all NaNs. `ml_dtypes` canonicalizes NaNs to `0x7fc0`,
  while this implementation preserves the original high payload bits.

The finite fast path never changes those bits. Finite inputs above 128 KiB use
a `uint16` view with a rare-tie correction; narrower inputs use a `uint32`
rounding operation. Inputs containing NaN or infinity take the separate
special-preserving path.

### INT8

INT8 uses a dynamic symmetric per-tensor scale
`max(abs(x)) / 127` and `np.rint` rounding. For a scale computed from the
tensor itself, scaled values are guaranteed to be in `[-127, 127]`, so the
encoder does not make a second clipping pass. When a caller supplies an
explicit scale, values are clipped to `[-127, 127]`. Dequantization is
`code * scale`. All-zero tensors use a unit scale. All-subnormal tensors fall
back to the smallest positive float32 scale when `amax / 127` would underflow
to zero. Tensors containing NaN or infinity are rejected with `ValueError`
instead of being silently saturated, including when an explicit scale is
supplied.

### Reduced-mode nonlinear arithmetic

The FP32 reference path keeps its float64 transcendental math. When activation
compression is active, SiLU, GELU, sigmoid, and softplus compute their forward
and recomputed backward values in float32. This is below the storage
quantization error already introduced by BF16/INT8, avoids a large amount of
avoidable conversion work, and remains deterministic. Reduced modes also skip
the forward derivative array that would only be discarded; the derivative is
recomputed in backward, as before.

### Residency rules

Only differentiable, non-parameter `float32` tensors of at least 1024 bytes are
compressed. Values that affect routing, normalization statistics, recurrent
scan output/state history, softmax, losses, final logits, optimizer state, and
master parameters stay FP32. BF16 gradients are accumulated in FP32 and
compressed once when each leaf gradient is complete; clipping and the optimizer
decode one parameter at a time.

## Activation-cache bytes

Dense-tiny, sequence 512, one profiled training step. Bytes are the live
storage reachable from the loss graph before backward; parameter tensors are
excluded.

| activation storage | cache bytes | cache MiB | vs FP32 | stored tensors | FP32 tensors |
|---|---:|---:|---:|---:|---:|
| `fp32` | 35,359,360 | 33.72 | 0.0% | 521 | 521 |
| `bf16` | 21,590,976 | 20.59 | -38.9% | 241 BF16 | 280 |
| `int8` | 14,707,748 | 14.03 | -58.4% | 241 INT8 | 280 |

The 521-tensor total is the same in every mode; the difference is only the
physical representation. `-38.9%` is slightly below the ideal 50% because 280
small or precision-critical tensors stay FP32. Codec changes do not alter these
modeled bytes.

## Gradient bytes

Dense-tiny has 1,024,000 parameter values, all of which receive gradients on
every step. The completed gradient set is therefore exactly:

| gradient storage | gradient bytes | vs FP32 |
|---|---:|---:|
| `fp32` | 1,024,000 | 0.0% |
| `bf16` | 512,000 | -50.0% |

For MoE models the total varies slightly between configurations because
activation quantization can change expert assignment and therefore which
expert parameters receive a gradient in a given batch. In the 150-step
moe-tiny runs the totals were 1,569,792 to 1,610,752 bytes in FP32 and 743,936
bytes with BF16 gradients.

## Peak RSS and throughput

### Dense-tiny, sequence 512

20 steps, median of three runs. `max RSS` is exact high-water RSS; `end RSS`
is the resident set after the last step. `last grad norm` is the global L2 norm
before clipping (threshold 1.0).

| activation / gradient | max RSS | end RSS | vs FP32 max | CPU s/step | wall s/step | backward s/step | last loss | last grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32 / fp32` | 102.44 MiB | 99.15 MiB | 0.0% | 0.2604 | 0.2680 | 0.2547 | 6.07695 | 3.6734 |
| `fp32 / bf16` | 101.71 MiB | 100.00 MiB | -0.7% | 0.3073 | 0.3147 | 0.2874 | 6.07698 | 3.6735 |
| `bf16 / fp32` | 84.42 MiB | 82.87 MiB | -17.6% | 0.2291 | 0.2339 | 0.2207 | 6.07086 | 3.6820 |
| `int8 / fp32` | 77.02 MiB | 74.80 MiB | -24.8% | 0.2189 | 0.2241 | 0.2111 | 6.05299 | 3.4402 |
| `bf16 / bf16` | 82.84 MiB | 80.78 MiB | -19.1% | 0.2873 | 0.2970 | 0.2689 | 6.07482 | 3.6806 |

### Dense-tiny, sequence 64

30 steps, median of three runs, no evaluation. Cache bytes are the profiled
graph size for this sequence length.

| activation storage | max RSS | cache bytes | CPU s/step | wall s/step |
|---|---:|---:|---:|---:|
| `fp32` | 66.15 MiB | 4,393,600 | 0.0771 | 0.0827 |
| `bf16` | 64.29 MiB | 2,696,128 | 0.0776 | 0.0808 |
| `int8` | 63.30 MiB | 1,848,356 | 0.0764 | 0.0868 |

At this size the conversion work and the smaller allocation footprint roughly
cancel. The storage feature is a memory optimization; it is not guaranteed to
be faster for short sequences.

Observations:

- With the optimized codecs, the sequence-512 dense path is faster with
  quantized activations than with FP32. The INT8 cache has the lowest RSS and
  the lowest CPU step time; BF16 is close.
- BF16 gradient storage still adds optimizer/clipping conversion work here
  (about 0.04–0.06 s/step), while saving only about 1 MiB of peak RSS because
  the activation cache dominates. Use it when the completed gradient set is
  large relative to activations, not as a general speed setting.
- Sequence 64 is a different trade-off: all three activation modes are within
  measurement noise, and the cache savings are smaller in absolute bytes.
- The earlier implementation's BF16/INT8 activation modes were about 1.9–2.0x
  slower than FP32 on sequence 512. The current result reverses that: faster
  BF16/INT8 codecs, no redundant computed-scale clipping, in-place INT8
  dequantization, finite-value BF16 fast paths, and float32 reduced-mode
  nonlinearities cut the conversion and recomputation overhead.

## Training quality

### Held-out byte corpus, 60 steps

The corpus is a deterministic concatenation of the repository's README,
`PORT_NOTES.md`, package sources, and tests, split 80/20 into train/eval
streams. It was rebuilt for this run (373,070 bytes). Sequence length 512,
60 steps, learning rate `3e-4`, evaluation on a fixed 512-byte held-out window.

| activation / gradient | first loss | last train loss | final eval loss | last grad norm | max RSS | CPU s/step |
|---|---:|---:|---:|---:|---:|---:|
| `fp32 / fp32` | 6.06243 | 4.41416 | 4.23030 | 1.9520 | 106.19 MiB | 0.2622 |
| `bf16 / fp32` | 6.06346 | 4.41965 | 4.23437 | 1.8525 | 88.25 MiB | 0.2348 |
| `int8 / fp32` | 6.06344 | 4.42946 | 4.22758 | 1.7915 | 80.90 MiB | 0.2191 |
| `bf16 / bf16` | 6.06346 | 4.42257 | 4.23589 | 1.8395 | 86.75 MiB | 0.2626 |

The eval-loss spread is 0.0083 nats. This is consistent with the storage
quantization being small relative to optimization noise and with evaluation
remaining FP32.

### Long stability runs

Dense-tiny, sequence 256, 300 steps, learning rate `1e-3`; moe-tiny, sequence
256, 150 steps, learning rate `1e-3`. Both use the same fixed token stream for
all modes and a fixed 256-token held-out eval window.

| preset | activation / gradient | first loss | last train loss | final eval loss | last grad norm | max RSS | gradient bytes | CPU s/step |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dense-tiny | `fp32 / fp32` | 6.00272 | 5.56845 | 5.56308 | 0.3295 | 82.45 MiB | 1,024,000 | 0.1426 |
| dense-tiny | `bf16 / fp32` | 6.00305 | 5.56601 | 5.56511 | 0.3320 | 74.38 MiB | 1,024,000 | 0.1370 |
| dense-tiny | `int8 / fp32` | 5.97934 | 5.56725 | 5.56288 | 0.3316 | 70.71 MiB | 1,024,000 | 0.1376 |
| dense-tiny | `bf16 / bf16` | 6.00305 | 5.56520 | 5.56356 | 0.3319 | 73.17 MiB | 512,000 | 0.1630 |
| moe-tiny | `fp32 / fp32` | 6.17217 | 5.56672 | 5.54224 | 0.5189 | 82.62 MiB | 1,610,752 | 0.1299 |
| moe-tiny | `bf16 / fp32` | 6.16761 | 5.57047 | 5.54822 | 0.5161 | 75.57 MiB | 1,569,792 | 0.1964 |
| moe-tiny | `int8 / fp32` | 6.24254 | 5.56890 | 5.54919 | 0.5072 | 72.31 MiB | 1,569,792 | 0.1947 |
| moe-tiny | `bf16 / bf16` | 6.16761 | 5.56704 | 5.55476 | 0.5225 | 73.79 MiB | 743,936 | 0.2085 |

All modes converge. Dense eval losses span 0.0022 nats and MoE eval losses
span 0.0125 nats. No mode produced non-finite losses or gradients. These
sequence-256 CPU times are noisier and less favorable to quantized storage than
the sequence-512 dense measurements; use sequence length and preset shape when
choosing a mode.

## Conversion overhead

Isolated codec benchmark on one million `float32` values. Each operation runs
30 times per measured group; seven groups; `time.process_time()` is divided by
the group size. This removes per-call timer noise and excludes descheduling.
Times are milliseconds.

| operation | median | best |
|---|---:|---:|
| BF16 encode | 4.75 ms | 4.48 ms |
| BF16 decode | 1.45 ms | 1.39 ms |
| BF16 encode+decode | 6.06 ms | 5.95 ms |
| INT8 quantize | 4.25 ms | 4.03 ms |
| INT8 dequantize | 1.65 ms | 1.39 ms |
| INT8 quantize+dequantize | 6.26 ms | 5.81 ms |

Compared with the previous codec implementation in the same environment, the
finite BF16 encoder is about 1.9x faster, INT8 quantization about 1.4x faster,
and INT8 dequantization about 1.3x faster. The step-level result is larger than
the codec result alone because reduced modes also avoid discarded derivative
work and use float32 transcendental math.

The training cost is not just the codec: compressed activations are decoded
whenever a forward expression or backward closure needs them, so a payload can
be decoded more than once per step.

## Reverse-traversal reference cycle

The original `Tensor.backward` built its topological order with a recursive
nested `visit` function. Because `visit` calls itself, its closure contained a
cell referring back to the function: a cycle that retained the completed
`topo` list, and through it every activation tensor in the step, until a
generational GC pass. Peak RSS consequently grew with step count.

The traversal is now an explicit stack. Order is preserved, so FP32 math is
unchanged. A/B against the pristine checkout:

| run | current max RSS | original max RSS | current CPU s/step | original CPU s/step | last loss |
|---|---:|---:|---:|---:|---:|
| dense-tiny, 20 steps | 105.29 MiB | 152.55 MiB | 0.270 | 0.282 | identical |
| dense-tiny, 40 steps | 105.97 MiB | 225.00 MiB | 0.283 | 0.263 | 5.99477 (both) |

The 40-step original figure is 2.1x the current one because the original
retains multiple step graphs until GC. The current FP32 path is stable at
roughly 106 MiB from 20 to 40 steps. This change is independent of the storage
formats and benefits the default path as well.

## Reproduction

Build a corpus from repository text if desired:

```python
from pathlib import Path

root = Path(".")
parts = []
for pattern in ("README.md", "PORT_NOTES.md", "smaulbrain/*.py", "tests/*.py"):
    for path in sorted(root.glob(pattern)):
        parts.append(path.read_bytes())
Path("/tmp/precision_corpus.bin").write_bytes(b"".join(parts))
```

Then run the controlled modes:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       FLEXBLAS_NUM_THREADS=1

for mode in "fp32 fp32" "fp32 bf16" "bf16 fp32" "int8 fp32" "bf16 bf16"; do
  set -- $mode
  python benchmarks/precision_benchmark.py \
    --preset dense-tiny --seq-len 512 --steps 60 --warmup 1 \
    --eval-every 10 --eval-tokens 512 \
    --corpus /tmp/precision_corpus.bin \
    --activation-storage "$1" --gradient-storage "$2" \
    --json-out "/tmp/precision_${1}_${2}.json"
done
```

The benchmark also runs against a pristine older checkout that has no
`smaulbrain.precision` module: it falls back to `fp32` and omits the
profiled-cache fields. Use `PYTHONPATH=/path/to/pristine/checkout` so that the
older package wins over an editable install.

## Caveats

- Only the tiny presets were measured end to end; full presets were not run
  because of RAM and time limits. Cache savings scale with sequence length and
  model width, so the relative memory benefit should grow. Conversion cost and
  the benefit of reduced-mode nonlinear math also depend on shape; sequence 64
  is neutral here, while sequence 512 is faster.
- Peak RSS includes faulted shared-library pages, allocator retention, and
  page-cache effects. Exact `ru_maxrss` is reproducible within a few percent
  on this host but is noisier than the modeled cache bytes. Medians are used
  for repeated runs.
- INT8 uses one symmetric scale per tensor. A tensor whose values span many
  orders of magnitude will lose small values; this is a storage trade-off, not
  a claim that INT8 is lossless.
- BF16 gradients are compressed after FP32 accumulation, so accumulation
  itself keeps FP32 semantics. The optimizer decodes one gradient at a time;
  optimizer moments and master weights stay FP32.
- There is no reconstruction or weight-quantization loss term in this
  repository. Training still optimizes cross-entropy plus optional MTP
  cross-entropy.
- Evaluation and generation do not use activation storage; the feature
  targets training-cache memory only.
- CPU timings were collected under external system load. Use the repeated
  medians and `mean_step_cpu_s`, not a single wall-clock run.
