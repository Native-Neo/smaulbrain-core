# Dust Activation-Perturbation Training: Results

This report evaluates the experimental Dust trainer in `smaulbrain/dust.py`
against the default FP32 NumPy backpropagation path. Dust is an isolated,
opt-in alternative selected with `--trainer dust`; it is not the default and it
does not reproduce the scale of the original paper.

Paper: QLabs Research, *Dust: Pretraining Transformers Without Backpropagation*
(October 2026), <https://qlabs.sh/research/dust>. Official implementation:
<https://github.com/qlabs-eng/dust>.

## Summary

- **Dust learns without invoking backward.** On real repository byte text,
  dense-tiny, sequence length 64 and 40 optimization steps, Dust K=2 improved
  held-out loss from about 6.0 to a median best of 4.07 (learning rate 0.05) or
  4.10 (learning rate 0.02). The project-default backprop baseline (learning
  rate `3e-4`) reached 4.48. A tuned backprop baseline (learning rate `3e-3`)
  reached 3.51 and was both better and cheaper.
- **The 60-step follow-up narrows the gap, but this is still a
  tiny-population test.** In a single-seed 60-step run, Dust K=4 reached 3.081,
  marginally below tuned backprop's 3.088, while Dust K=2 reached 3.154
  (lr 0.05) and 3.214 (lr 0.02). This is not a quality win: it is one seed,
  and K=4 still cost about 9x the tuned-backprop training time. The paper's
  populations are 256 to 16,384 draws; these runs use 2-4 draws.
- **The speed refactor helped materially, but compute is still the dominant
  cost.** The perturbed passes now use float32 nonlinearities, independent
  groups share one batched forward, and the head runs in a deterministic
  background stream. The 40-step training times fell from 47-78 s to
  15-26 s, a 3.0-3.2x speedup over the earlier serial implementation. Dust
  nevertheless took 6.1-6.7x tuned-backprop training time at K=2 and 10.5x at
  K=4 in the 40-step comparison; the 60-step ratios were 5.9-6.3x and 9.2x.
  A Dust update still evaluates many perturbed block passes, while backprop
  evaluates one forward and one reverse pass.
- **The clean cache is smaller; peak memory is still higher.** At sequence
  length 64, Dust's clean cache was 1.49 MiB versus backprop's 4.26 MiB
  profiled activation cache, a 65% reduction. The 40-step runs (which include
  an 8,192-token evaluation) peaked at 130.2 versus 121.8 MiB. Dedicated
  5-step memory points with an evaluation window equal to the training
  sequence length show Dust at 76.7 versus 69.8 MiB (sequence 64), 102.5
  versus 85.2 (256), and 118.6 versus 103.8 (512). The concurrent head pass
  and allocator retention add several MiB at short sequences. Adaptive head
  batching reduced the 512-token peak from 128.1 MiB in the earlier serial
  version to 118.6 MiB, but the paper's memory advantage still does not
  materialize in this NumPy implementation.
- **Verdict:** the implementation is a genuine forward-only training path with
  deterministic updates, exact checkpoint resume, and verified estimator math.
  The optimized path is about 3.1x faster than the initial serial version at
  sequence 64, but at the CPU scale tested here it is still not a practical
  replacement for backprop: tuned backprop is better and roughly 6-10x cheaper
  per step, and Dust's smaller clean cache does not lower peak RSS. The
  experiment does not test the paper's large-population regime.

## What was implemented

The estimator follows the paper's activation-space perturbation rule:

1. Run one clean forward and cache the inputs/outputs needed for the local
   outer products and per-token losses.
2. For direct-loss sites, add independent Gaussian noise to a linear layer's
   output at every token, rerun from that block onward, and reward each
   token's noise by the centered loss reduction. The reward-weighted noise
   estimates the output error; the outer product with the cached input is the
   weight gradient.
3. For attention internals, recompute only the block's attention output from
   cached clean activations. Score each perturbation by its alignment with the
   estimated attention-output error. Queries/gates keep their own token's
   score; keys and values receive future-token credit with decay `gamma`.
4. Jitter the language-model head on cached logits, one vocabulary slab at a
   time, and scatter/add the tied embedding and head gradients.
5. Apply SGD with momentum.

SmaulBRAIN-specific mapping and deviations:

| Paper | This implementation |
|---|---|
| Dense GPT blocks with separate `c_proj` writers | Dense SmaulBRAIN blocks; attention and FFN output projections are jittered together, matching the paper's two writer sites |
| One `mlp.c_fc` | Two FFN input projections (`Wa`, `Wb`) jittered together |
| Attention-output hubs `o@layer` | `attn.hub` before `Wo`; shallow/deep sigmas are separate |
| Attention internals q/k/v/gate/value-embedding | `r`/`k`/`v`; `r` uses own-token credit, `k`/`v` use future credit |
| Untied `lm_head`, `15*tanh` logit cap | Tied embedding/head, no cap; embedding and head gradients are summed |
| Head reward centered over batch | Centered over draws (SmaulBRAIN has no batch axis) |
| `2L` residual mixing scalars via weight-space ES | Non-matrix parameters are frozen by default; optional ES over all of them is experimental |
| `gamma = 0.98`, sigma 0.2/0.4 | Default `gamma = 0.5` and sigma 0.05 (see tuning below) |
| Populations 256-16k, head `4*population` | K=2-4 block draws, 1024 head draws split equally over all 256-column slabs each step |

The independent forward in `DustForward` is parity-tested against
`Model.forward`. A from-block rerun without jitter is exactly the clean
forward. Dust never calls `Model.loss_and_backward` or any autograd backward
closure; a test monkeypatches the autograd backward method and verifies it is
not invoked.

## Correctness and integration tests

`tests/test_dust.py` covers:

- clean-forward and from-block rerun parity against `Model.forward`;
- estimator diagnostics against exact gradients: direct cosine about 0.95 at
  512 draws, local attention R about 0.97 and K/V about 0.88 with gamma
  0.5-0.7, and head slabs about 0.96 at 4096 draws; the tests use conservative
  thresholds below those diagnostics;
- uneven head-slab partitions;
- the no-backward invariant;
- deterministic gradients for a fixed seed/step and different gradients for a
  different step;
- bitwise equality between the batched direct-group forward and independent
  `_direct_group` calls, and between serial and concurrent head estimation;
- frozen non-matrix parameters, optional scalar ES restoration, and optimizer
  validation before mutation;
- held-out loss improvement on a small periodic pattern;
- checkpoint manifest/config metadata and cross-trainer mismatch rejection;
- bitwise-exact resume and Ctrl+C saving at a committed step;
- resume restoring the checkpoint's Dust learning rate and momentum rather
  than the launching command line;
- CLI/config validation, including negative perturbation seeds, unsupported
  MoE/MTP/quantized/storage configurations, and NaN/Inf rejection.

`python -m pytest -q` passes 82 tests (39 Dust, 43 existing) in about 7-8
seconds; `python -m pytest -q tests/test_dust.py` passes 39 in about 7 seconds.
The default suite excludes the stale `tests/_reference` Torch/legacy
expectations.

## Performance implementation

The estimator mathematics and checkpoint format are unchanged by the speed
work; the optimizations are algebraic/vectorization changes plus scheduling.

- **Float32 perturbed nonlinearities.** The clean forward remains the exact
  float64-intermediate reference used for parity with `Model.forward`.
  Perturbed forwards use float32 SiLU/sigmoid. They already carry stochastic
  reward noise, and the float32 error is far below that noise; the clean
  logits and clean loss used by the estimator are still the reference values.
- **Batched direct groups.** Hub, writer, and hidden groups at a layer are
  evaluated in one forward with a batch row per draw. Rows from other groups
  have zero jitter at the current group's sites, so each row is exactly the
  separate-forward computation. A focused test verifies bitwise equality with
  three independent `_direct_group` calls; batched rows also keep the original
  RNG draw order for the default draw counts.
- **Batched head.** Head logits are still max-shifted, jittered per slab, and
  reward-weighted in float32. The head row batch is capped at 32 rows and is
  reduced for long sequences so each transient head array stays near 4 MiB.
  This keeps the long-sequence cache/allocator footprint below the earlier
  serial implementation.
- **Concurrent head stream.** For the default head budget (at least 256 draws)
  on a multi-CPU machine, the head pass runs in one worker thread while the
  recurrent direct/local passes run on the main thread. A cloned generator is
  advanced through the current serial draw order, preserving the serial head
  stream position and draw budget. A test runs the same step with
  `cpu_count()` forced to 1 and 4, including a draw count larger than the
  direct chunk, and requires byte-identical gradients. On a single CPU, or
  with a small head budget, the head runs serially.
- **No fallback.** If a perturbed forward, head partition, or estimated
  gradient becomes non-finite, the existing `FloatingPointError` path still
  aborts the step; no optimization silently falls back to backprop.

The speedup is modest in absolute terms because the recurrent block loop is
still Python/NumPy and the head still evaluates a large Gaussian softmax
population. At sequence 64 the optimized step is about 0.39 s versus 1.21 s
before; at sequence 512 it is 3.35 s versus 10.05 s. The parallel head adds a
few MiB of peak RSS at short sequences, which is included in the reported
memory points.

## Benchmark methodology

All numbers were produced by `benchmarks/dust_benchmark.py`, which launches
each configuration in a fresh subprocess so `ru_maxrss` is a clean process
high-water mark. BLAS/OpenMP/MKL/FlexBLAS thread counts were pinned to 1
before NumPy import. The measurements were collected on Linux with Python
3.14.7 and NumPy 2.4.6; the implementation itself is Python 3.9-compatible and
has no new third-party dependency.

- **Model:** `dense-tiny`, 15 layers, `d_model=32`, FFN width 128, vocabulary
  259, 256,000 parameters at initialization. No MTP, no growth.
- **Data:** a 342,955-byte corpus built deterministically from `README.md`,
  `PORT_NOTES.md`, `smaulbrain/*.py`, and `tests/*.py`
  (SHA-256 `813cfbd2cc3eba7de0cd833528434f148e5c0d71c129c0592981fc2a3281593c`).
  The first 80% (274,364 bytes) is the training stream; an 8,192-byte window
  from the held-out 20% is the fixed evaluation set.
- **Sequence length:** 64 tokens (63 next-token predictions per step). This is
  far shorter than the paper's 2,048-token sequences and was chosen for CPU
  feasibility.
- **Main comparison:** 40 optimization steps, three initialization seeds
  (0, 1, 2), held-out evaluation every 10 steps.
- **Backprop:** FP32 SmaulOPT, global gradient-norm clip 1.0, learning rates
  `3e-4` (project default) and `3e-3` (tuned for this run).
- **Dust:** SGD with momentum 0.9, no weight decay, no gradient clipping,
  K=2 or 4 draws for each block group, 1,024 head draws, chunk 8, sigma 0.05,
  gamma 0.5, and deterministic per-step RNG from
  `SeedSequence([dust_seed, global_step])`.
- **Memory accounting:** backprop activation-cache and gradient bytes come
  from one profiled FP32 step; Dust clean-cache, estimator, and temporary
  bytes are maxima reported by the engine; optimizer bytes are the actual
  state arrays. Peak RSS includes the Python/NumPy process baseline and the
  8,192-token held-out evaluation forward, which is much longer than one
  training sequence; the dedicated memory-scaling section uses an evaluation
  window equal to the training sequence length to isolate training memory.
- **Timing:** `train_s` is the sum of the per-step training times and excludes
  evaluation. `s/step` is the mean training step. Evaluation time is reported
  separately because the 8,192-token held-out window is much larger than one
  training sequence and otherwise dominates a 40-step backprop run. The
  checked-in JSON for the current runs records `train_wall_s` directly; when
  an older child run lacks it, `train_s` is the sum of that run's per-step
  `history[*].wall_s`.

The corpus contains repository text and code. It is real byte-level text, but
it is small, self-similar, and not a substitute for FineWeb or another
pretraining corpus. Absolute losses are not comparable to the paper.

## Main results

Medians over seeds 0, 1, 2; brackets are the observed min/max. `best` is the
best held-out loss seen at any evaluation, including the final one.

| configuration | final eval | best eval | train s | s/step | eval s | peak RSS MiB | cache MiB | grad MiB | opt MiB | update norm | last grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| backprop, lr `3e-4` (default) | 4.482 [4.373, 4.496] | 4.482 [4.373, 4.496] | 2.63 | 0.066 | 9.47 | 121.75 | 4.26 | 0.977 | 1.027 | 2.49 | 4.33 |
| backprop, lr `3e-3` (tuned) | 3.561 [3.540, 3.880] | 3.506 [3.452, 3.564] | 2.44 | 0.061 | 9.48 | 121.75 | 4.26 | 0.977 | 1.027 | 17.96 | 1.37 |
| Dust K=2, lr 0.05 | 4.438 [4.163, 5.087] | 4.069 [3.873, 4.163] | 16.36 | 0.409 | 10.03 | 129.79 | 1.49 | 0.969 | 0.977 | 61.63 | 5.45 |
| Dust K=2, lr 0.02 | 4.114 [4.113, 4.362] | 4.099 [3.998, 4.233] | 14.78 | 0.369 | 9.28 | 130.19 | 1.49 | 0.969 | 0.977 | 32.44 | 5.17 |
| Dust K=4, lr 0.05 | 4.559 [4.037, 4.714] | 4.034 [3.927, 4.156] | 25.55 | 0.639 | 9.38 | 127.81 | 1.49 | 0.969 | 0.977 | 42.66 | 3.32 |

`last grad norm` is the global norm before clipping. Backprop is clipped to
1.0; Dust uses no clipping by default. Backprop's 688 autograd graph tensors
are not included in the cache byte count but contribute to peak RSS; Dust
reports zero graph tensors and 56 (K=2) or 112 (K=4) full-forward-equivalent
block evaluations per step.

Training-time ratios relative to tuned backprop:

| configuration | train-time ratio |
|---|---:|
| Dust K=2, lr 0.05 | 6.7x |
| Dust K=2, lr 0.02 | 6.1x |
| Dust K=4, lr 0.05 | 10.5x |

The profiled backprop path stores 4.26 MiB of activation tensors at sequence
64; Dust stores 1.49 MiB of clean activations, 0.109 MiB of estimator errors,
and up to 4.0 MiB of transient draw/head buffers. Dust also keeps a gradient
dict for the trained matrix parameters (0.969 MiB) and momentum (0.977 MiB).
The sum is smaller than backprop's tape plus cache, but the 8,192-token
evaluation plus allocator/transient overhead leaves Dust about 8 MiB higher at
the run peak. Adaptive head batching keeps the longer-sequence footprint in
check; see the dedicated memory points below.

### Held-out trajectories

Backprop (tuned) and Dust (K=2, lr 0.05) both reached their best evaluation
before the final step in some seeds; the table keeps both final and best so
early stopping cannot hide a regression. Dust's training loss was noisier:
the K=2/lr 0.05 seed 0 run improved to 4.069 at step 30 and then rose to 5.087
at step 40, while its best evaluation was 4.069. The K=2/lr 0.02 seed 0 run
peaked at 3.998 and ended at 4.113. This is consistent with the estimator's
noisy gradient and the large SGD step size; it is not evidence of a numerical
failure (all losses and gradients were finite).

### Precision references

These are 10-step references with seed 0 and tuned backprop lr `3e-3`, to
compare activation-storage formats under the same protocol. They are not
40-step quality results.

| configuration | final eval at step 10 | train s (10 steps) | s/step | peak RSS MiB | cache MiB |
|---|---:|---:|---:|---:|---:|
| FP32 activations / FP32 gradients | 4.491 | 0.53 | 0.053 | 121.67 | 4.26 |
| BF16 activations / FP32 gradients | 4.504 | 0.96 | 0.096 | 119.49 | 2.61 |
| INT8 activations / FP32 gradients | 4.461 | 1.01 | 0.101 | 118.79 | 1.79 |

At sequence 64 the storage conversion increases step time by about 1.8-1.9x,
while peak RSS changes by only a few MiB and is within run-to-run allocator
variance. The longer-sequence result in
`benchmarks/PRECISION_RESULTS.md` is the relevant memory reference for
BF16/INT8 storage. Dust always uses its own FP32 cache and rejects non-FP32
storage flags.

## 60-step single-seed follow-up

A longer run checks whether the 40-step ranking is a transient. This is one
seed (11), so it is not used for the main three-seed comparison.

| configuration | final eval | best eval | train s | s/step | peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| backprop, lr `3e-4` (default) | 4.098 | 4.098 | 3.47 | 0.058 | 121.65 |
| backprop, lr `3e-3` (tuned) | 3.088 | 3.088 | 4.23 | 0.071 | 121.64 |
| Dust K=2, lr 0.05 | 3.154 | 3.154 | 26.45 | 0.441 | 130.03 |
| Dust K=2, lr 0.02 | 3.214 | 3.214 | 24.96 | 0.416 | 129.95 |
| Dust K=4, lr 0.05 | 3.081 | 3.081 | 38.94 | 0.649 | 127.98 |

Training-time ratios relative to tuned backprop in this 60-step run are 6.3x
(K=2, lr 0.05), 5.9x (K=2, lr 0.02), and 9.2x (K=4, lr 0.05). Dust closes
most of the quality gap with more steps; K=4 is 0.007 nats below tuned
backprop on this single seed, but that is within seed noise and costs about
9x as much training time. This does not test the paper's claim that large
populations can match or exceed backprop; it tests only 2-4 draws.

## Natural-language check

The repository corpus is code and documentation. To check that the result is
not specific to that text, a second single-seed run used Tiny Shakespeare
(1,115,394 bytes; SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`; downloaded
from
<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>).
The model, sequence length, 40-step budget, held-out window (8,192 bytes), and
optimizer settings are otherwise unchanged; seed is 0.

| configuration | final eval | train s | s/step | peak RSS MiB |
|---|---:|---:|---:|---:|
| backprop, lr `3e-4` (default) | 4.512 | 2.27 | 0.057 | 123.59 |
| backprop, lr `3e-3` (tuned) | 3.142 | 2.58 | 0.065 | 123.61 |
| Dust K=2, lr 0.05 | 3.669 | 26.17 | 0.654 | 129.29 |
| Dust K=2, lr 0.02 | 3.688 | 16.49 | 0.412 | 132.19 |

All four runs started at 6.167 and improved their held-out loss monotonically.
Dust beat the project-default backprop learning rate by 0.82-0.84 nats but
lost to tuned backprop by 0.53-0.55 nats. Training time was about 6.4-10.1x
tuned backprop; the lr 0.05 run was slower than the lr 0.02 run despite
identical shapes, which reflects shared-CPU variance rather than an
algorithmic difference. The ranking is the same as on the repository corpus.
This is one seed and one 8 KB evaluation window; it is a sanity check, not a
natural-language benchmark.

## Memory scaling

Longer sequences make the activation cache a larger fraction of the footprint,
but they also enlarge Dust's draw batches and head slabs. These are 5-step
memory points (seed 0, Dust K=2 lr 0.02), not quality comparisons. The held-out
evaluation uses a window equal to the training sequence length, so the
8,192-token evaluation from the 40-step table does not dominate the peak.

| sequence length | config | train s | s/step | peak RSS MiB | start RSS MiB | params MiB | grad MiB | opt MiB | cache/clean MiB | estimator MiB | temp MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | backprop | 0.30 | 0.059 | 69.83 | 55.76 | 0.98 | 0.98 | 1.03 | 4.26 | - | - |
| 64 | Dust K=2 | 1.95 | 0.391 | 76.73 | 56.61 | 0.98 | 0.97 | 0.98 | 1.49 | 0.11 | 4.00 |
| 256 | backprop | 0.78 | 0.156 | 85.19 | 55.79 | 0.98 | 0.98 | 1.03 | 16.91 | - | - |
| 256 | Dust K=2 | 7.48 | 1.495 | 102.53 | 56.77 | 0.98 | 0.97 | 0.98 | 5.94 | 0.44 | 8.00 |
| 512 | backprop | 1.42 | 0.283 | 103.78 | 55.75 | 0.98 | 0.98 | 1.03 | 33.79 | - | - |
| 512 | Dust K=2 | 16.77 | 3.354 | 118.64 | 59.90 | 0.98 | 0.97 | 0.98 | 11.88 | 0.88 | 8.00 |

At all three lengths Dust's clean-cache reduction is offset by the concurrent
head pass, draw scratch, and allocator overhead. Dust peaks 6.9, 17.3, and
14.9 MiB *higher* than backprop at 64, 256, and 512, even though its clean
cache is 2.8, 11.0, and 21.9 MiB smaller. The earlier serial implementation
peaked at 128.1 MiB at 512; adaptive head batching brought that down to
118.6 MiB. The engine's declared `temp_bytes` (4.0, 8.0, and 8.0 MiB) counts
the largest simultaneous draw/head buffers, not allocator retention or all
transient arrays; the Python/NumPy
allocator keeps freed blocks, and the head's `(draw, token, slab)` arrays grow
with sequence length. `--dust-chunk` still controls the direct draw batch; the
head batch is capped internally by a transient-size budget and is not a user
knob.

The training-step ratio also worsens with sequence length in this harness:
0.391/0.059 = 6.6x at 64, 1.495/0.156 = 9.6x at 256, and 3.354/0.283 =
11.9x at 512. The optimized batching and concurrent head keep this far below
the earlier serial ratios (20.8x, 39.2x, and 46.5x), but the Python loops over
blocks and draws still dominate at longer T. The 40- and 60-step quality
results above are all sequence 64.

This is an implementation-level result, not a claim about a fused GPU Dust
kernel. The clean cache is genuinely smaller, but in this CPU NumPy harness the
end-to-end memory saving does not materialize. The paper's memory argument
should not be transferred to this implementation without a fused or
memory-reusing estimator.

## Tuning evidence

The paper's constants do not transfer unchanged to this architecture. The
following single-seed checks (seed 11, same corpus, 40 or 60 steps, K=2 unless
noted) motivated the defaults. They are exploratory, not a full grid. The
non-learning-rate ablations were collected before the speed refactor; the
refactor preserves the estimator and the serial RNG stream, apart from float32
perturbed nonlinearities and batched arithmetic, so they are retained as
tuning evidence rather than rerun as new benchmark rows.

- **Credit decay.** At 40 steps, end-of-run held-out loss was 3.777 with gamma
  0.5, 3.621 with 0.98, and 3.734 with 0.9. At 60 steps it was 3.233 with 0.5,
  3.167 with 0.9, and 3.270 with 0.98. The differences are small and
  seed-dependent; 0.9/0.98 did not consistently beat 0.5 across lengths. The
  local attention gradient check favored gamma around 0.5 (R cosine 0.97; K/V
  0.88), matching SmaulBRAIN's mean recurrence decay. The 0.5 default is an
  architecture-based choice, not a claim that it won every comparison. The
  paper's 0.98 remains available through `--dust-gamma`.
- **Noise scale.** Sigma 0.2 for all direct/local/head sites gave end-of-run
  loss 3.904 at 40 steps versus 3.777 for 0.05. A hub-only cosine check was
  0.72 at sigma 0.05 and 0.30 at 0.20 at sequence 64. The default is 0.05.
  The paper's 0.2/0.4 schedule is available with the separate sigma flags.
- **Draws and learning rate.** In the current three-seed 40-step table,
  K=2/lr 0.05 gave 4.069 best median; K=2/lr 0.02 gave 4.099; K=4/lr 0.05 gave
  4.034. At 60 steps (single seed 11), K=2/lr 0.05 gave 3.154, K=2/lr 0.02
  gave 3.214, and K=4/lr 0.05 gave 3.081. More draws improve the estimate but
  cost proportionally more time; the CPU default stays at K=2.
- **Scalar ES.** Two-sided ES with 2 draws over all non-matrix parameters
  destabilized the 40-step run (best 4.294 versus 3.777 with them frozen).
  The paper's ES covers only 2L residual scalars; SmaulBRAIN's non-matrix
  arrays do not map cleanly, so they remain frozen by default.
- **Gradient alignment.** On a fixed batch, the cosine between Dust and
  backprop parameter gradients was about 0.037 at K=2, 0.070 at K=8, and 0.20
  at K=32; the norm ratio fell from about 7.2 to 1.4. Alignment improves with
  population, but even K=32 is far below the paper's populations. Dust's
  gradients are intentionally not identical to backprop's; low alignment does
  not by itself imply failure, but it explains the noisy optimization.

## Limitations

- **Scale.** One tiny model, one 343 KB corpus, sequence length 64,
  40-60 steps, CPU only. The paper uses GPT-scale models, FineWeb,
  2,048-token sequences, and 100k-20M token budgets. No conclusion here
  extends to the paper's regime.
- **Population.** K=2-4 draws versus the paper's 256-16k. The head has 1,024
  draws, but the block groups do not. Large-population behavior is untested.
- **Baseline tuning.** Backprop was tuned over two learning rates; Dust over
  two learning rates and two draw counts. A larger grid could change the
  exact gap, but the 6-10x training-cost difference at sequence 64 is
  structural.
- **Uniform layer allocation.** Every block group gets K draws. The official
  code allocates more draws to early blocks; this allocation was not tuned.
- **No schedule or Adam.** Dust uses constant-learning-rate SGD with momentum,
  as in the paper's main comparison. The paper's Adam variant and any warmup
  or decay schedule were not reproduced.
- **No MoE, MTP, quantization, paging, or reduced-precision storage.** These
  are rejected rather than approximated. The paper's estimator does not
  define discrete top-k routing, MTP objectives, or quantized forward
  semantics for this codebase.
- **Non-matrix parameters are frozen.** This includes RMSNorm gains and the
  per-channel `attn.w`/`attn.u` gates. The optional ES fallback is broader
  than the paper's scalar ES and was not beneficial.
- **Architecture differences.** Tied embedding/head, no `15*tanh` logit cap,
  no value embeddings, a single recurrent linear-attention path, and no
  batch axis. Direct rewards use own-token credit; only K/V receive future
  credit.
- **Head adaptation.** SmaulBRAIN has no batch, so the head reward is centered
  over draws. All vocabulary slabs are visited each step with equal draws,
  rather than cycling slabs across chunks as in the official code.
- **Memory accounting.** Profiled cache bytes exclude Python object and tape
  closure overhead; peak RSS includes them. At sequence 64 the cache is too
  small to move peak RSS much; at 256/512 the transient allocator footprint
  dominates and Dust peaks higher, as reported above.
- **Timing.** Single-machine, shared-CPU measurements. The main comparison is
  a median of three seeds; the 60-step, natural-language, memory, and
  precision follow-ups are single runs. Model setup and Python import are
  excluded from `train_s` but included in peak RSS. Absolute times are not
  portable.
- **Optimizer.** Dust uses SGD with momentum, matching the paper's main
  comparison. The paper's Adam experiments were not reproduced.

## Reproduction

Build the corpus and run the main comparison:

```bash
python benchmarks/dust_benchmark.py \
    --corpus /tmp/opencode/dust_corpus.bin \
    --steps 40 --seeds 0 1 2 --seq-len 64 \
    --bp-lrs 3e-4 3e-3 \
    --dust-configs k2:0.05 k2:0.02 k4:0.05 \
    --precision-reference \
    --json-out benchmarks/dust_results.json
```

The reported runs used the explicit
`/tmp/opencode/dust_corpus.bin` described above (342,955 bytes; SHA-256
`813cfbd2...`). The script builds a corpus from repository text if that file
and `/tmp/opencode/precision_corpus.bin` do not exist; because the repository
changes, the rebuilt hash can differ from the reported one, and the JSON
records whichever file was actually used. Raw per-run JSON is written under a
temporary work directory; `--resume` reuses it.

The 60-step follow-up in this report:

```bash
python benchmarks/dust_benchmark.py --corpus /tmp/opencode/dust_corpus.bin \
    --steps 60 --seeds 11 --seq-len 64 --eval-tokens 8192 --eval-every 10 \
    --bp-lrs 3e-4 3e-3 \
    --dust-configs k2:0.05 k2:0.02 k4:0.05 \
    --json-out benchmarks/dust_results_60.json
```

The memory points (one per sequence length):

```bash
for seq in 64 256 512; do
  python benchmarks/dust_benchmark.py --corpus /tmp/opencode/dust_corpus.bin \
      --steps 5 --seeds 0 --seq-len "$seq" --eval-tokens "$seq" --eval-every 5 \
      --bp-lrs 3e-3 --dust-configs k2:0.02 \
      --json-out "/tmp/opencode/dust_mem_${seq}.json"
done
```

The natural-language check:

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    -o /tmp/opencode/tinyshakespeare.txt
python benchmarks/dust_benchmark.py --corpus /tmp/opencode/tinyshakespeare.txt \
    --steps 40 --seeds 0 --seq-len 64 --eval-tokens 8192 --eval-every 10 \
    --bp-lrs 3e-4 3e-3 --dust-configs k2:0.05 k2:0.02 \
    --json-out benchmarks/dust_results_shakespeare.json
```

Run the tests with:

```bash
python -m pytest -q
```

The default suite excludes the stale `tests/_reference` Torch/legacy
expectations.

## Bottom line

Dust is implemented faithfully enough to test the paper's core mechanism:
activation perturbations, per-token rewards, local attention credit, head
slabs, and SGD updates without a backward pass. It genuinely learns on real
byte text and checkpoints/resumes exactly. The speed work made the reference
implementation about 3.1x faster (at sequence 64, 1.21 s to 0.39 s per step;
at 512, 10.05 s to 3.35 s) without changing the estimator or checkpoint
semantics. At this CPU scale, however, it is still about 6-7x slower per
training step than tuned FP32 backprop at K=2 and 10.5x at K=4 in the 40-step
run, and it does not reliably beat that baseline. Its clean activation cache
is smaller, but peak RSS is not: at sequence 512 the 5-step memory point
peaked at 118.6 MiB for Dust versus 103.8 MiB for backprop, because the
concurrent head pass, draw scratch, and allocator retention dominate. Larger
populations may close the quality gap, as the paper reports, but they would
increase the compute and scratch-memory cost further. Dust therefore remains
an experimental research path, not a recommended default.
