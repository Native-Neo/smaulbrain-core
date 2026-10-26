# smaulbrain-core

Byte-level recurrent language model with mixture-of-experts, ternary/FP8
quantization, and a factored optimizer, in pure NumPy (CPU-only, float32).

## Features

- **Byte-level vocabulary** (259 tokens: identity bytes 0..255 plus BOS/EOS/PAD);
  no tokenizer dependency.
- **Recurrent linear attention**: per-layer state is O(d_model), independent of
  sequence length, with fp32 accumulation.
- **Mixture-of-experts**: shared expert pool with top-k routing, capacity
  control, and evidence-based expert growth (leave-one-out held-out loss).
- **Quantization**: fp32, packed 2-bit ternary (with an optional native C++
  backend and a portable NumPy fallback), and software-emulated FP8 E4M3.
- **Crash-safe checkpoints**: atomic publish, sha256-verified safetensors
  payloads (`params.safetensors`, `optim.safetensors`), `.bak` fallback
  recovery, and a two-phase growth protocol (pending, then committed) that
  resumes exactly.
- File-backed training streams are SHA-256 fingerprinted at startup and saved
  with checkpoint source metadata; resume checks content before continuing.
  Older checkpoints remain loadable and validate only their recorded path and
  size.
- **Expert pruning** from checkpoints using held-out loss evidence.
- **Multi-token prediction (MTP)**: DeepSeek-V3-style lightweight modules that
  predict several tokens ahead with the shared embedding/head, plus
  **speculative decoding** that drafts with MTP and verifies in one batched
  forward (greedy output is token-identical to plain decoding).
- **Experimental Dust trainer**: opt-in activation-perturbation training
  without autograd for dense FP32 models. Backpropagation remains the default;
  see [Experimental: Dust activation-perturbation training](#experimental-dust-activation-perturbation-training).

## Presets

| preset     | kind  | d_model | layers | params (init)   |
|------------|-------|---------|--------|-----------------|
| dense-tiny | dense | 32      | 15     | 256,000         |
| dense-full | dense | 1000    | 19     | 256,000,000     |
| moe-tiny   | moe   | 64      | 4      | 412,928 (32 experts, top-2) |
| moe-full   | moe   | 512     | 6      | 334,149,632 (64 experts, top-8) |

Parameter counts above are with `--mtp-depth 0`. Each enabled MTP depth adds
`6*d_model^2 + 6*d_model` shared params: 6,336 per depth for dense-tiny,
1,575,936 per depth for moe-full.

## Install

```bash
pip install smaulbrain-core
```

From source (this repository):

```bash
pip install .
```

## Quick start

Forward/backward on a synthetic batch:

```python
import numpy as np
from smaulbrain import config as cfg, model as M

spec = cfg.PRESETS["moe-tiny"]
params = M.build_params(spec, np.random.default_rng(0), spec.n_experts_init)
model = M.Model(spec, params)
loss, grads, info = model.loss_and_backward([65, 66, 67, 68])
```

Training with the crash-safe loop:

```python
import argparse
from smaulbrain import cli, config as cfg, train_loop as TL, data as D

ap = argparse.ArgumentParser()
cli.add_common_args(ap, need_train=True)
args = ap.parse_args(["--preset", "moe-tiny", "--max-steps", "100",
                      "--seq-len", "128", "--checkpoint", "checkpoints/run"])
preset = cli.validate_args(args)
stream = D.ByteStream(data=D.synth_bytes(1 << 16, seed=0))
trainer = TL.Trainer(cfg.PRESETS[preset], args, stream)
trainer.train(max_steps=args.max_steps, ckpt_every=50)
```

Resume, prune, and inference reuse the same checkpoint directory:

```python
trainer.load("checkpoints/run")                        # exact resume
from smaulbrain import prune_cmd, inference
prune_cmd.prune_checkpoint("checkpoints/run", rm_worst=2, eval_batches=[tokens],
                           min_samples=256, assume_yes=True)
engine = inference.InferenceEngine(trainer.model, temperature=0.0, speculative=2)
print(engine.generate([65, 66, 67], max_new_tokens=16))
```

## Multi-token prediction (MTP)

`--mtp-depth K` (default 10) adds K shared modules on top of the main model.
Depth k predicts `tok[i+k+1]` from the main hidden at position `i` and the
embedding of `tok[i+k]`, using a 2d->d projection, a recurrent-attention block
and the tied embedding/head (DeepSeek-V3 style, no stop-gradient). With
`K=10`, one hidden state at position `i` is trained to predict the next 10
tokens, one depth per step ahead (any depth works; 60+ is supported):

```
loss = CE(main) + Σ_k  mtp_loss_weight * CE(depth k)      # default weight 0.1
```

Deep chains add one small module per depth (roughly `6d²+6d` params each) and
one sequential module forward per training step, so they cost more than they
change the main loss; lower `--mtp-loss-weight` if the aggregate auxiliary
term (`K * weight`, 1.0 at the default 10, 6.0 at depth 60) crowds the main
loss.

- MTP params are shared across experts, so growth/pruning/paging are
  unaffected; they are saved/loaded like every other parameter.
- The checkpoint owns the architecture: loading an older (pre-MTP) checkpoint
  resumes with `mtp_depth=0` instead of failing the exact-count check, and a
  checkpoint's depth/weight override the command line on resume.
- `mtp_param_count(spec, depth)` and the `mtp_depth`/`mtp_params` fields of
  `arch_report()` report the extra capacity.

## Speculative decoding

`InferenceEngine(model, speculative=K)` drafts up to K tokens with the MTP
modules, verifies them with a single batched main-model forward, and accepts
the matching prefix:

- `temperature=0`: greedy output is token-identical to plain decoding (the
  verifier scores every position with the exact single-token math, and
  capacity limiting is disabled because sequential decoding never drops
  tokens).
- `temperature>0`: drafts are accepted with the standard `min(1, p/q)` rule
  and rejects are resampled from the normalized residual `(p-q)+`, so the
  emitted distribution is unchanged.
- Draft-module states are advanced only over verified tokens (never polluted
  by rejected drafts); `export_states()`/`import_states()` carry the draft
  pipeline along when it exists, so greedy generation resumes exactly.
- Any trained depth can draft: a deeper module whose state still lags the
  frontier is advanced *tentatively* inside the draft pass over the drafts of
  that same pass (e.g. depth 3 needs `tok[p+2] = draft 1` to reach position
  `p`), one batched step per depth, and the tentative states are discarded at
  the end of the round. `engine.draft_depth = min(speculative, mtp_depth)`.
- Cost grows quadratically with the draft depth (each round re-extends every
  depth from its verified position, about `K²/2` module rows for K drafts),
  so `speculative` around 8-16 is the practical sweet spot on CPU; large K
  (up to the model depth, 10 by default) trades speed for maximum lookahead.


## Quantization modes

Supported `forward_quant`/`backward_mode` combinations:
`fp32/fp32`, `fp8/fp8`, `ternary/fp32`, `ternary/ternary`, `ternary/fp8`.

Ternary weights are executed directly from packed 2-bit storage. On CPU the
package compiles a small C++ kernel on first use (cached); without a compiler
it transparently falls back to a NumPy decode-and-multiply path. FP8 is
software-emulated on CPU.

Packed ternary is faster for complete training steps on the dense workload.
On `dense-tiny` at sequence 512, with optimizer updates and re-packing included,
the measured step was 146.1 ms versus 197.6 ms for FP32 (1.35x); forward was
70.8 ms versus 127.8 ms (1.81x). Backward was 65.6 ms versus 60.9 ms because
the straight-through weight gradient remains an FP32 GEMM even though the
input gradient comes from packed weights. At sequence 64 the step is about
1.12x faster. See
[`benchmarks/TERNARY_RESULTS.md`](benchmarks/TERNARY_RESULTS.md) for the frozen
pure-compute comparison, kernel A/B numbers, and reproduction commands:

```bash
python benchmarks/ternary_benchmark.py --steps 15 --warmup 3 --runs 3 \
    --seq-lens 64 512 --modes fp32 ternary
```

The NCM runtime can also use native sparse direct-ternary update helpers. They
sample update coordinates from a strided gradient view without building Python
magnitude/probability/mask tensors, then apply queued updates to packed trits in
C++ without concatenating the per-backward update lists. These helpers are
optional APIs; callers can detect their availability and retain a portable
fallback.

## Reduced-precision activation and gradient storage

Training keeps every differentiable intermediate alive until backward. At long
sequence lengths those cached activations, not the parameters, dominate RAM.
Two opt-in flags compress that cache and completed parameter gradients; the
default remains the original all-float32 path. They are accepted by
`cli.add_common_args()` alongside the other training options:

```text
--activation-storage {fp32,bf16,int8}   # cached activations
--gradient-storage {fp32,bf16}          # completed parameter gradients
```

For example, with the `argparse` setup from the quick start:

```python
args = ap.parse_args([
    "--preset", "dense-tiny", "--seq-len", "512",
    "--activation-storage", "bf16",
    "--gradient-storage", "bf16",
    "--checkpoint", "checkpoints/run",
])
```

| format | cached activation storage | notes |
|--------|---------------------------|-------|
| `fp32` | 4 bytes/value | default, bit-for-bit original path |
| `bf16` | 2 bytes/value | raw IEEE bfloat16 bits in `uint16`, round-to-nearest-even; no `ml_dtypes` dependency |
| `int8` | 1 byte/value + one scale | dynamic symmetric per-tensor scale; computed-scale codes stay in `[-127, 127]`, while caller-supplied scales are clipped |

Implementation details and limits:

- BF16 is stored as its actual 16-bit bit pattern, not as a float32 array that
  was briefly cast. Decoding widens with a shift; `NaN`, `±Inf`, and signed zero
  are preserved. Float32 subnormals follow IEEE bfloat16 rounding, so values
  below the bfloat16 subnormal range round to zero. `Tensor.data` dequantizes
  on access, so the logical dtype remains float32.
- INT8 rejects non-finite tensors instead of silently saturating. All-zero
  tensors use a unit scale. When the scale is computed from the tensor, the
  encoder skips a redundant clipping pass because the rounded codes are
  already in range; an explicit scale is clipped to the signed range.
- Reduced activation modes use float32 transcendental math for SiLU, GELU,
  sigmoid, and softplus. The FP32 reference path keeps float64 math. This is
  below the storage-quantization error already present in BF16/INT8 and avoids
  unnecessary conversion and recomputation work.
- `--gradient-storage bf16` compresses each parameter's completed gradient
  buffer after float32 accumulation. Optimizer state remains float32, and
  clipping/updates decode one parameter at a time.
- Precision-critical values stay float32 even when compression is enabled:
  RMSNorm statistics, router scores/expert selection, recurrent scan state
  history, softmax/cross-entropy and MSE losses, final logits, optimizer state,
  and master parameters. Model architecture, checkpoint format, and loss
  definitions are unchanged.
- Compression is primarily a memory optimization. With the optimized codecs,
  dense-tiny at sequence 512 measured 12-16% lower CPU step time for BF16/INT8
  than FP32, while at sequence 64 the modes are within measurement noise. The
  relative conversion cost depends on tensor shapes, sequence length, and
  preset; benchmark the target configuration before choosing a mode.
- Storage choices are runtime options, not checkpoint architecture. They are
  recorded in the manifest for reproducibility, and old checkpoints load with
  any setting.

`benchmarks/precision_benchmark.py` runs identical seeds, data, clipping, and
optimizer settings across configurations and reports peak RSS, activation-cache
bytes, gradient bytes, and per-phase timings:

```bash
python benchmarks/precision_benchmark.py --preset dense-tiny --seq-len 512 \
    --steps 60 --lr 1e-3 --activation-storage bf16 --gradient-storage bf16
```

Measured memory/throughput/quality tables, codec validation, and reproduction
commands are in [`benchmarks/PRECISION_RESULTS.md`](benchmarks/PRECISION_RESULTS.md).

## Experimental: Dust activation-perturbation training

`--trainer dust` selects an isolated alternative to backpropagation based on
the Dust activation-perturbation estimator
([paper](https://qlabs.sh/research/dust),
[official code](https://github.com/qlabs-eng/dust)). It is **experimental and
not the default**. The quality measurements in this repository are CPU-scale
checks on `dense-tiny`, sequence length 64, 2-4 draws, and 40-60 steps; memory
points up to sequence length 512 are reported separately. This is not a
reproduction of the paper's pretraining runs, which use much larger corpora
and populations.

What is implemented:

- Gaussian perturbations of cached linear-layer outputs, per-token centered
  loss rewards, future-token credit for keys/values, and direct loss credit for
  writer/hub/embedding sites. Attention internals are credited through the
  estimated attention-output error, as in the paper.
- An independent NumPy forward (`smaulbrain/dust.py`) that never calls the
  autograd tape or `Model.loss_and_backward`. Clean-forward and from-block
  rerun parity are covered by tests.
- SGD with momentum, configurable draws/sigmas/gamma/seed, deterministic
  per-step perturbation RNG, and exact checkpoint resume. The checkpoint
  manifest records `algorithm: dust` and the full Dust configuration, and
  loading a Dust checkpoint in backprop mode (or vice versa) is rejected.
- A batched performance path: perturbed nonlinearities use float32, independent
  direct groups share one forward, and the head uses adaptive float32 slabs.
  On multi-CPU machines the head runs in a deterministic background thread
  while the block loop runs; the cloned RNG stream preserves serial results.
  Tests require bitwise equality between batched/parallel and separate/serial
  paths. The clean forward still uses the exact reference arithmetic.
- Ctrl+C during a Dust step is deferred until the update commits; the trainer
  then saves a resumable checkpoint.

Limits:

- **Dense FP32 only.** MoE, MTP depth greater than zero, FP8/ternary
  forward/backward modes, and activation or gradient storage other than
  `fp32` are rejected with an explicit error. These combinations were not
  implemented because the verified estimator does not define them for this
  architecture, or because their forward arithmetic would change the
  perturbation semantics.
- Non-matrix parameters (RMSNorm gains, `attn.w`, `attn.u`) are **frozen by
  default**. `--dust-scalar-draws K` enables an experimental two-sided
  weight-space ES fallback over all of them. That is broader than the paper's
  two residual scalars per layer and was not beneficial in the CPU tests.
- Dust still stores one clean activation cache, a gradient dict for the trained
  matrix parameters, and SGD momentum state. It removes the autograd graph and
  activation gradients; it is not a constant-memory trainer. Clean-cache bytes
  are smaller than backprop's activation cache, but in the matched CPU memory
  points peak RSS did not improve, and at sequence 512 Dust still peaked
  higher than backprop because its concurrent head pass, draw scratch, and
  allocator retention dominated (118.6 versus 103.8 MiB). The memory argument
  depends on a fused or memory-reusing estimator.
- The speed work made the earlier serial NumPy implementation about 3.1x
  faster (sequence 64: 1.21 s to 0.39 s per training step; sequence 512:
  10.05 s to 3.35 s). It is still slower than tuned FP32 backprop: about 6-7x
  at K=2 and 10.5x at K=4 in the 40-step dense-tiny comparison, and about 6x
  and 9.2x in the single-seed 60-step run. The step-time ratio grows with
  sequence length (about 12x at 512 in the memory points). The paper's
  population is 256 to 16,384 draws; the default draw counts here are
  deliberately small.

Example, using the `argparse` setup from the quick start:

```python
args = ap.parse_args([
    "--preset", "dense-tiny", "--trainer", "dust",
    "--seq-len", "64", "--mtp-depth", "0",
    "--checkpoint", "checkpoints/dust",
    "--dust-writer-draws", "2", "--dust-hidden-draws", "2",
    "--dust-hub-draws", "2", "--dust-embed-draws", "2",
    "--dust-local-draws", "2", "--dust-head-draws", "1024",
    "--dust-gamma", "0.5", "--dust-lr", "0.05",
    "--dust-momentum", "0.9", "--dust-seed", "1234",
])
```

`--dust-sigma`, `--dust-hub-sigma`, `--dust-hub-sigma-deep`,
`--dust-local-sigma`, and `--dust-head-sigma` control the perturbation
scales. The defaults are 0.05, tuned down from the paper's 0.2/0.4 because
SmaulBRAIN's activation scale makes the larger perturbations biased. The
default credit decay is 0.5, chosen to match the recurrence decay; the paper's
0.98 is available explicitly. `--dust-grad-clip` defaults to 0 (no clipping).
`--grad-clip` is backprop's clipping flag; passing a non-default value in Dust
mode while `--dust-grad-clip` is left at 0 is rejected with a pointer to
`--dust-grad-clip` rather than being silently ignored.

`benchmarks/dust_benchmark.py` runs matched backprop and Dust configurations
in fresh subprocesses and reports held-out loss, training time, peak RSS,
activation-cache bytes, gradient bytes, optimizer bytes, and estimator
scratch bytes:

```bash
python benchmarks/dust_benchmark.py --steps 40 --seeds 0 1 2 \
    --dust-configs k2:0.05 k2:0.02 k4:0.05 --precision-reference
```

The measured tables, tuning ablations, and limitations are in
[`benchmarks/DUST_RESULTS.md`](benchmarks/DUST_RESULTS.md).

## Native kernels

`smaulbrain.kernels.build_here()` compiles the optional scalar C++ reference
kernels. Set `SMAULBRAIN_NATIVE_OPT=0` to force the pure-NumPy optimizer
update path.

## License

See `LICENSE`.
