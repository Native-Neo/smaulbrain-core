# smaulbrain-core

Byte-level recurrent language model with mixture-of-experts, ternary/FP8
quantization, and a factored optimizer, on PyTorch.

## Features

- **Byte-level vocabulary** (259 tokens: identity bytes 0..255 plus BOS/EOS/PAD);
  no tokenizer dependency.
- **Recurrent linear attention**: per-layer state is O(d_model), independent of
  sequence length, with fp32 accumulation.
- **Mixture-of-experts**: shared expert pool with top-k routing, capacity
  control, and evidence-based expert growth (leave-one-out held-out loss).
- **Quantization**: fp32, packed 2-bit ternary (with an optional native C++
  backend and a portable NumPy fallback), and software-emulated FP8 E4M3.
- **Crash-safe checkpoints**: atomic publish, sha256-verified payloads,
  `.bak` fallback recovery, and a two-phase growth protocol (pending, then
  committed) that resumes exactly.
- **Expert pruning** from checkpoints using held-out loss evidence.
- **Multi-token prediction (MTP)**: DeepSeek-V3-style lightweight modules that
  predict several tokens ahead with the shared embedding/head, plus
  **speculative decoding** that drafts with MTP and verifies in one batched
  forward (greedy output is token-identical to plain decoding).

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
import torch
from smaulbrain import config as cfg, model as M

spec = cfg.PRESETS["moe-tiny"]
params = M.build_params(spec, torch.Generator().manual_seed(0), spec.n_experts_init)
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

The NCM runtime can also use native sparse direct-ternary update helpers. They
sample update coordinates from a strided gradient view without building Python
magnitude/probability/mask tensors, then apply queued updates to packed trits in
C++ without concatenating the per-backward update lists. These helpers are
optional APIs; callers can detect their availability and retain a portable
fallback.

## Native kernels

`smaulbrain.kernels.build_here()` compiles the optional scalar C++ reference
kernels. Set `SMAULBRAIN_NATIVE_OPT=0` to force the pure-PyTorch optimizer
update path.

## License

See `LICENSE`.
