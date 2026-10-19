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

## Presets

| preset     | kind  | d_model | layers | params (init)   |
|------------|-------|---------|--------|-----------------|
| dense-tiny | dense | 32      | 15     | 256,000         |
| dense-full | dense | 1000    | 19     | 256,000,000     |
| moe-tiny   | moe   | 64      | 4      | 412,928 (32 experts, top-2) |
| moe-full   | moe   | 512     | 6      | 334,149,632 (64 experts, top-8) |

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
engine = inference.InferenceEngine(trainer.model, temperature=0.0)
print(engine.generate([65, 66, 67], max_new_tokens=16))
```

## Quantization modes

Supported `forward_quant`/`backward_mode` combinations:
`fp32/fp32`, `fp8/fp8`, `ternary/fp32`, `ternary/ternary`, `ternary/fp8`.

Ternary weights are executed directly from packed 2-bit storage. On CPU the
package compiles a small C++ kernel on first use (cached); without a compiler
it transparently falls back to a NumPy decode-and-multiply path. FP8 is
software-emulated on CPU.

## Native kernels

`smaulbrain.kernels.build_here()` compiles the optional scalar C++ reference
kernels. Set `SMAULBRAIN_NATIVE_OPT=0` to force the pure-PyTorch optimizer
update path.

## License

See `LICENSE`.
