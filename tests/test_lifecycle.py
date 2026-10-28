"""End-to-end lifecycle: train, crash-safe publish, fresh resume, prune,
reload, inference.  The quarantined torch-era copies of these flows
(tests/_reference/test_train_ckpt.py, test_growth_prune.py,
test_inference_spec.py) can no longer run; without a live pin, module-level
rot in rarely-exercised paths survives the suite (a dropped numpy import
shipped in prune_cmd and only surfaced here).
"""
import argparse

import numpy as np

from smaulbrain import cli, config as cfg, data as D, inference
from smaulbrain import prune_cmd, train_loop as TL


def _args(checkpoint):
    ap = argparse.ArgumentParser()
    cli.add_common_args(ap, need_train=True)
    args = ap.parse_args([
        "--preset", "moe-tiny", "--checkpoint", str(checkpoint),
        "--seq-len", "32", "--mtp-depth", "2", "--growth-every-steps", "0",
        "--trainer", "backprop", "--seed", "7",
    ])
    cli.validate_args(args)
    return args


def _trainer(checkpoint, data):
    return TL.Trainer(cfg.PRESETS["moe-tiny"], _args(checkpoint),
                      D.ByteStream(data=data))


def test_train_resume_prune_infer_lifecycle(tmp_path):
    data = D.synth_bytes(1 << 16, seed=0)
    ckpt = tmp_path / "run"

    tr = _trainer(ckpt, data)
    tr.train(max_steps=4, log_every=0, ckpt_every=2, grad_clip=1.0)
    assert tr.sched.global_step == 4
    assert (ckpt / "manifest.json").is_file()
    snapshot = {k: v.data.copy() for k, v in tr.model.p.items()}

    # A fresh trainer must restore bitwise.
    tr2 = _trainer(ckpt, data)
    tr2.load(ckpt)
    assert tr2.sched.global_step == 4
    for k, v in snapshot.items():
        assert np.array_equal(tr2.model.p[k].data, v), k

    tr2.train(max_steps=6, log_every=0, ckpt_every=2, grad_clip=1.0)
    tr2.save()

    eval_batches = [D.ByteStream(data=data).next_tokens(32) for _ in range(4)]
    report = prune_cmd.prune_checkpoint(str(ckpt), 2, eval_batches,
                                        min_samples=1, dry_run=False,
                                        assume_yes=True)
    tr3 = _trainer(ckpt, data)
    tr3.load(ckpt)
    experts = sum(1 for k in tr3.model.p
                  if k.startswith("expert.") and k.endswith(".Win"))
    assert experts == report["new_count"]

    # Speculative decoding must be exact greedy.
    engine = inference.InferenceEngine(tr3.model, seed=0, temperature=0.0,
                                       speculative=2)
    plain = inference.InferenceEngine(tr3.model, seed=0, temperature=0.0)
    a = engine.generate([65, 66, 67], max_new_tokens=8)
    b = plain.generate([65, 66, 67], max_new_tokens=8)
    assert list(a) == list(b)
    assert all(0 <= int(t) < 256 for t in a)
