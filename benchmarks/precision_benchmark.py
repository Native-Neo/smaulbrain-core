#!/usr/bin/env python3
"""Controlled activation/gradient storage benchmark.

Runs the same model initialization, token stream, optimizer and clipping for
each storage configuration.  Peak RSS is sampled from /proc; activation-cache
and gradient bytes come from the model's opt-in profiling path (or are omitted
when running this harness against an older checkout that lacks it).

Examples:
    python benchmarks/precision_benchmark.py --steps 40 --seq-len 512
    python benchmarks/precision_benchmark.py --activation-storage bf16 \
        --gradient-storage bf16 --steps 40 --seq-len 512
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import resource
import threading
import time
import tracemalloc

import numpy as np

from smaulbrain import config as cfg
from smaulbrain import growth as growth
from smaulbrain import model as M
from smaulbrain import smaulopt as optmod

try:
    from smaulbrain import precision as P
except ImportError:  # running this harness against a pre-storage checkout
    P = None


def _is_packed(value):
    return P is not None and isinstance(value, P.PackedGradient)


def _gradient_array(value):
    return P.gradient_array(value) if P is not None else np.asarray(value)


def _gradient_nbytes(value):
    return value.nbytes if _is_packed(value) else np.asarray(value).nbytes


def _rss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


class RssSampler:
    def __init__(self, interval=0.002):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, _rss_kb())

    def __enter__(self):
        self.peak = _rss_kb()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False


def _supports(fn, name):
    return name in inspect.signature(fn).parameters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="dense-tiny", choices=sorted(cfg.PRESETS))
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-tokens", type=int, default=256)
    ap.add_argument("--mtp-depth", type=int, default=0)
    ap.add_argument("--activation-storage", default="fp32",
                    choices=list(P.ACTIVATION_STORAGE_MODES) if P else ["fp32"])
    ap.add_argument("--gradient-storage", default="fp32",
                    choices=list(P.GRADIENT_STORAGE_MODES) if P else ["fp32"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--corpus", default=None,
                    help="optional raw-byte corpus; split into train/eval streams "
                         "instead of uniform random tokens")
    ap.add_argument("--eval-fraction", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tracemalloc", action="store_true",
                    help="also report peak Python/NumPy allocation (slower)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    spec = cfg.PRESETS[args.preset]
    params = M.build_params(spec, np.random.default_rng(args.seed),
                            spec.n_experts_init, mtp_depth=args.mtp_depth)
    model_kwargs = dict(mtp_depth=args.mtp_depth)
    if _supports(M.Model.__init__, "activation_storage"):
        model_kwargs.update(activation_storage=args.activation_storage,
                            gradient_storage=args.gradient_storage)
    model = M.Model(spec, params, **model_kwargs)
    del params
    opt = optmod.SmaulOPT(lr=args.lr)
    opt.register(model.p)

    # Fixed training and held-out token streams; identical for every config.
    train_rng = np.random.default_rng(args.seed + 1)
    eval_rng = np.random.default_rng(args.seed + 2)
    if args.corpus:
        if not 0.0 < args.eval_fraction < 1.0:
            raise SystemExit("--eval-fraction must be between 0 and 1")
        with open(args.corpus, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
        split = int(raw.size * (1.0 - args.eval_fraction))
        train_raw, eval_raw = raw[:split], raw[split:]
        if train_raw.size < args.seq_len + 1:
            raise SystemExit("corpus training split is shorter than --seq-len")
        if eval_raw.size < args.eval_tokens + 1:
            raise SystemExit("corpus eval split is shorter than --eval-tokens")
        starts = train_rng.integers(0, train_raw.size - args.seq_len + 1, size=args.steps)
        tokens = np.stack([train_raw[s:s + args.seq_len] for s in starts])
        eval_start = int(eval_rng.integers(0, eval_raw.size - args.eval_tokens + 1))
        eval_tokens = eval_raw[eval_start:eval_start + args.eval_tokens]
    else:
        tokens = train_rng.integers(0, 256, (args.steps, args.seq_len)).astype(np.int64)
        eval_tokens = eval_rng.integers(0, 256, (args.eval_tokens,)).astype(np.int64)

    profiled = _supports(M.Model.loss_and_backward, "profile")
    if args.tracemalloc:
        tracemalloc.start()
        tracemalloc.reset_peak()
    steps = []
    start_rss_kb = _rss_kb()
    with RssSampler() as sampler:
        for i in range(args.steps):
            tok = tokens[i]
            c0 = time.process_time()
            t0 = time.perf_counter()
            if profiled:
                loss, grads, info = model.loss_and_backward(tok, profile=(i == args.warmup))
            else:
                loss, grads, info = model.loss_and_backward(tok)
            t1 = time.perf_counter()
            total_sq = 0.0
            grad_bytes = 0
            packed = 0
            for g in grads.values():
                dense = _gradient_array(g)
                grad_bytes += _gradient_nbytes(g)
                packed += int(_is_packed(g))
                total_sq += float(np.sum(dense.astype(np.float64) ** 2))
            grad_norm = math.sqrt(total_sq)
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at step {i}")
            clip_scale = (args.grad_clip / grad_norm
                          if grad_norm > args.grad_clip and grad_norm > 0 else 1.0)
            if clip_scale != 1.0:
                for g in grads.values():
                    if _is_packed(g):
                        g.mul_(clip_scale)
                    else:
                        g *= clip_scale
            t2 = time.perf_counter()
            opt.step(model.p, grads, collect_norms=False)
            t3 = time.perf_counter()
            c1 = time.process_time()
            record = {
                "step": i,
                "loss": float(loss),
                "loss_main": float(info.get("loss_main", loss)),
                "mtp_losses": list(info.get("mtp_losses", [])),
                "grad_norm": grad_norm,
                "grad_bytes": int(grad_bytes),
                "packed_gradients": int(packed),
                "backward_and_grad_s": t1 - t0,
                "clip_s": t2 - t1,
                "optimizer_s": t3 - t2,
                "step_cpu_s": c1 - c0,
                "rss_kb": _rss_kb(),
            }
            if "profile" in info:
                record["profile"] = info["profile"]
            steps.append(record)
            if args.eval_every and (i + 1) % args.eval_every == 0:
                record["eval_loss"] = float(growth.eval_loss_on_tokens(model, eval_tokens))

    measured = [s for s in steps if s["step"] >= args.warmup]
    mean_backward = sum(s["backward_and_grad_s"] for s in measured) / max(1, len(measured))
    mean_step = sum(s["backward_and_grad_s"] + s["clip_s"] + s["optimizer_s"]
                    for s in measured) / max(1, len(measured))
    mean_cpu_step = sum(s["step_cpu_s"] for s in measured) / max(1, len(measured))
    tracemalloc_peak = None
    if args.tracemalloc:
        tracemalloc_peak = int(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    result = {
        "preset": args.preset,
        "seq_len": args.seq_len,
        "mtp_depth": args.mtp_depth,
        "corpus": args.corpus,
        "eval_fraction": args.eval_fraction,
        "activation_storage": args.activation_storage,
        "gradient_storage": args.gradient_storage,
        "steps": args.steps,
        "warmup": args.warmup,
        "start_rss_kb": int(start_rss_kb),
        "peak_rss_kb": int(sampler.peak),
        "sampled_peak_rss_kb": int(sampler.peak),
        # ru_maxrss is the kernel's exact high-water mark for the process, so
        # it also catches allocations shorter than the sampler interval.
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "end_rss_kb": _rss_kb(),
        "tracemalloc_peak_bytes": tracemalloc_peak,
        "mean_backward_s": mean_backward,
        "mean_step_s": mean_step,
        "tokens_per_s": (args.seq_len / mean_step) if mean_step else None,
        "mean_step_cpu_s": mean_cpu_step,
        "tokens_per_cpu_s": (args.seq_len / mean_cpu_step) if mean_cpu_step else None,
        "first_loss": steps[0]["loss"],
        "last_loss": steps[-1]["loss"],
        "final_eval_loss": next(
            (s["eval_loss"] for s in reversed(steps) if "eval_loss" in s), None),
        "last_grad_norm": steps[-1]["grad_norm"],
        "last_grad_bytes": steps[-1]["grad_bytes"],
        "profile": next((s["profile"] for s in steps if "profile" in s), None),
        "history": steps,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
