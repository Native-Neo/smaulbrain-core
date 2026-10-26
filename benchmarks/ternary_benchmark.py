#!/usr/bin/env python3
"""Controlled packed-ternary vs FP32 forward/backward benchmark.

Every mode starts from the same parameters and token stream. Unless
``--frozen-weights`` is given, each step also runs the optimizer and
invalidates the packed-weight cache, matching the training loop; forward time
therefore includes re-packing changed ternary weights.

Examples:
    python benchmarks/ternary_benchmark.py --steps 20 --seq-lens 64 512
    python benchmarks/ternary_benchmark.py --frozen-weights --modes fp32 ternary
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

import numpy as np

from smaulbrain import config as cfg
from smaulbrain import model as M
from smaulbrain import quant
from smaulbrain import smaulopt as optmod


MODES = {
    "fp32": ("fp32", "fp32"),
    "ternary-ste": ("ternary", "fp32"),
    "ternary": ("ternary", "ternary"),
}


def _median(values):
    return float(statistics.median(values))


def _measure(spec, tokens, forward_quant, backward_mode, args):
    params = M.build_params(spec, np.random.default_rng(args.seed),
                            spec.n_experts_init, mtp_depth=args.mtp_depth)
    model = M.Model(spec, params, forward_quant=forward_quant,
                    backward_mode=backward_mode, mtp_depth=args.mtp_depth,
                    activation_storage=args.activation_storage,
                    gradient_storage=args.gradient_storage)
    opt = optmod.SmaulOPT(lr=args.lr)
    opt.register(model.p)

    pack_state = {"pack_s": 0.0, "cache_get_s": 0.0, "pack_builds": 0}
    if args.profile_phases:
        original_get = model.packed.get
        original_prefetch = model.packed.prefetch

        def timed_get(key, weight):
            before = model.packed.builds
            started = time.perf_counter()
            result = original_get(key, weight)
            elapsed = time.perf_counter() - started
            pack_state["cache_get_s"] += elapsed
            if model.packed.builds != before:
                pack_state["pack_s"] += elapsed
                pack_state["pack_builds"] += 1
            return result

        def timed_prefetch(*args, **kwargs):
            before = model.packed.builds
            started = time.perf_counter()
            result = original_prefetch(*args, **kwargs)
            elapsed = time.perf_counter() - started
            pack_state["cache_get_s"] += elapsed
            built = model.packed.builds - before
            if built:
                pack_state["pack_s"] += elapsed
                pack_state["pack_builds"] += built
            return result

        model.packed.get = timed_get
        model.packed.prefetch = timed_prefetch

    update = not args.frozen_weights
    for _ in range(args.warmup):
        loss, grads, _info = model.loss_and_backward(tokens)
        if not math.isfinite(loss):
            raise RuntimeError("non-finite loss during warmup")
        if update:
            opt.step(model.p, grads, collect_norms=False)
            model.packed.invalidate(grads.keys())
            model.fp8.invalidate(grads.keys())

    # Check the timed configuration once without charging it to the
    # measurements. Dense-tiny gives every parameter a gradient.
    loss, grads, _info = model.loss_and_backward(tokens)
    if not math.isfinite(loss):
        raise RuntimeError("non-finite loss before measurement")
    for name, grad in grads.items():
        if not np.isfinite(grad).all():
            raise RuntimeError(f"non-finite gradient for {name}")

    records = []
    for _ in range(args.steps):
        if args.profile_phases:
            pack_state["pack_s"] = 0.0
            pack_state["cache_get_s"] = 0.0
            pack_state["pack_builds"] = 0
        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        loss, grads, info = model.loss_and_backward(tokens, profile=True)
        wall1 = time.perf_counter()
        cpu1 = time.process_time()
        optimizer_s = 0.0
        invalidate_s = 0.0
        if update:
            mark = time.perf_counter()
            opt.step(model.p, grads, collect_norms=False)
            middle = time.perf_counter()
            model.packed.invalidate(grads.keys())
            model.fp8.invalidate(grads.keys())
            end = time.perf_counter()
            optimizer_s = middle - mark
            invalidate_s = end - middle
        wall2 = time.perf_counter()
        cpu2 = time.process_time()
        profile = info["profile"]
        records.append({
            "loss": float(loss),
            "forward_s": float(profile["forward_s"]),
            "backward_s": float(profile["backward_s"]),
            "step_s": float(wall2 - wall0),
            "step_cpu_s": float(cpu2 - cpu0),
            "optimizer_s": float(optimizer_s),
            "invalidate_s": float(invalidate_s),
            # Gradients are copied out of the graph after the profiled
            # backward; this is the remaining part of the step.
            "grad_extract_s": float((wall1 - wall0)
                                    - profile["forward_s"]
                                    - profile["backward_s"]),
            "pack_s": float(pack_state["pack_s"]) if args.profile_phases else None,
            "pack_builds": (int(pack_state["pack_builds"])
                            if args.profile_phases else None),
            "cache_get_s": (float(pack_state["cache_get_s"])
                            if args.profile_phases else None),
        })

    metrics = {
        "forward_s": _median([r["forward_s"] for r in records]),
        "backward_s": _median([r["backward_s"] for r in records]),
        "step_s": _median([r["step_s"] for r in records]),
        "step_cpu_s": _median([r["step_cpu_s"] for r in records]),
        "optimizer_s": _median([r["optimizer_s"] for r in records]),
        "invalidate_s": _median([r["invalidate_s"] for r in records]),
        "grad_extract_s": _median([r["grad_extract_s"] for r in records]),
        "pack_s": (_median([r["pack_s"] for r in records])
                   if args.profile_phases else None),
        "pack_builds": (_median([r["pack_builds"] for r in records])
                        if args.profile_phases else None),
        "cache_get_s": (_median([r["cache_get_s"] for r in records])
                        if args.profile_phases else None),
        "first_loss": records[0]["loss"],
        "last_loss": records[-1]["loss"],
        "samples": records,
    }
    return metrics


def _run_once(spec, args, modes):
    sequences = []
    for seq_len in args.seq_lens:
        rng = np.random.default_rng(args.seed + 1)
        tokens = rng.integers(0, spec.vocab_size, seq_len).astype(np.int64)
        configs = {}
        for mode in modes:
            forward_quant, backward_mode = MODES[mode]
            configs[mode] = _measure(
                spec, tokens, forward_quant, backward_mode, args)
        sequences.append({
            "seq_len": int(seq_len),
            "modes": configs,
        })
    return sequences


def _aggregate(runs):
    """Median of per-run medians, preserving the sample records from run 0."""
    first = runs[0]
    out = []
    for seq_idx, seq in enumerate(first):
        modes = {}
        for mode, metrics in seq["modes"].items():
            aggregated = {}
            for key, value in metrics.items():
                if key in ("first_loss", "last_loss", "samples") or value is None:
                    aggregated[key] = value
                else:
                    aggregated[key] = _median(
                        [run[seq_idx]["modes"][mode][key] for run in runs])
            modes[mode] = aggregated
        out.append({"seq_len": seq["seq_len"], "modes": modes})
    return out


def _comparisons(sequences):
    for seq in sequences:
        base = seq["modes"].get("fp32")
        if base is None:
            continue
        for mode, metrics in seq["modes"].items():
            for metric in ("forward_s", "backward_s", "step_s"):
                if metrics[metric] > 0.0:
                    metrics[metric + "_vs_fp32"] = base[metric] / metrics[metric]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="dense-tiny", choices=sorted(cfg.PRESETS))
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[64, 512])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=1,
                    help="fresh model/token-stream repetitions; report the "
                         "median of per-run medians")
    ap.add_argument("--modes", nargs="+", default=["fp32", "ternary"],
                    choices=sorted(MODES))
    ap.add_argument("--frozen-weights", action="store_true",
                    help="do not run the optimizer or invalidate packed weights")
    ap.add_argument("--activation-storage", default="fp32",
                    choices=["fp32", "bf16", "int8"])
    ap.add_argument("--gradient-storage", default="fp32",
                    choices=["fp32", "bf16"])
    ap.add_argument("--profile-phases", action="store_true",
                    help="time packed-cache rebuilds separately; adds a small "
                         "Python wrapper around lookups and prefetch")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mtp-depth", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if any(seq_len < 2 for seq_len in args.seq_lens):
        raise SystemExit("--seq-lens values must be at least 2")
    if not (args.lr >= 0.0) or not math.isfinite(args.lr):
        raise SystemExit("--lr must be a finite non-negative number")
    if args.mtp_depth < 0:
        raise SystemExit("--mtp-depth must be non-negative")

    spec = cfg.PRESETS[args.preset]
    runs = []
    for run_idx in range(args.runs):
        # Rotate mode order between runs so warm-up, thermal drift, and
        # background load do not systematically favor one configuration.
        shift = run_idx % len(args.modes)
        modes = args.modes[shift:] + args.modes[:shift]
        runs.append(_run_once(spec, args, modes))
    sequences = _aggregate(runs)
    _comparisons(sequences)
    result = {
        "preset": args.preset,
        "ternary_backend": quant.ternary_backend_name(),
        "seed": args.seed,
        "mtp_depth": args.mtp_depth,
        "frozen_weights": bool(args.frozen_weights),
        "activation_storage": args.activation_storage,
        "gradient_storage": args.gradient_storage,
        "profile_phases": bool(args.profile_phases),
        "steps": args.steps,
        "warmup": args.warmup,
        "runs": args.runs,
        "modes": list(args.modes),
        "threads": {
            "smaulbrain": os.environ.get("SMAULBRAIN_THREADS"),
            "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp": os.environ.get("OMP_NUM_THREADS"),
        },
        "loadavg_after": list(os.getloadavg()),
        "sequences": sequences,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
