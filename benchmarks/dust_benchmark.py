#!/usr/bin/env python3
"""Matched CPU benchmark: FP32 backpropagation versus experimental Dust.

The default comparison uses dense-tiny, sequence length 64, the same corpus
split, the same initialization seed and a fixed held-out evaluation window.
Each configuration runs in a fresh subprocess so ``ru_maxrss`` is a clean
process high-water mark.

Examples:
    # Quick smoke run (10 steps, one seed)
    python benchmarks/dust_benchmark.py --quick

    # Controlled comparison used for benchmarks/DUST_RESULTS.md
    python benchmarks/dust_benchmark.py \
        --corpus /tmp/opencode/dust_corpus.bin \
        --steps 40 --seeds 0 1 2 \
        --dust-configs k2:0.05 k2:0.02 k4:0.05 \
        --bp-lrs 3e-4 3e-3 \
        --precision-reference
"""
from __future__ import annotations

# Pin BLAS before NumPy is imported so every child and the parent agree.
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
             "MKL_NUM_THREADS", "FLEXBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import hashlib
import json
import math
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smaulbrain import cli
from smaulbrain import config as cfg
from smaulbrain import data as D
from smaulbrain import growth as G
from smaulbrain import train_loop as TL


# ---------------------------------------------------------------------------
# Corpus and small utilities
# ---------------------------------------------------------------------------
def build_corpus(path: Path) -> bytes:
    """Use an existing corpus or build one from the repository's text."""
    if path.exists():
        return path.read_bytes()
    parts = []
    for pattern in ("README.md", "PORT_NOTES.md", "smaulbrain/*.py", "tests/*.py"):
        for item in sorted(ROOT.glob(pattern)):
            parts.append(item.read_bytes())
    raw = b"".join(parts)
    if not raw:
        raise SystemExit("could not build a corpus")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def default_corpus() -> Path:
    for candidate in (Path("/tmp/opencode/dust_corpus.bin"),
                      Path("/tmp/opencode/precision_corpus.bin")):
        if candidate.exists():
            return candidate
    return Path("/tmp/opencode/dust_corpus.bin")


def rss_kb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _mb(value) -> float:
    return round(float(value) / (1024.0 * 1024.0), 3) if value is not None else None


def _rss_mb(kb) -> float:
    # Linux reports ru_maxrss and VmRSS in KiB.
    return round(float(kb) / 1024.0, 3) if kb is not None else None


def _summary(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# One child run
# ---------------------------------------------------------------------------
def _make_args(preset, mode, seed, seq_len, lr, dust_lr, draws,
               gamma, sigma, activation_storage, gradient_storage):
    ap = argparse.ArgumentParser()
    cli.add_common_args(ap, need_train=True)
    ckpt = str(Path(tempfile.gettempdir()) / "opencode" /
               f"dust_bench_ckpt_{os.getpid()}")
    argv = [
        "--preset", preset,
        "--checkpoint", ckpt,
        "--seq-len", str(seq_len),
        "--mtp-depth", "0",
        "--growth-every-steps", "0",
        "--seed", str(seed),
        "--trainer", mode,
        "--activation-storage", activation_storage,
        "--gradient-storage", gradient_storage,
        "--lr", str(lr),
        "--dust-lr", str(dust_lr),
        "--dust-gamma", str(gamma),
        "--dust-sigma", str(sigma),
        "--dust-hub-sigma", str(sigma),
        "--dust-hub-sigma-deep", str(sigma),
        "--dust-local-sigma", "0.05",
        "--dust-head-sigma", "0.05",
        "--dust-writer-draws", str(draws),
        "--dust-hidden-draws", str(draws),
        "--dust-hub-draws", str(draws),
        "--dust-embed-draws", str(draws),
        "--dust-local-draws", str(draws),
        "--dust-head-draws", "1024",
        "--dust-chunk", "8",
        "--dust-scalar-draws", "0",
        "--dust-grad-clip", "0",
    ]
    args = ap.parse_args(argv)
    cli.validate_args(args)
    return args


def _optimizer_nbytes(opt) -> int:
    total = 0
    for state in opt.state.values():
        for value in state.values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
    return total


def run_one(cfg_dict: dict) -> dict:
    """Run one configuration in this process and return a JSON-safe record."""
    preset = cfg_dict["preset"]
    mode = cfg_dict["mode"]
    seed = int(cfg_dict["seed"])
    steps = int(cfg_dict["steps"])
    seq_len = int(cfg_dict["seq_len"])
    eval_tokens_n = int(cfg_dict["eval_tokens"])
    eval_every = int(cfg_dict["eval_every"])
    corpus_path = Path(cfg_dict["corpus"])
    raw = build_corpus(corpus_path)
    corpus_sha = _sha(raw)
    split = int(len(raw) * 0.8)
    train_raw = raw[:split]
    eval_raw = raw[split:split + eval_tokens_n]
    if len(train_raw) < seq_len + 1:
        raise SystemExit("training split is shorter than --seq-len")
    if len(eval_raw) < 8:
        raise SystemExit("evaluation split is too short")
    eval_tokens = np.frombuffer(eval_raw, dtype=np.uint8).astype(np.int64)

    args = _make_args(
        preset, mode, seed, seq_len, float(cfg_dict["lr"]),
        float(cfg_dict["dust_lr"]), int(cfg_dict["draws"]),
        float(cfg_dict["gamma"]), float(cfg_dict["sigma"]),
        cfg_dict.get("activation_storage", "fp32"),
        cfg_dict.get("gradient_storage", "fp32"))
    spec = cfg.PRESETS[preset]
    start_rss = rss_kb()
    start = time.perf_counter()
    trainer = TL.Trainer(spec, args, D.ByteStream(data=train_raw))

    params_bytes = int(sum(v.data.nbytes for v in trainer.model.p.values()))
    initial_params = {k: v.data.copy() for k, v in trainer.model.p.items()}
    initial_eval = float(G.eval_loss_on_tokens(trainer.model, eval_tokens))

    # Profile one ordinary backprop graph up front for its cache/gradient
    # bytes.  This is excluded from the per-step timings.
    bp_profile = None
    if mode == "backprop":
        profile_tokens = np.frombuffer(
            train_raw[:seq_len + 1], dtype=np.uint8).astype(np.int64)
        _, _, prof_info = trainer.model.loss_and_backward(
            profile_tokens, profile=True)
        bp_profile = prof_info.get("profile", {})

    grad_norms = []
    last_grad_bytes = 0

    def record_gradient_norm(grads):
        nonlocal last_grad_bytes
        total = 0.0
        nbytes = 0
        for grad in grads.values():
            value = np.asarray(grad)
            nbytes += int(value.nbytes)
            flat = value.reshape(-1)
            if flat.size:
                wide = flat.astype(np.float64)
                total += float(np.dot(wide, wide))
        grad_norms.append(math.sqrt(total))
        last_grad_bytes = nbytes

    # The optimizer sees clipped gradients, so wrap the gradient producer to
    # record the un-clipped global norm used by the trainer's clipping step.
    if mode == "backprop":
        original_bp = trainer.model.loss_and_backward

        def recording_bp(tokens, profile=False):
            loss, grads, info = original_bp(tokens, profile=profile)
            record_gradient_norm(grads)
            return loss, grads, info

        trainer.model.loss_and_backward = recording_bp
    else:
        original_dust = trainer.dust.loss_and_gradients

        def recording_dust(tokens, step=0):
            loss, grads, info = original_dust(tokens, step=step)
            record_gradient_norm(grads)
            return loss, grads, info

        trainer.dust.loss_and_gradients = recording_dust

    history = []
    eval_history = [{"step": 0, "eval_loss": initial_eval}]
    dust_meta = {
        "max_clean_cache_bytes": 0,
        "max_estimator_bytes": 0,
        "max_temp_bytes": 0,
        "max_full_forward_equivalents": 0.0,
        "draws": 0,
        "head_draws": 0,
        "local_draws": 0,
        "scalar_forwards": 0,
    }
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    try:
        for index in range(1, steps + 1):
            step_wall0 = time.perf_counter()
            step_cpu0 = time.process_time()
            trainer.train(
                max_steps=index, log_every=0, ckpt_every=0,
                grad_clip=1.0 if mode == "backprop" else 100.0)
            step_wall = time.perf_counter() - step_wall0
            step_cpu = time.process_time() - step_cpu0
            loss = float(trainer.last_metrics["loss"])
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {index}")
            record = {
                "step": index,
                "loss": loss,
                "smooth_loss": float(trainer.last_metrics.get("smooth_loss", loss)),
                "grad_norm": grad_norms[-1] if grad_norms else None,
                "grad_bytes": int(last_grad_bytes),
                "wall_s": step_wall,
                "cpu_s": step_cpu,
                "rss_kb": rss_kb(),
            }
            dust = trainer.last_metrics.get("dust")
            if dust:
                for key, dst in (("clean_cache_bytes", "max_clean_cache_bytes"),
                                 ("estimator_bytes", "max_estimator_bytes"),
                                 ("temp_bytes", "max_temp_bytes")):
                    dust_meta[dst] = max(dust_meta[dst], int(dust.get(key, 0)))
                dust_meta["max_full_forward_equivalents"] = max(
                    dust_meta["max_full_forward_equivalents"],
                    float(dust.get("full_forward_equivalents", 0.0)))
                dust_meta["draws"] = int(dust.get("draws", dust_meta["draws"]))
                dust_meta["head_draws"] = int(dust.get("head_draws", dust_meta["head_draws"]))
                dust_meta["local_draws"] = int(dust.get("local_draws", dust_meta["local_draws"]))
                dust_meta["scalar_forwards"] = int(
                    dust.get("scalar_forwards", dust_meta["scalar_forwards"]))
                record["dust"] = {
                    "full_forward_equivalents": dust.get("full_forward_equivalents"),
                    "clean_cache_bytes": dust.get("clean_cache_bytes"),
                    "estimator_bytes": dust.get("estimator_bytes"),
                    "temp_bytes": dust.get("temp_bytes"),
                    "draws": dust.get("draws"),
                    "head_draws": dust.get("head_draws"),
                    "local_draws": dust.get("local_draws"),
                    "scalar_forwards": dust.get("scalar_forwards"),
                }
            if eval_every and index % eval_every == 0:
                ev = float(G.eval_loss_on_tokens(trainer.model, eval_tokens))
                record["eval_loss"] = ev
                eval_history.append({"step": index, "eval_loss": ev})
            history.append(record)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as a failed child
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "config": cfg_dict,
            "corpus_sha256": corpus_sha,
            "corpus_bytes": len(raw),
            "train_bytes": len(train_raw),
            "eval_bytes": len(eval_raw),
        }
    wall_s = time.perf_counter() - wall0
    setup_wall_s = wall0 - start
    total_wall_s = time.perf_counter() - start
    cpu_s = time.process_time() - cpu0
    final_eval = float(G.eval_loss_on_tokens(trainer.model, eval_tokens))
    if not eval_history or eval_history[-1]["step"] != steps:
        eval_history.append({"step": steps, "eval_loss": final_eval})
    best_eval = min(item["eval_loss"] for item in eval_history)
    update_sq = 0.0
    for name, value in trainer.model.p.items():
        delta = value.data.astype(np.float64) - initial_params[name].astype(np.float64)
        update_sq += float(np.dot(delta.reshape(-1), delta.reshape(-1)))
    update_norm = math.sqrt(update_sq)
    param_norm = math.sqrt(sum(
        float(np.dot(v.data.astype(np.float64).reshape(-1),
                    v.data.astype(np.float64).reshape(-1)))
        for v in trainer.model.p.values()))
    opt_bytes = _optimizer_nbytes(trainer.opt)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    step_walls = [x["wall_s"] for x in history]
    step_cpus = [x["cpu_s"] for x in history]
    train_wall_s = sum(step_walls)
    eval_wall_s = max(0.0, wall_s - train_wall_s)
    grad_values = [x["grad_norm"] for x in history if x["grad_norm"] is not None]

    if mode == "backprop":
        cache_bytes = int(bp_profile.get("activation_cache", {}).get("bytes", 0)) \
            if bp_profile else 0
        profile_grad_bytes = int(bp_profile.get("gradient_bytes", 0)) \
            if bp_profile else 0
        graph_tensors = int(bp_profile.get("activation_cache", {}).get("tensors", 0)) \
            if bp_profile else 0
        estimator_bytes = 0
        temp_bytes = 0
        full_fwd_eq = None
    else:
        cache_bytes = int(dust_meta["max_clean_cache_bytes"])
        profile_grad_bytes = 0
        graph_tensors = 0
        estimator_bytes = int(dust_meta["max_estimator_bytes"])
        temp_bytes = int(dust_meta["max_temp_bytes"])
        full_fwd_eq = float(dust_meta["max_full_forward_equivalents"])

    result = {
        "status": "ok",
        "config": cfg_dict,
        "corpus": {
            "path": str(corpus_path),
            "sha256": corpus_sha,
            "bytes": len(raw),
            "train_bytes": len(train_raw),
            "eval_bytes": len(eval_raw),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "blas_threads": 1,
        },
        "params_bytes": params_bytes,
        "grad_bytes": int(last_grad_bytes),
        "profile_gradient_bytes": profile_grad_bytes,
        "optimizer_bytes": int(opt_bytes),
        "activation_cache_bytes": int(cache_bytes),
        "estimator_bytes": int(estimator_bytes),
        "temp_bytes": int(temp_bytes),
        "graph_tensors": int(graph_tensors),
        "full_forward_equivalents": full_fwd_eq,
        "start_rss_kb": int(start_rss),
        "peak_rss_kb": peak_rss,
        "start_rss_mb": _rss_mb(start_rss),
        "peak_rss_mb": _rss_mb(peak_rss),
        "setup_wall_s": setup_wall_s,
        "total_wall_s": total_wall_s,
        "wall_s": wall_s,
        "train_wall_s": train_wall_s,
        "eval_wall_s": eval_wall_s,
        "cpu_s": cpu_s,
        "mean_step_wall_s": statistics.mean(step_walls) if step_walls else None,
        "median_step_wall_s": statistics.median(step_walls) if step_walls else None,
        "mean_step_cpu_s": statistics.mean(step_cpus) if step_cpus else None,
        "median_step_cpu_s": statistics.median(step_cpus) if step_cpus else None,
        "tokens_per_s": (steps * (seq_len - 1) / wall_s) if wall_s > 0 else None,
        "tokens_per_s_train": (steps * (seq_len - 1) / train_wall_s)
                              if train_wall_s > 0 else None,
        "steps": steps,
        "seq_len": seq_len,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "best_eval": best_eval,
        "eval_history": eval_history,
        "train_loss_last": history[-1]["loss"] if history else None,
        "train_loss_mean": statistics.mean(x["loss"] for x in history) if history else None,
        "grad_norm_first": grad_values[0] if grad_values else None,
        "grad_norm_last": grad_values[-1] if grad_values else None,
        "grad_norm_median": statistics.median(grad_values) if grad_values else None,
        "grad_norm_max": max(grad_values) if grad_values else None,
        "update_norm": update_norm,
        "param_norm_final": param_norm,
        "nonfinite_count": 0,
        "history": history,
    }
    if mode == "dust":
        result["dust"] = dust_meta
        result["dust_config"] = trainer.dust.config_dict()
    else:
        result["backprop_profile"] = bp_profile
        result["backprop_lr"] = float(args.lr)
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_dust_configs(values):
    out = []
    for item in values:
        try:
            draws_text, lr_text = item.split(":", 1)
            draws_text = draws_text.strip().lower()
            if draws_text.startswith("k"):
                draws_text = draws_text[1:]
            out.append({"draws": int(draws_text), "dust_lr": float(lr_text)})
        except ValueError:
            raise SystemExit(
                f"--dust-configs entries must look like K:LR, got {item!r}")
    return out


def child_command(script, result, out_path):
    cmd = [
        sys.executable, str(script), "--one",
        "--out", str(out_path),
        "--preset", result["preset"],
        "--mode", result["mode"],
        "--seed", str(result["seed"]),
        "--steps", str(result["steps"]),
        "--seq-len", str(result["seq_len"]),
        "--eval-tokens", str(result["eval_tokens"]),
        "--eval-every", str(result["eval_every"]),
        "--corpus", result["corpus"],
        "--lr", str(result["lr"]),
        "--dust-lr", str(result["dust_lr"]),
        "--draws", str(result["draws"]),
        "--gamma", str(result["gamma"]),
        "--sigma", str(result["sigma"]),
        "--activation-storage", result.get("activation_storage", "fp32"),
        "--gradient-storage", result.get("gradient_storage", "fp32"),
        "--label", result["label"],
    ]
    return cmd


def aggregate(runs):
    groups = {}
    for run in runs:
        key = run.get("label") or run.get("config", {}).get("label")
        if not key:
            key = "unknown"
        groups.setdefault(key, []).append(run)
    table = {}
    for label, items in sorted(groups.items()):
        ok = [x for x in items if x.get("status") == "ok"]
        failed = [x for x in items if x.get("status") != "ok"]
        if not ok:
            table[label] = {"failed": len(failed), "errors": [x.get("error") for x in failed]}
            continue
        def col(name):
            return [x.get(name) for x in ok]
        table[label] = {
            "mode": ok[0]["config"]["mode"],
            "draws": ok[0]["config"]["draws"],
            "lr": ok[0]["config"]["dust_lr"] if ok[0]["config"]["mode"] == "dust"
                  else ok[0]["config"]["lr"],
            "seeds": [x["config"]["seed"] for x in ok],
            "failed": len(failed),
            "final_eval": _summary(col("final_eval")),
            "best_eval": _summary(col("best_eval")),
            "initial_eval": _summary(col("initial_eval")),
            "wall_s": _summary(col("wall_s")),
            "train_wall_s": _summary(col("train_wall_s")),
            "eval_wall_s": _summary(col("eval_wall_s")),
            "setup_wall_s": _summary(col("setup_wall_s")),
            "total_wall_s": _summary(col("total_wall_s")),
            "cpu_s": _summary(col("cpu_s")),
            "mean_step_wall_s": _summary(col("mean_step_wall_s")),
            "mean_step_cpu_s": _summary(col("mean_step_cpu_s")),
            "tokens_per_s": _summary(col("tokens_per_s")),
            "peak_rss_mb": _summary(col("peak_rss_mb")),
            "start_rss_mb": _summary(col("start_rss_mb")),
            "params_mb": _summary([_mb(x["params_bytes"]) for x in ok]),
            "grad_mb": _summary([_mb(x["grad_bytes"]) for x in ok]),
            "optimizer_mb": _summary([_mb(x["optimizer_bytes"]) for x in ok]),
            "activation_cache_mb": _summary([_mb(x["activation_cache_bytes"]) for x in ok]),
            "estimator_mb": _summary([_mb(x["estimator_bytes"]) for x in ok]),
            "temp_mb": _summary([_mb(x["temp_bytes"]) for x in ok]),
            "grad_norm_last": _summary(col("grad_norm_last")),
            "update_norm": _summary(col("update_norm")),
            "full_forward_equivalents": _summary(col("full_forward_equivalents")),
        }
    return table


def print_table(table):
    print("\nMatched dense-tiny CPU benchmark")
    print("label | mode | seeds | final eval median [min,max] | best eval | "
          "train s | CPU s | peak RSS MiB | cache MiB | grad MiB | opt MiB | s/step")
    print("---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:")
    for label, row in table.items():
        if "final_eval" not in row:
            print(f"{label} | FAILED | | | | | | | | | |")
            continue
        fe = row["final_eval"]
        be = row["best_eval"]
        wall = row.get("train_wall_s") or {}
        if wall.get("median") is None:
            wall = row["wall_s"]
        cpu = row["cpu_s"]
        rss = row["peak_rss_mb"]
        cache = row["activation_cache_mb"]
        grad = row["grad_mb"]
        opt = row["optimizer_mb"]
        step = row["mean_step_wall_s"]
        def fmt(s, digits=3):
            if s["median"] is None:
                return "-"
            return f"{s['median']:.{digits}f} [{s['min']:.{digits}f},{s['max']:.{digits}f}]"
        print(f"{label} | {row['mode']} | {','.join(map(str,row['seeds']))} | "
              f"{fmt(fe,4)} | {be['median']:.4f} | {wall['median']:.2f} | "
              f"{cpu['median']:.2f} | {rss['median']:.2f} | "
              f"{cache['median']:.2f} | {grad['median']:.3f} | "
              f"{opt['median']:.3f} | {step['median']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="run one child configuration")
    ap.add_argument("--out", default=None, help="child JSON output path")
    ap.add_argument("--label", default=None)
    ap.add_argument("--preset", default="dense-tiny", choices=sorted(cfg.PRESETS))
    ap.add_argument("--mode", default="backprop", choices=("backprop", "dust"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--eval-tokens", type=int, default=8192)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dust-lr", type=float, default=0.05)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--activation-storage", default="fp32")
    ap.add_argument("--gradient-storage", default="fp32")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bp-lrs", type=float, nargs="+", default=[3e-4, 3e-3])
    ap.add_argument("--dust-configs", nargs="+", default=["k2:0.05", "k2:0.02", "k4:0.05"])
    ap.add_argument("--precision-reference", action="store_true",
                    help="also run 10-step BF16 and INT8 backprop references")
    ap.add_argument("--precision-steps", type=int, default=10)
    ap.add_argument("--quick", action="store_true",
                    help="10 steps, seed 0, Dust K=2 only")
    ap.add_argument("--json-out", default=str(ROOT / "benchmarks" / "dust_results.json"))
    ap.add_argument("--resume", action="store_true",
                    help="reuse existing per-run JSON files under --work-dir")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus_path = Path(args.corpus) if args.corpus else default_corpus()
    raw = build_corpus(corpus_path)
    corpus_sha = _sha(raw)
    split = int(len(raw) * 0.8)
    eval_n = min(args.eval_tokens, len(raw) - split)
    if eval_n < 8:
        raise SystemExit("corpus has no usable held-out split")

    if args.one:
        cfg_dict = {
            "preset": args.preset, "mode": args.mode, "seed": args.seed,
            "steps": args.steps, "seq_len": args.seq_len,
            "eval_tokens": eval_n, "eval_every": args.eval_every,
            "corpus": str(corpus_path), "lr": args.lr,
            "dust_lr": args.dust_lr, "draws": args.draws,
            "gamma": args.gamma, "sigma": args.sigma,
            "activation_storage": args.activation_storage,
            "gradient_storage": args.gradient_storage,
            "label": args.label or args.mode,
        }
        result = run_one(cfg_dict)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2))
        else:
            print(json.dumps(result, indent=2))
        return 0 if result.get("status") == "ok" else 1

    if args.quick:
        steps = 10
        seeds = [0]
        dust_specs = [{"draws": 2, "dust_lr": 0.05}]
        bp_lrs = [3e-4]
        precision = False
    else:
        steps = args.steps
        seeds = args.seeds
        dust_specs = parse_dust_configs(args.dust_configs)
        bp_lrs = args.bp_lrs
        precision = args.precision_reference

    work_dir = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="dust_bench_", dir="/tmp/opencode"))
    work_dir.mkdir(parents=True, exist_ok=True)

    runs_spec = []
    for seed in seeds:
        for lr in bp_lrs:
            runs_spec.append({
                "label": f"bp-lr{lr:g}", "preset": args.preset,
                "mode": "backprop", "seed": seed, "steps": steps,
                "seq_len": args.seq_len, "eval_tokens": eval_n,
                "eval_every": args.eval_every, "corpus": str(corpus_path),
                "lr": lr, "dust_lr": 0.05, "draws": 0,
                "gamma": args.gamma, "sigma": args.sigma,
                "activation_storage": args.activation_storage,
                "gradient_storage": args.gradient_storage,
            })
        for spec in dust_specs:
            runs_spec.append({
                "label": f"dust-k{spec['draws']}-lr{spec['dust_lr']:g}",
                "preset": args.preset, "mode": "dust", "seed": seed,
                "steps": steps, "seq_len": args.seq_len,
                "eval_tokens": eval_n, "eval_every": args.eval_every,
                "corpus": str(corpus_path), "lr": 3e-4,
                "dust_lr": spec["dust_lr"], "draws": spec["draws"],
                "gamma": args.gamma, "sigma": args.sigma,
                "activation_storage": "fp32", "gradient_storage": "fp32",
            })
    if precision:
        # Isolate the activation-storage format; parameter gradients and the
        # optimizer remain FP32, matching the existing precision references.
        for storage in ("bf16", "int8"):
            runs_spec.append({
                "label": f"bp-{storage}-act", "preset": args.preset,
                "mode": "backprop", "seed": seeds[0],
                "steps": args.precision_steps, "seq_len": args.seq_len,
                "eval_tokens": eval_n, "eval_every": args.precision_steps,
                "corpus": str(corpus_path), "lr": bp_lrs[-1], "dust_lr": 0.05,
                "draws": 0, "gamma": args.gamma, "sigma": args.sigma,
                "activation_storage": storage, "gradient_storage": "fp32",
            })

    meta = {
        "generated_unix": time.time(),
        "corpus": str(corpus_path), "corpus_sha256": corpus_sha,
        "corpus_bytes": len(raw), "train_bytes": split,
        "eval_bytes": eval_n, "preset": args.preset,
        "steps": steps, "seq_len": args.seq_len,
        "seeds": seeds, "bp_lrs": bp_lrs,
        "dust_configs": dust_specs,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "blas_threads": 1,
        },
        "work_dir": str(work_dir),
    }
    runs = []
    for spec in runs_spec:
        label = spec["label"]
        safe = (label + f"-seed{spec['seed']}-{spec['mode']}"
                + (f"-act{spec['activation_storage']}" if spec["mode"] == "backprop" else ""))
        out_path = work_dir / f"{safe}.json"
        if args.resume and out_path.exists():
            try:
                result = json.loads(out_path.read_text())
                result.setdefault("config", spec)
                runs.append(result)
                print(f"[resume] {safe}", flush=True)
                continue
            except Exception:
                pass
        cmd = child_command(Path(__file__).resolve(), spec, out_path)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        print(f"[run] {safe}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            result = {
                "status": "failed", "config": spec,
                "error": "\n".join(err[-12:]),
                "corpus_sha256": corpus_sha,
            }
            out_path.write_text(json.dumps(result, indent=2))
        else:
            result = json.loads(out_path.read_text())
        runs.append(result)
        print(f"  -> {result.get('status')} in {elapsed:.1f}s", flush=True)

    table = aggregate(runs)
    payload = {"meta": meta, "aggregate": table, "runs": runs}
    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print_table(table)
    print(f"\nWrote {out_json}")
    print(f"Per-run JSON: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
