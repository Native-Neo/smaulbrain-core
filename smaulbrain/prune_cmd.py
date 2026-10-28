"""Evidence-based expert pruning with atomic checkpoint replacement."""
import math

import numpy as np

from . import checkpoint as _ckpt
from . import growth as _growth
from . import model as _model
from .npgrad import Tensor


def rank_for_prune(scores: dict, n_remove: int, top_k: int, n_experts: int):
    for eid, score in scores.items():
        if not math.isfinite(float(score)):
            raise ValueError(f"non-finite quality score for expert {eid}; "
                             "pruning evidence is unusable")
    if n_remove <= 0:
        raise ValueError("--rm-worst must be positive")
    if n_remove >= n_experts:
        raise ValueError(f"cannot remove {n_remove} of {n_experts} experts "
                         f"(routing requires top_k={top_k} <= remaining)")
    if n_experts - n_remove < top_k:
        raise ValueError(f"removal would leave {n_experts - n_remove} experts, "
                         f"violating top_k={top_k}")
    order = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
    return [eid for eid, _ in order[:n_remove]]


def prune_checkpoint(ckpt_dir: str, rm_worst: int, eval_batches, min_samples: int,
                     dry_run=False, assume_yes=False):
    blob = _ckpt.load_checkpoint(ckpt_dir)
    man = blob["manifest"]
    if not man["preset"].startswith("moe"):
        raise ValueError("pruning applies to MoE checkpoints only")
    from . import config as _cfg
    spec = _cfg.PRESETS[man["preset"]]
    params = blob["params"]
    eids = _model.expert_ids(params)
    mtp_depth = int(man.get("config", {}).get("mtp_depth", 0))
    # quality evidence from current model
    m = _model.Model(spec, params, forward_quant=man["config"].get("forward_quant", "fp32"),
                     backward_mode=man["config"].get("backward_mode", "fp32"),
                     mtp_depth=mtp_depth)
    scores, nsamp = _growth.rank_experts(m, eval_batches, min_samples)
    if scores is None:
        raise RuntimeError(f"insufficient quality evidence ({nsamp} < {min_samples}); "
                           f"pruning refused")
    victims = rank_for_prune(scores, rm_worst, spec.top_k, len(eids))
    report = {"candidates": victims,
              "scores": {str(v): scores[v] for v in victims},
              "samples": nsamp,
              "prev_count": len(eids), "new_count": len(eids) - len(victims)}
    if dry_run:
        return report
    if not assume_yes:
        raise RuntimeError(f"refusing destructive replacement without --yes; "
                           f"dry-run report: {report}")
    removed = set(victims)
    keep = [e for e in eids if e not in removed]
    remap = {old: new for new, old in enumerate(keep)}
    new_params = {k: v for k, v in params.items() if not k.startswith("expert.")}
    for old in keep:
        for suffix in ("Win", "Wout"):
            new_params[f"expert.{remap[old]}.{suffix}"] = params[f"expert.{old}.{suffix}"]
    R = params["router"]
    R = R.data if isinstance(R, Tensor) else np.asarray(R)
    new_params["router"] = R[:, keep].copy()
    opt_state = blob["opt_state"]
    new_opt = {k: v for k, v in opt_state.items()
               if not (k.startswith("expert.") and int(k.split(".")[1]) in removed)}
    renamed = {}
    for k, v in new_opt.items():
        if k.startswith("expert."):
            parts = k.split(".")
            renamed[f"expert.{remap[int(parts[1])]}.{'.'.join(parts[2:])}"] = v
        else:
            renamed[k] = v
    man2 = dict(man)
    man2["n_experts_current"] = len(keep)
    man2["expert_registry"] = list(range(len(keep)))
    man2["arch"] = _cfg.arch_report(spec, len(keep), mtp_depth=mtp_depth)
    man2.setdefault("prune_history", []).append(
        {"removed": victims, "scores": {str(v): scores[v] for v in victims},
         "prev_count": len(eids), "new_count": len(keep)})
    renamed.pop("router", None)  # shape-bound; re-initialized on load
    _ckpt.save_checkpoint(ckpt_dir, man2, new_params, renamed)
    # Release the first copy (params + optimizer moments) before re-reading the written checkpoint.
    del blob, params, m, R, opt_state, new_params, new_opt, renamed
    blob2 = _ckpt.load_checkpoint(ckpt_dir)
    m2 = _model.Model(spec, blob2["params"], mtp_depth=mtp_depth)
    logits, _ = m2.forward([65, 66, 67, 68], train=False)
    logits = logits.data if isinstance(logits, Tensor) else np.asarray(logits)
    if not np.isfinite(logits).all():
        raise RuntimeError("pruned model failed validation")
    return report
