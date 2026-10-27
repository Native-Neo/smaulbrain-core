"""Automatic dynamic expert growth (canonical two-phase protocol).

Triggers (logical OR, both enabled by default):
  periodic: every --growth-every-steps completed optimizer steps (0 disables).
  loss:     smoothed CE (nats/byte) crosses from above to at/below
            --growth-loss-nats ('none' disables). Edge-triggered with rearm.

Event lifecycle: NO_EVENT -> WAITING_FOR_RANKING -> PREPARED -> VALIDATED ->
COMMITTED. Committed only when the checkpoint with the grown model +
scheduler state is atomically published. (Phase A pending checkpoint, then
Phase B committed checkpoint.)

Quality metric (source selection AND prune ranking): leave-one-out held-out
loss contribution: score(e) = loss_masked(e) - loss_baseline over eval
tokens (>= min_eval_samples, else WAITING_FOR_RANKING/insufficient-evidence).
Deterministic tie-break by expert id. Usage counts are reported but never
the ranking key.
"""
import time
import math

import numpy as np

from . import config as _cfg
from .npgrad import Tensor, cross_entropy

GROWTH_FULL_ADD = 8
GROWTH_TINY_ADD = 2

def _add_count_for(spec):
    return GROWTH_FULL_ADD if spec.name == "moe-full" else GROWTH_TINY_ADD

def _as_np(value):
    if isinstance(value, Tensor):
        value = value.data
    return np.asarray(value, dtype=np.float32)

def _numel(value):
    return int(np.asarray(value.data if isinstance(value, Tensor) else value).size)


class GrowthScheduler:
    """Pure trigger bookkeeping; state mirrors the checkpoint scheduler record."""

    def __init__(self, every_steps=10000, loss_nats=0.5, loss_beta=0.98,
                 min_eval_samples=256):
        self.every_steps = int(every_steps)
        self.loss_threshold = loss_nats  # None disables
        self.loss_beta = float(loss_beta)
        self.min_eval_samples = int(min_eval_samples)
        self.global_step = 0
        self.smooth_loss = None
        self.loss_armed = True
        self.next_periodic_step = every_steps if every_steps and every_steps > 0 else None
        self.event_count = 0
        self.last_growth_step = None
        self.history = []
        self.pending = None

    # -- persistence --
    def to_record(self) -> dict:
        return {
            "global_step": self.global_step,
            "growth_every_steps": self.every_steps,
            "next_periodic_step": self.next_periodic_step,
            "loss_threshold": self.loss_threshold,
            "loss_beta": self.loss_beta,
            "min_eval_samples": self.min_eval_samples,
            "smooth_loss": self.smooth_loss,
            "loss_armed": self.loss_armed,
            "event_count": self.event_count,
            "last_growth_step": self.last_growth_step,
            "history": [dict(h) for h in self.history],
            "pending": dict(self.pending) if self.pending else None,
        }

    @classmethod
    def from_record(cls, rec: dict, min_eval_samples=None):
        # Legacy records predate min_eval_samples; the caller's value applies then.
        fallback = 256 if min_eval_samples is None else int(min_eval_samples)
        s = cls(every_steps=rec.get("growth_every_steps", 10000),
                loss_nats=rec.get("loss_threshold", 0.5),
                loss_beta=rec.get("loss_beta", 0.98),
                min_eval_samples=int(rec.get("min_eval_samples", fallback)))
        s.global_step = int(rec.get("global_step", 0))
        s.smooth_loss = rec.get("smooth_loss")
        s.loss_armed = bool(rec.get("loss_armed", True))
        s.next_periodic_step = rec.get("next_periodic_step")
        s.event_count = int(rec.get("event_count", 0))
        s.last_growth_step = rec.get("last_growth_step")
        s.history = [dict(h) for h in rec.get("history", [])]
        s.pending = dict(rec["pending"]) if rec.get("pending") else None
        return s

    # -- per-step trigger evaluation (call AFTER incrementing global_step) --
    def note_optimizer_step(self, loss_nats):
        """Update smoothing + evaluate triggers. Returns reason or None."""
        self.global_step += 1
        if loss_nats is not None and math.isfinite(loss_nats):
            b = self.loss_beta
            self.smooth_loss = (float(loss_nats) if self.smooth_loss is None
                                else b * self.smooth_loss + (1 - b) * float(loss_nats))
        step_due = (self.every_steps > 0 and self.next_periodic_step is not None
                    and self.global_step >= self.next_periodic_step)
        loss_due = False
        if (self.loss_threshold is not None and self.loss_armed
                and self.smooth_loss is not None and math.isfinite(self.smooth_loss)):
            if self.smooth_loss <= float(self.loss_threshold):
                loss_due = True
        # rearm: loss strictly above threshold re-arms for a future crossing
        if (self.loss_threshold is not None and not self.loss_armed
                and self.smooth_loss is not None and math.isfinite(self.smooth_loss)
                and self.smooth_loss > float(self.loss_threshold)):
            self.loss_armed = True
        if step_due and loss_due:
            return "both"
        if step_due:
            return "step"
        if loss_due:
            return "loss"
        return None

    def create_pending(self, reason: str, smooth_snapshot=None) -> dict:
        eid = (self.history[-1]["id"] + 1) if self.history else 1
        if self.pending is not None:
            raise RuntimeError("growth event already pending")
        self.pending = {
            "id": eid,
            "step": self.global_step,
            "reason": reason,
            "loss_value": self.smooth_loss if smooth_snapshot is None else smooth_snapshot,
            "interval": self.every_steps,
            "threshold": self.loss_threshold,
            "armed_before": self.loss_armed,
            "ranking": None,
            "sources": None,
            "mutation": None,
            "status": "WAITING_FOR_RANKING",
        }
        return self.pending

    def commit_event(self, entry: dict) -> None:
        if any(h["id"] == entry["id"] for h in self.history):
            raise ValueError(f"duplicate committed event id {entry['id']}")
        self.history.append(entry)
        self.event_count += 1
        self.last_growth_step = entry["step"]
        self.pending = None
        # advance periodic schedule past the event step (exactly-once advancement)
        if self.every_steps > 0:
            base = entry["step"]
            k = base // self.every_steps + 1
            self.next_periodic_step = k * self.every_steps
        # disarm loss trigger only if it participated
        if entry["reason"] in ("loss", "both"):
            self.loss_armed = False


# ---------------- quality ranking ----------------

def eval_loss_on_tokens(model, tokens) -> float:
    logits, _ = model.forward(tokens, train=False)
    tgt = np.asarray(tokens[1:], dtype=np.int64)
    return float(cross_entropy(logits[:-1], tgt).item())


def rank_experts(model, eval_batches, min_samples: int, max_batches=None):
    total = sum(max(len(b) - 1, 0) for b in eval_batches)
    if total < min_samples: return None, total
    batches = list(eval_batches if max_batches is None else eval_batches[:max_batches])
    if not batches: return None, total
    base = [eval_loss_on_tokens(model, b) for b in batches]
    base_mean = sum(base) / len(base)
    from . import model as _m
    scores = {}
    for eid in _m.expert_ids(model.p):
        masked = [eval_loss_on_tokens_disabled(model, b, eid) for b in batches]
        scores[eid] = float(sum(masked) / len(masked) - base_mean)
    return scores, total


def eval_loss_on_tokens_disabled(model, tokens, disabled):
    logits, _ = model.forward(tokens, train=False, disabled=(disabled,))
    tgt = np.asarray(tokens[1:], dtype=np.int64)
    return float(cross_entropy(logits[:-1], tgt).item())


def prepare_sources(scores: dict, n_needed: int):
    if any(not math.isfinite(float(s)) for s in scores.values()):
        return None  # unusable evidence anywhere: ranking cannot be trusted
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(order) < n_needed:
        return None
    return [eid for eid, _ in order[:n_needed]]


def clone_with_mutation(params: dict, sources: list, new_ids: list, noise_scale: float, seed: int, n_exact_unperturbed: int):
    rng = np.random.default_rng(int(seed)); detail = []
    for rank, (src, new) in enumerate(zip(sources, new_ids)):
        for suffix in ("Win", "Wout"):
            skey, nkey = f"expert.{src}.{suffix}", f"expert.{new}.{suffix}"; W = _as_np(params[skey])
            if rank < n_exact_unperturbed:
                params[nkey] = W.copy()
                detail.append({"expert": new, "source": src, "tensor": suffix, "mutated": False})
            else:
                rms = float(np.sqrt(np.mean(W.astype(np.float64) ** 2)))
                sigma = noise_scale * (rms if rms > 0 else 1.0)
                noise = rng.standard_normal(W.shape, dtype=np.float32)
                params[nkey] = W + noise * np.float32(sigma)
                detail.append({"expert": new, "source": src, "tensor": suffix, "mutated": True, "sigma": sigma})
    return detail


def grow_candidate(params: dict, router_W, sources: list, new_ids: list, noise_scale: float, seed: int,
                   n_exact_unperturbed: int, fail_at=None):
    if fail_at == "during_clone": raise RuntimeError("injected failure during_clone")
    # Unchanged tensors are shared, not copied: nothing below mutates them in place, and cloning would double model RAM here.
    cand = {k: _as_np(v).copy() if k in ("router",) else _as_np(v) for k, v in params.items()}
    router_W = _as_np(router_W)
    if fail_at == "during_mutate": raise RuntimeError("injected failure during_mutate")
    detail = clone_with_mutation(cand, sources, new_ids, noise_scale, seed, n_exact_unperturbed)
    rng = np.random.default_rng(int(seed) + 999); old_e = router_W.shape[1]; new_e = old_e + len(new_ids)
    router = np.empty((router_W.shape[0], new_e), dtype=np.float32)
    router[:, :old_e] = router_W
    for i, src in enumerate(sources):
        col = router_W[:, src]
        rms = float(np.sqrt(np.mean(col.astype(np.float64) ** 2)))
        noise = rng.standard_normal(col.shape, dtype=np.float32)
        router[:, old_e + i] = col + noise * np.float32(0.01 * (rms or 1.0))
    cand["router"] = router
    return cand, detail


def validate_candidate(spec,cand_params,opt_names) -> None:
    eids=sorted(int(k.split(".")[1]) for k in cand_params if k.endswith(".Win"))
    if eids != list(range(len(eids))): raise ValueError("candidate expert registry not dense")
    for eid in eids:
        n=_numel(cand_params[f"expert.{eid}.Win"])+_numel(cand_params[f"expert.{eid}.Wout"])
        if n != _cfg.EXPECTED_EXPERT_PARAMS[spec.name]: raise ValueError(f"candidate expert {eid} has {n} params")
    if _as_np(cand_params["router"]).shape[1] != len(eids): raise ValueError("candidate router width disagrees with registry")
    if not all(np.isfinite(_as_np(v)).all() for v in cand_params.values()): raise ValueError("candidate contains non-finite values")
