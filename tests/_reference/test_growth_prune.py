"""Growth invariants, resume regressions, pruning, and paging-equivalence tests.

Covers the audit claims for smaulbrain/{growth,prune_cmd,train_loop}.py
with moe-tiny / dense-tiny presets, seq_len 32, and few steps per case.
"""
import argparse
import copy

import pytest
import torch

from smaulbrain import checkpoint as CK
from smaulbrain import cli
from smaulbrain import config as cfg
from smaulbrain import data as D
from smaulbrain import growth as G
from smaulbrain import model as M
from smaulbrain import prune_cmd as P
from smaulbrain import smaulopt as OPT
from smaulbrain import train_loop as TL

SEQ_LEN = 32
CORPUS = 1 << 13


# ---------------------------------------------------------------- helpers
def make_args(checkpoint, preset="moe-tiny", extra=()):
    ap = argparse.ArgumentParser()
    cli.add_common_args(ap, need_train=True)
    argv = ["--preset", preset, "--checkpoint", str(checkpoint), "--seq-len", str(SEQ_LEN),
            "--growth-every-steps", "0", "--growth-loss-nats", "none",
            "--growth-min-eval-samples", "16", "--mtp-depth", "0", "--seed", "7"]
    argv += list(extra)
    args = ap.parse_args(argv)
    cli.validate_args(args)
    return args


def corpus():
    return D.synth_bytes(CORPUS, seed=3)


def build_trainer(checkpoint, preset="moe-tiny", extra=(), eval_stream=None):
    return TL.Trainer(cfg.PRESETS[preset], make_args(checkpoint, preset, extra),
                      D.ByteStream(data=corpus()), eval_stream=eval_stream)


def tensors(p):
    return {k: v.detach().clone() for k, v in p.items()}


def params_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def opt_snapshot(opt):
    return {k: {sk: (sv.detach().clone() if isinstance(sv, torch.Tensor) else sv)
                for sk, sv in state.items()} for k, state in opt.state.items()}


def losses(trainer, n):
    return [e["loss"] for e in trainer.log if "loss" in e][-n:]


def eval_tokens(n_batches=1, seed=101):
    return [list(D.synth_bytes(SEQ_LEN + 1, seed=seed + i)) for i in range(n_batches)]


class _SnapTrainer(TL.Trainer):
    """Captures pre-growth params/optimizer state for exactness invariants."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pre_growth_p = None
        self.pre_growth_opt = None

    def _resolve_pending(self, fail_at=None):
        if self.pre_growth_p is None and self.sched.pending is not None:
            self.pre_growth_p = tensors(self.model.p)
            self.pre_growth_opt = opt_snapshot(self.opt)
        return super()._resolve_pending(fail_at=fail_at)


# ---------------------------------------------------------------- growth
def test_growth_event_router_resize_and_candidate_repro(tmp_path):
    ck = tmp_path / "run"
    t = _SnapTrainer(cfg.PRESETS["moe-tiny"],
                     make_args(ck, extra=["--growth-every-steps", "1", "--growth-seed", "123"]),
                     D.ByteStream(data=corpus()))
    t.train(max_steps=1, log_every=1)
    assert t.pre_growth_p is not None and t.pre_growth_opt is not None

    entry = t.sched.history[0]
    assert entry["id"] == 1 and entry["step"] == 1 and entry["reason"] == "step"
    assert entry["prev_count"] == 32 and entry["new_count"] == 34
    assert entry["new_ids"] == [32, 33]
    src = entry["sources"]
    assert len(src) == 2 and len(set(src)) == 2 and all(0 <= s < 32 for s in src)
    assert t.sched.pending is None and t.sched.event_count == 1

    # registry is dense, router grew, old parameters are untouched
    assert M.expert_ids(t.model.p) == list(range(34))
    assert t.model.p["router"].shape[1] == 34
    for e in range(32):
        for suf in ("Win", "Wout"):
            assert torch.equal(t.model.p[f"expert.{e}.{suf}"], t.pre_growth_p[f"expert.{e}.{suf}"])

    # resized router optimizer state == old columns preserved, new columns zero
    old, new = t.pre_growth_opt, t.opt.state
    assert torch.equal(new["router"]["m"][:, :32], old["router"]["m"])
    assert bool((new["router"]["m"][:, 32:] == 0).all())
    assert torch.equal(new["router"]["R"], old["router"]["R"])
    assert torch.equal(new["router"]["C"][:32], old["router"]["C"])
    assert bool((new["router"]["C"][32:] == 0).all())
    # old expert moments preserved; new experts start from a fresh zero state
    for e in range(32):
        for suf in ("Win", "Wout"):
            assert torch.equal(new[f"expert.{e}.{suf}"]["m"], old[f"expert.{e}.{suf}"]["m"])
    for e in (32, 33):
        for suf in ("Win", "Wout"):
            st = new[f"expert.{e}.{suf}"]
            assert bool((st["m"] == 0).all()) and bool((st["R"] == 0).all()) and bool((st["C"] == 0).all())

    # the resize is exactly hand-padding the pre-growth state
    ref = OPT.SmaulOPT(lr=t.opt.lr, beta1=t.opt.beta1, beta2=t.opt.beta2,
                       eps=t.opt.eps, weight_decay=t.opt.weight_decay)
    ref_state = {}
    for k, s in old.items():
        if k == "router":
            m = torch.zeros_like(t.model.p["router"])
            m[:, :32] = s["m"]
            C = torch.zeros(t.model.p["router"].shape[1])
            C[:32] = s["C"]
            ref_state[k] = {"m": m, "R": s["R"].clone(), "C": C}
        else:
            ref_state[k] = {sk: (sv.clone() if isinstance(sv, torch.Tensor) else sv)
                            for sk, sv in s.items()}
    ref.load_state_dict(ref_state, {k: ("R" in ref_state[k]) for k in ref_state})
    for k, s in ref.state.items():
        for sk, sv in s.items():
            if isinstance(sv, torch.Tensor):
                assert torch.equal(new[k][sk], sv), (k, sk)

    # the committed candidate is exactly the recorded deterministic transform
    mut = entry["mutation"]
    cand, detail = G.grow_candidate(t.pre_growth_p, t.pre_growth_p["router"], src,
                                    entry["new_ids"], mut["scale"], mut["seed"],
                                    mut["n_exact_unperturbed"])
    assert params_equal(cand, tensors(t.model.p))
    assert all(d["mutated"] for d in detail)  # moe-tiny clones nothing unperturbed


def test_periodic_growth_fires_exactly_on_schedule(tmp_path):
    ck = tmp_path / "run"
    t = build_trainer(ck, extra=["--growth-every-steps", "2"])
    t.train(max_steps=4, log_every=1)
    assert [h["id"] for h in t.sched.history] == [1, 2]
    assert [h["step"] for h in t.sched.history] == [2, 4]
    assert t.sched.next_periodic_step == 6
    assert M.expert_ids(t.model.p) == list(range(36))


def test_growth_blocked_when_evidence_insufficient(tmp_path):
    ck = tmp_path / "run"
    t = build_trainer(ck, extra=["--growth-every-steps", "1", "--growth-min-eval-samples", "256"])
    with pytest.raises(RuntimeError, match="growth blocked"):
        t.train(max_steps=1, log_every=1)

    man = CK.load_checkpoint(ck)["manifest"]
    pend = man["scheduler"]["pending"]
    assert pend is not None and pend["status"] == "WAITING_FOR_RANKING"
    assert pend["ranking"]["insufficient"] is True and pend["ranking"]["samples"] < 256
    assert man["scheduler"]["history"] == []


def test_dense_growth_trigger_is_skipped(tmp_path):
    ck = tmp_path / "run"
    t = build_trainer(ck, preset="dense-tiny", extra=["--growth-every-steps", "1"])
    t.train(max_steps=2, log_every=1)
    skipped = [e for e in t.log if e.get("event") == "growth-skipped-dense"]
    assert [e["step"] for e in skipped] == [1, 2]
    assert t.sched.pending is None and t.sched.history == []


# ---------------------------------------------------------------- resume regressions
def test_resume_restores_eval_stream_cursor_and_continuation(tmp_path):
    db = corpus()
    ck = tmp_path / "run"
    t1 = build_trainer(ck, extra=["--growth-every-steps", "1"],
                       eval_stream=D.ByteStream(data=db))
    t1.train(max_steps=3, log_every=1, ckpt_every=3)
    assert len(t1.sched.history) == 3
    assert t1.eval_stream.cursor > 0
    assert CK.load_checkpoint(ck)["manifest"]["config"]["eval_cursor"] == t1.eval_stream.cursor

    t2 = build_trainer(ck, extra=["--growth-every-steps", "1"],
                       eval_stream=D.ByteStream(data=db))
    t2.load(ck)
    assert t2.eval_stream.cursor == t1.eval_stream.cursor

    t1.train(max_steps=5, log_every=1)
    t2.train(max_steps=5, log_every=1)
    assert losses(t1, 2) == losses(t2, 2)
    assert params_equal(tensors(t1.model.p), tensors(t2.model.p))
    assert [h["id"] for h in t1.sched.history] == [h["id"] for h in t2.sched.history]


def test_d2r_trainers_use_isolated_mirrors(tmp_path):
    ck = tmp_path / "run"
    t1 = build_trainer(ck, extra=["--paging", "D2R"])
    t1.train(max_steps=3, log_every=1, ckpt_every=3)
    p1 = tensors(t1.model.p)

    t2 = build_trainer(ck, extra=["--paging", "D2R"])
    t2.load(ck)
    assert params_equal(p1, tensors(t2.model.p))

    t1.train(max_steps=5, log_every=1)
    t2.train(max_steps=5, log_every=1)
    assert losses(t1, 2) == losses(t2, 2)
    assert params_equal(tensors(t1.model.p), tensors(t2.model.p))
    # each trainer owns its own disk mirror: no cross-instance corruption
    assert t1.model.store.disk_dir != t2.model.store.disk_dir


# ---------------------------------------------------------------- paging
def test_paging_r_and_d2r_train_identically(tmp_path):
    tR = build_trainer(tmp_path / "r", extra=["--paging", "R"])
    tD = build_trainer(tmp_path / "d", extra=["--paging", "D2R"])
    tR.train(max_steps=3, log_every=1)
    tD.train(max_steps=3, log_every=1)
    assert losses(tR, 3) == losses(tD, 3)
    assert params_equal(tensors(tR.model.p), tensors(tD.model.p))


# ---------------------------------------------------------------- pruning
def grown_checkpoint(tmp_path, steps=2):
    ck = tmp_path / "run"
    t = build_trainer(ck, extra=["--growth-every-steps", "1"])
    t.train(max_steps=steps, log_every=1, ckpt_every=steps)
    return ck


def test_rank_for_prune_guards():
    with pytest.raises(ValueError, match="must be positive"):
        P.rank_for_prune({}, 0, 2, 4)
    with pytest.raises(ValueError, match="cannot remove"):
        P.rank_for_prune({}, 4, 2, 4)
    with pytest.raises(ValueError, match="top_k"):
        P.rank_for_prune({}, 3, 2, 4)
    # non-finite evidence must refuse pruning, not rank by sort accident
    with pytest.raises(ValueError, match="non-finite"):
        P.rank_for_prune({0: float("nan")}, 1, 1, 1)
    with pytest.raises(ValueError, match="non-finite"):
        P.rank_for_prune({0: 0.0, 1: float("inf")}, 1, 1, 2)
    # ascending scores, deterministic tie-break by expert id
    scores = {0: -1.0, 1: 0.5, 2: 0.5, 3: 2.0}
    assert P.rank_for_prune(scores, 2, 1, 4) == [0, 1]
    assert P.rank_for_prune(scores, 1, 1, 4) == [0]


def test_prune_dry_run_requires_consent_and_evidence(tmp_path):
    ck = grown_checkpoint(tmp_path)
    good = CK._sha256(str(ck / "params.pt"))
    batches = eval_tokens()

    report = P.prune_checkpoint(str(ck), 1, batches, 16, dry_run=True)
    assert report["candidates"] and report["prev_count"] == 36 and report["new_count"] == 35
    assert CK._sha256(str(ck / "params.pt")) == good
    assert CK.load_checkpoint(ck)["manifest"].get("prune_history") is None

    with pytest.raises(RuntimeError, match="refusing destructive replacement"):
        P.prune_checkpoint(str(ck), 1, batches, 16)
    assert CK._sha256(str(ck / "params.pt")) == good

    with pytest.raises(RuntimeError, match="pruning refused"):
        P.prune_checkpoint(str(ck), 1, batches, 10**9, assume_yes=True)
    assert CK._sha256(str(ck / "params.pt")) == good

    dense = tmp_path / "dense"
    t = build_trainer(dense, preset="dense-tiny")
    t.train(max_steps=1, log_every=1, ckpt_every=1)
    with pytest.raises(ValueError, match="MoE checkpoints only"):
        P.prune_checkpoint(str(dense), 1, batches, 16, assume_yes=True)


def test_prune_exact_remap_and_metadata(tmp_path):
    ck = grown_checkpoint(tmp_path)
    blob = CK.load_checkpoint(ck)
    params, opt0, man = blob["params"], blob["opt_state"], blob["manifest"]
    n = man["n_experts_current"]
    assert n == 36
    batches = eval_tokens()

    report = P.prune_checkpoint(str(ck), 1, batches, 16, assume_yes=True)
    victim = report["candidates"][0]
    assert report["prev_count"] == n and report["new_count"] == n - 1
    assert report["samples"] >= 16

    # the reported ranking is reproducible from the pre-prune checkpoint
    m = M.Model(cfg.PRESETS["moe-tiny"], params, mtp_depth=0)
    scores, nsamp = G.rank_experts(m, batches, 16)
    assert nsamp == report["samples"]
    assert victim == sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))[0][0]

    blob2 = CK.load_checkpoint(ck)
    p2, o2, man2 = blob2["params"], blob2["opt_state"], blob2["manifest"]
    keep = [e for e in range(n) if e != victim]
    assert M.expert_ids(p2) == list(range(n - 1))
    for new, old in enumerate(keep):
        for suf in ("Win", "Wout"):
            assert torch.equal(p2[f"expert.{new}.{suf}"], params[f"expert.{old}.{suf}"])
    assert torch.equal(p2["router"], params["router"][:, keep])

    # optimizer moments follow the surviving experts; router state is re-initialized
    assert "router" not in o2
    assert len(o2) == len(opt0) - 3  # dropped router + one removed expert (Win, Wout)
    for new, old in enumerate(keep):
        for suf in ("Win", "Wout"):
            for sk in ("m", "R", "C"):
                assert torch.equal(o2[f"expert.{new}.{suf}"][sk], opt0[f"expert.{old}.{suf}"][sk])

    assert man2["n_experts_current"] == n - 1
    assert man2["expert_registry"] == list(range(n - 1))
    assert man2["arch"] == cfg.arch_report(cfg.PRESETS["moe-tiny"], n - 1, mtp_depth=0)
    hist = man2["prune_history"][-1]
    assert hist["removed"] == [victim] and hist["prev_count"] == n and hist["new_count"] == n - 1

    # a pruned checkpoint resumes and trains
    t = build_trainer(ck)
    t.load(ck)
    assert M.expert_ids(t.model.p) == list(range(n - 1))
    t.train(max_steps=t.sched.global_step + 1, log_every=1)
    assert losses(t, 1)
