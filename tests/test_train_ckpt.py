"""Checkpoint, resume, crash-recovery, paging-store, data, and CLI/config tests.

Covers the audit claims for smaulbrain/{checkpoint,train_loop,paging,data,cli,config}.py
with dense-tiny / moe-tiny presets, seq_len 32, and 2-4 steps per case.
"""
import argparse
import copy
import json
import os

import numpy as np
import pytest
import torch

from smaulbrain import checkpoint as CK
from smaulbrain import cli
from smaulbrain import config as cfg
from smaulbrain import data as D
from smaulbrain import model as M
from smaulbrain import paging as PG
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


def opt_diff(a, b):
    """Return a description of the first difference, or None if bitwise equal."""
    if a.keys() != b.keys():
        return f"state keys differ: {sorted(set(a) ^ set(b))}"
    for k in a:
        for sk in a[k]:
            x, y = a[k][sk], b[k][sk]
            if isinstance(x, torch.Tensor):
                if not torch.equal(x, y):
                    return f"{k}.{sk} differs"
            elif x != y:
                return f"{k}.{sk}: {x!r} != {y!r}"
    return None


def losses(trainer, n):
    return [e["loss"] for e in trainer.log if "loss" in e][-n:]


# ---------------------------------------------------------------- exact resume
def test_resume_exact_moe_tiny(tmp_path):
    ck = tmp_path / "run"
    t1 = build_trainer(ck)
    t1.train(max_steps=3, log_every=1, ckpt_every=3)
    p1, o1, s1 = tensors(t1.model.p), opt_snapshot(t1.opt), copy.deepcopy(t1.sched.to_record())

    t2 = build_trainer(ck)
    t2.load(ck)
    assert params_equal(p1, t2.model.p)
    assert opt_diff(o1, opt_snapshot(t2.opt)) is None
    assert t2.sched.to_record() == s1
    assert t2.stream.cursor == t1.stream.cursor

    t1.train(max_steps=5, log_every=1)
    t2.train(max_steps=5, log_every=1)
    assert losses(t1, 2) == losses(t2, 2)
    assert params_equal(tensors(t1.model.p), tensors(t2.model.p))
    assert opt_diff(opt_snapshot(t1.opt), opt_snapshot(t2.opt)) is None
    assert t1.sched.to_record() == t2.sched.to_record()


def test_resume_exact_dense_tiny(tmp_path):
    ck = tmp_path / "run"
    t1 = build_trainer(ck, preset="dense-tiny")
    t1.train(max_steps=3, log_every=1, ckpt_every=3)
    p1, o1, s1 = tensors(t1.model.p), opt_snapshot(t1.opt), copy.deepcopy(t1.sched.to_record())

    t2 = build_trainer(ck, preset="dense-tiny")
    t2.load(ck)
    assert params_equal(p1, t2.model.p)
    assert opt_diff(o1, opt_snapshot(t2.opt)) is None
    assert t2.sched.to_record() == s1

    t1.train(max_steps=5, log_every=1)
    t2.train(max_steps=5, log_every=1)
    assert losses(t1, 2) == losses(t2, 2)
    assert params_equal(tensors(t1.model.p), tensors(t2.model.p))


def test_resume_exact_mtp_depth_2(tmp_path):
    ck = tmp_path / "run"
    t1 = build_trainer(ck, extra=["--mtp-depth", "2", "--mtp-loss-weight", "0.25"])
    t1.train(max_steps=3, log_every=1, ckpt_every=3)
    assert any(k.startswith("mtp.") for k in t1.model.p)
    p1, o1 = tensors(t1.model.p), opt_snapshot(t1.opt)

    t2 = build_trainer(ck, extra=["--mtp-depth", "2", "--mtp-loss-weight", "0.25"])
    t2.load(ck)
    assert t2.model.mtp_depth == 2 and t2.model.mtp_loss_weight == 0.25
    assert params_equal(p1, t2.model.p)
    assert opt_diff(o1, opt_snapshot(t2.opt)) is None

    t1.train(max_steps=5, log_every=1)
    t2.train(max_steps=5, log_every=1)
    assert losses(t1, 2) == losses(t2, 2)
    assert params_equal(tensors(t1.model.p), tensors(t2.model.p))


def test_resume_pre_mtp_checkpoint_adopts_checkpoint_depth(tmp_path):
    """A depth-0 checkpoint resumed with --mtp-depth 3 must adopt depth 0."""
    ck = tmp_path / "run"
    t1 = build_trainer(ck, extra=["--mtp-depth", "0"])
    t1.train(max_steps=3, log_every=1, ckpt_every=3)

    t2 = build_trainer(ck, extra=["--mtp-depth", "3"])
    t2.load(ck)
    assert t2.model.mtp_depth == 0
    assert not any(k.startswith("mtp.") for k in t2.model.p)
    assert [e for e in t2.log if e.get("event") == "mtp-config-adopted"]

    t1.train(max_steps=5, log_every=1)
    t2.train(max_steps=5, log_every=1)
    assert losses(t1, 2) == losses(t2, 2)
    assert params_equal(tensors(t1.model.p), tensors(t2.model.p))


def test_resume_deeper_checkpoint_overrides_requested_depth(tmp_path):
    """The checkpoint owns the architecture: depth 2 wins over --mtp-depth 0."""
    ck = tmp_path / "run"
    t1 = build_trainer(ck, extra=["--mtp-depth", "2", "--mtp-loss-weight", "0.5"])
    t1.train(max_steps=3, log_every=1, ckpt_every=3)

    t2 = build_trainer(ck, extra=["--mtp-depth", "0"])
    t2.load(ck)
    assert t2.model.mtp_depth == 2
    assert t2.model.mtp_loss_weight == 0.5

    t1.train(max_steps=5, log_every=1)
    t2.train(max_steps=5, log_every=1)
    assert losses(t1, 2) == losses(t2, 2)
    assert params_equal(tensors(t1.model.p), tensors(t2.model.p))


# ---------------------------------------------------------------- crash recovery
CRASH_POINTS = ["before_pending", "during_pending", "after_pending", "before_candidate",
                "during_clone", "during_mutate", "after_validate", "during_optinit",
                "during_commit", "after_commit", "after_step"]


@pytest.fixture(scope="module")
def crash_reference(tmp_path_factory):
    """Uninterrupted run: step 1 triggers one growth event, step 2 continues."""
    ck = tmp_path_factory.mktemp("crash-ref") / "run"
    t = build_trainer(ck, extra=["--growth-every-steps", "1"])
    t.train(max_steps=2, log_every=1, ckpt_every=1)
    return {"losses": losses(t, 2), "params": tensors(t.model.p),
            "ids": [h["id"] for h in t.sched.history],
            "n_experts": len(M.expert_ids(t.model.p))}


@pytest.mark.parametrize("fail_at", CRASH_POINTS)
def test_crash_injection_recovers_exactly(tmp_path, fail_at, crash_reference):
    ck = tmp_path / "run"
    t = build_trainer(ck, extra=["--growth-every-steps", "1"])
    kwargs = {"fail_at": fail_at} if fail_at == "after_step" else {"growth_fail_at": fail_at}
    # _Injected covers most points; grow_candidate's clone/mutate points raise RuntimeError
    with pytest.raises(RuntimeError):
        t.train(max_steps=1, log_every=1, ckpt_every=1, **kwargs)

    # on-disk checkpoint must be loadable, checksummed and self-consistent
    blob = CK.load_checkpoint(ck)
    man, params = blob["manifest"], blob["params"]
    ids = [h["id"] for h in man["scheduler"]["history"]]
    assert len(ids) == len(set(ids))
    assert man["n_experts_current"] == len(M.expert_ids(params))
    assert M.expert_ids(params) == list(range(man["n_experts_current"]))
    pending = man["scheduler"]["pending"]
    assert pending is None or pending["id"] not in ids

    # resume: the pending event (if any) resolves exactly once
    t2 = build_trainer(ck, extra=["--growth-every-steps", "1"])
    t2.load(ck)
    t2.train(max_steps=2, log_every=1, ckpt_every=1)

    assert losses(t2, 1) == [crash_reference["losses"][1]]
    assert params_equal(tensors(t2.model.p), crash_reference["params"])
    assert [h["id"] for h in t2.sched.history] == crash_reference["ids"]
    assert len(M.expert_ids(t2.model.p)) == crash_reference["n_experts"]
    man2 = CK.load_checkpoint(ck)["manifest"]
    ids2 = [h["id"] for h in man2["scheduler"]["history"]]
    assert len(ids2) == len(set(ids2))
    assert man2["n_experts_current"] == crash_reference["n_experts"]


def test_crash_before_commit_reruns_growth_deterministically(tmp_path, crash_reference):
    """A crash between commit_event and the Phase B publish (during_commit) must
    re-resolve the same event with the same sources and mutation seed."""
    ck = tmp_path / "run"
    t = build_trainer(ck, extra=["--growth-every-steps", "1"])
    with pytest.raises(TL._Injected):
        t.train(max_steps=1, log_every=1, ckpt_every=1, growth_fail_at="during_commit")
    blob = CK.load_checkpoint(ck)
    pend = blob["manifest"]["scheduler"]["pending"]
    assert pend is not None and pend["sources"] is not None
    pre_growth_experts = len(M.expert_ids(blob["params"]))

    t2 = build_trainer(ck, extra=["--growth-every-steps", "1"])
    t2.load(ck)
    # max_steps must pass the crash step so the retained pending event resolves first
    t2.train(max_steps=2, log_every=1)
    entry = t2.sched.history[0]
    assert entry["id"] == pend["id"] and entry["sources"] == pend["sources"]
    assert entry["new_count"] == pre_growth_experts + 2
    assert params_equal(tensors(t2.model.p), crash_reference["params"])


# ---------------------------------------------------------------- publish / backup
def test_bak_fallback_and_republish(tmp_path):
    ck = tmp_path / "run"
    t = build_trainer(ck)
    t.train(max_steps=1, log_every=1, ckpt_every=1)
    good = CK._sha256(str(ck / "params.pt"))

    # simulate a crash between target -> .bak and tmp -> target
    bak = tmp_path / "run.bak"
    os.rename(ck, bak)
    assert CK._resolve_checkpoint_dir(str(ck)) == str(bak)
    assert CK.load_checkpoint(str(ck))["dir"] == str(bak)

    t.load(ck)
    assert t.stream.cursor == t.args.seq_len + 1
    t.train(max_steps=2, log_every=1, ckpt_every=1)  # republishes the target
    assert (ck / "manifest.json").exists()
    assert CK.load_checkpoint(str(ck))["dir"] == str(ck)
    assert CK._sha256(str(ck / "params.pt")) != good  # params advanced


def test_publish_failure_rolls_back(tmp_path):
    ck = tmp_path / "run"
    t = build_trainer(ck)
    t.train(max_steps=1, log_every=1, ckpt_every=1)
    before = CK._sha256(str(ck / "params.pt"))

    real_rename = os.rename
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("injected publish failure")
        return real_rename(src, dst)

    os.rename = flaky
    try:
        with pytest.raises(OSError):
            t.train(max_steps=2, log_every=1, ckpt_every=1)
    finally:
        os.rename = real_rename
    assert (ck / "manifest.json").exists()
    assert CK._sha256(str(ck / "params.pt")) == before
    assert not (tmp_path / "run.bak").exists()
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".ckpt.")]


def test_checkpoint_integrity_and_payload_validation(tmp_path):
    ck = tmp_path / "run"
    t = build_trainer(ck, preset="dense-tiny")
    t.train(max_steps=1, log_every=1, ckpt_every=1)
    blob = CK.load_checkpoint(ck)
    man = json.loads((ck / "manifest.json").read_text())

    # missing payload
    miss = tmp_path / "miss"
    import shutil
    shutil.copytree(ck, miss)
    os.remove(miss / "optim.pt")
    with pytest.raises(FileNotFoundError):
        CK.load_checkpoint(miss)

    # tampered payload
    bad = tmp_path / "bad"
    shutil.copytree(ck, bad)
    with open(bad / "params.pt", "r+b") as f:
        f.seek(0, os.SEEK_END)
        f.write(b"x")
    with pytest.raises(ValueError, match="integrity mismatch"):
        CK.load_checkpoint(bad)

    # unsupported version
    with open(bad / "manifest.json", "w") as f:
        json.dump(dict(man, version="bogus"), f)
    with pytest.raises(ValueError, match="incompatible"):
        CK.load_checkpoint(bad)

    # payload validation
    dup = dict(man, scheduler=dict(man["scheduler"], history=[{"id": 1}, {"id": 1}]))
    with pytest.raises(ValueError, match="duplicate committed event ids"):
        CK.save_checkpoint(str(tmp_path / "dup"), dup, blob["params"], blob["opt_state"])
    pd = dict(man, scheduler=dict(man["scheduler"], history=[{"id": 4}], pending={"id": 4}))
    with pytest.raises(ValueError, match="already committed"):
        CK.save_checkpoint(str(tmp_path / "pd"), pd, blob["params"], blob["opt_state"])
    reg = dict(man, n_experts_current=3, expert_registry=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="registry"):
        CK.save_checkpoint(str(tmp_path / "reg"), reg, blob["params"], blob["opt_state"])
    assert CK.blank_scheduler_state(7)["next_periodic_step"] == 7
    assert CK.blank_scheduler_state(0)["next_periodic_step"] is None


def test_legacy_npz_checkpoint_loads(tmp_path):
    ck = tmp_path / "run"
    t = build_trainer(ck, preset="dense-tiny")
    t.train(max_steps=1, log_every=1, ckpt_every=1)
    blob = CK.load_checkpoint(ck)
    man = json.loads((ck / "manifest.json").read_text())

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    np.savez(legacy / "params.npz", **{k: v.numpy() for k, v in blob["params"].items()})
    flat = {}
    for name, state in blob["opt_state"].items():
        for sub, val in state.items():
            flat[f"{name}::{sub}"] = val.numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
    np.savez(legacy / "optim.npz", **flat)
    lman = dict(man, version=CK.LEGACY_VERSION,
                files={"params.npz": CK._sha256(str(legacy / "params.npz")),
                       "optim.npz": CK._sha256(str(legacy / "optim.npz"))})
    (legacy / "manifest.json").write_text(json.dumps(lman))

    lb = CK.load_checkpoint(legacy)
    assert params_equal(lb["params"], blob["params"])
    assert "embed" in lb["opt_state"] and set(lb["opt_state"]["embed"]) == {"m", "R", "C"}


# ---------------------------------------------------------------- paging store
def _weights(fill, d=4, h=3):
    return {"Win": torch.full((d, h), float(fill)), "Wout": torch.full((h, d), float(fill))}


def _store_dir(tmp_path, n=3):
    dd = tmp_path / "mirror"
    dd.mkdir()
    for e in range(n):
        torch.save(_weights(e), dd / f"expert_{e}.pt")
    return dd


def test_expert_store_eviction_and_pinning(tmp_path):
    dd = _store_dir(tmp_path)
    s = PG.ExpertStore("D2R", 3, disk_dir=str(dd), ram_cache_max_experts=1)
    assert s.get(0)["Win"][0, 0].item() == 0.0
    assert s.get(1)["Win"][0, 0].item() == 1.0
    assert s.evictions == 1 and list(s.cache) == [1]

    s.pin(0)
    s.get(0)
    s.pin(1)
    with pytest.raises(PG.PagingError, match="pinned"):
        s.get(1)

    # a pinned oldest entry is skipped in favour of an unpinned victim
    # (here the just-loaded, unpinned expert 1 is the victim; expert 0 survives)
    s2 = PG.ExpertStore("D2R", 3, disk_dir=str(dd), ram_cache_max_experts=1)
    s2.pin(0)
    s2.get(0)
    s2.get(1)
    assert list(s2.cache) == [0] and s2.evictions == 1


def test_expert_store_write_back_invalidate_remove_register(tmp_path):
    dd = _store_dir(tmp_path)
    s = PG.ExpertStore("D2R", 3, disk_dir=str(dd), ram_cache_max_experts=4)
    s.get(0)
    s.write_back(0, _weights(9.0))
    assert s.get(0)["Win"][0, 0].item() == 9.0
    assert 0 in s._dirty
    s.invalidate_all()
    assert not s.cache and not s._dirty
    assert torch.load(dd / "expert_0.pt", weights_only=True)["Win"][0, 0].item() == 9.0

    # uncached write_back goes straight to disk
    s.write_back(2, _weights(7.0))
    assert torch.load(dd / "expert_2.pt", weights_only=True)["Win"][0, 0].item() == 7.0

    s.get(2)
    s.remove_expert(2)
    assert not (dd / "expert_2.pt").exists()
    assert 2 not in s.cache and 2 not in s._dirty

    s.register_expert(5, _weights(1.0))
    assert s.n_experts == 6 and (dd / "expert_5.pt").exists()


def test_expert_store_staging_bound_and_legacy_npz_and_modes(tmp_path):
    dd = _store_dir(tmp_path)
    s = PG.ExpertStore("D2R", 3, disk_dir=str(dd), ram_cache_max_experts=4, staging_max_bytes=8)
    with pytest.raises(PG.PagingError, match="staging size"):
        s.get(0)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    np.savez(legacy / "expert_1.npz", Win=np.full((4, 3), 5.0, np.float32),
             Wout=np.full((3, 4), 6.0, np.float32))
    sl = PG.ExpertStore("D2R", 2, disk_dir=str(legacy))
    w = sl.get(1)
    assert w["Win"][0, 0].item() == 5.0 and w["Wout"][0, 0].item() == 6.0

    with pytest.raises(ValueError, match="unknown paging mode"):
        PG.ExpertStore("bogus", 1, ram_weights={0: _weights(1.0)})
    with pytest.raises(ValueError, match="requires disk_dir"):
        PG.ExpertStore("D2R", 1)
    with pytest.raises(ValueError, match="requires RAM-resident"):
        PG.ExpertStore("R", 1)
    with pytest.raises(PG.PagingError, match="compatible VRAM"):
        PG.ExpertStore("V", 1)
    sr = PG.ExpertStore("R", 2, ram_weights={0: _weights(1.0)})
    with pytest.raises(KeyError, match="not resident"):
        sr.get(1)
    sr.write_back(0, _weights(4.0))
    assert sr.get(0)["Win"][0, 0].item() == 4.0


# ---------------------------------------------------------------- ByteStream
def test_bytestream_wrap_and_state_roundtrip(tmp_path):
    p = tmp_path / "corpus.bin"
    payload = bytes(range(256)) * 8
    p.write_bytes(payload)
    bs = D.ByteStream(path=str(p))
    assert bs.size == len(payload)
    assert bs.next_tokens(SEQ_LEN) == list(payload[:SEQ_LEN + 1])
    assert bs.cursor == SEQ_LEN + 1

    sd = bs.state_dict()
    assert sd == {"cursor": bs.cursor, "path": str(p), "size": bs.size}

    # wrap at the epoch boundary restarts at offset 0
    bs.cursor = bs.size - 3
    assert bs.next_tokens(SEQ_LEN) == list(payload[:SEQ_LEN + 1])
    assert bs.cursor == SEQ_LEN + 1

    bs.cursor = 0
    bs.load_state_dict(sd)
    assert bs.cursor == SEQ_LEN + 1
    with pytest.raises(ValueError, match="dataset mismatch"):
        bs.load_state_dict({"cursor": 0, "path": str(p) + "x", "size": bs.size})
    with pytest.raises(ValueError, match="dataset mismatch"):
        bs.load_state_dict({"cursor": 0, "path": str(p), "size": bs.size + 1})

    bs.close()
    with pytest.raises(ValueError, match="closed"):
        bs.next_tokens(SEQ_LEN)


def test_bytestream_and_synth_bytes_edges(tmp_path):
    with pytest.raises(ValueError, match="needs path or data"):
        D.ByteStream()
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty corpus"):
        D.ByteStream(path=str(empty))
    small = D.ByteStream(data=b"abc")
    with pytest.raises(ValueError, match="smaller than seq_len"):
        small.next_tokens(8)
    with pytest.raises(ValueError, match="seq_len must be >= 1"):
        small.next_tokens(0)

    a = D.synth_bytes(1000, seed=5)
    b = D.synth_bytes(1000, seed=5)
    c = D.synth_bytes(1000, seed=6)
    assert len(a) == 1000 and a == b and a != c
    base = bytes((i * 37 + 11) % 251 + 2 for i in range(64))
    assert a[:64] == base
    assert D.synth_bytes(0, seed=0) == b""
    assert len(D.synth_bytes(5, seed=0, period=4)) == 5


# ---------------------------------------------------------------- cli/config
def test_resolve_preset_combinations():
    assert cfg.resolve_preset("full", True, False) == "moe-full"
    assert cfg.resolve_preset("full", False, True) == "dense-full"
    assert cfg.resolve_preset("tiny", True, False) == "moe-tiny"
    assert cfg.resolve_preset("tiny", False, True) == "dense-tiny"
    assert cfg.resolve_preset(" MOE-TINY ", False, False) == "moe-tiny"
    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.resolve_preset("full", True, True)
    for bad in (("full", False, False), ("tiny", False, False), ("bogus", False, False), ("", False, False)):
        with pytest.raises(ValueError):
            cfg.resolve_preset(*bad)
    with pytest.raises(ValueError, match="conflicts"):
        cfg.resolve_preset("dense-tiny", True, False)
    with pytest.raises(ValueError, match="conflicts"):
        cfg.resolve_preset("moe-full", False, True)


@pytest.mark.parametrize("raw,expected", [("0", 0), ("7", 7), (3, 3), (0, 0)])
def test_parse_growth_every_steps_ok(raw, expected):
    assert cfg.parse_growth_every_steps(raw) == expected


@pytest.mark.parametrize("raw", ["none", -1, "nan", "inf", "", None, "1.5"])
def test_parse_growth_every_steps_rejects(raw):
    with pytest.raises(ValueError, match="non-negative integer|>= 0"):
        cfg.parse_growth_every_steps(raw)


@pytest.mark.parametrize("raw,expected", [("none", None), ("0", 0.0), (0, 0.0), (1.5, 1.5), (None, 0.5)])
def test_parse_growth_loss_nats_ok(raw, expected):
    assert cfg.parse_growth_loss_nats(raw) == expected


@pytest.mark.parametrize("raw", [-1, "nan", "inf", "-inf", "", "abc"])
def test_parse_growth_loss_nats_rejects(raw):
    with pytest.raises(ValueError, match="finite non-negative"):
        cfg.parse_growth_loss_nats(raw)


@pytest.mark.parametrize("raw,expected", [("0", 0), ("10", 10), (0, 0)])
def test_parse_mtp_depth_ok(raw, expected):
    assert cfg.parse_mtp_depth(raw) == expected


@pytest.mark.parametrize("raw", ["none", -1, "nan", "inf", "", None])
def test_parse_mtp_depth_rejects(raw):
    with pytest.raises(ValueError, match="non-negative integer|>= 0"):
        cfg.parse_mtp_depth(raw)


@pytest.mark.parametrize("raw,expected", [("0", 0.0), ("0.5", 0.5), (0, 0.0)])
def test_parse_mtp_loss_weight_ok(raw, expected):
    assert cfg.parse_mtp_loss_weight(raw) == expected


@pytest.mark.parametrize("raw", ["none", -1, "nan", "inf", "", None])
def test_parse_mtp_loss_weight_rejects(raw):
    with pytest.raises(ValueError, match="finite non-negative"):
        cfg.parse_mtp_loss_weight(raw)


@pytest.mark.parametrize("extra,match", [
    (["--paging-cache-experts", "0"], "paging-cache-experts"),
    (["--expert-capacity-factor", "0"], "capacity-factor"),
    (["--expert-capacity-factor", "-1"], "capacity-factor"),
    (["--seq-len", "0"], "seq-len"),
    (["--seq-len", "-4"], "seq-len"),
    (["--forward-quant", "fp8", "--backward-mode", "ternary"], "unsupported quantization"),
])
def test_validate_args_rejects_bad_config(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        make_args(tmp_path / "run", extra=extra)


def test_validate_args_accepts_valid_and_is_idempotent(tmp_path):
    args = make_args(tmp_path / "run", extra=["--growth-every-steps", "5",
                                             "--growth-loss-nats", "0.25",
                                             "--mtp-depth", "1", "--mtp-loss-weight", "0",
                                             "--seq-len", "1"])
    assert args.growth_every_steps == 5 and args.growth_loss_nats == 0.25
    assert args.mtp_depth == 1 and args.mtp_loss_weight == 0
    assert args.seq_len == 1
    assert cli.validate_args(args) == "moe-tiny"
