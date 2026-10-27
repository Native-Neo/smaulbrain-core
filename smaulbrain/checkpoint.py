"""Atomic, checksummed NumPy/safetensors checkpoints with fallback recovery."""
import json
import os
import shutil
import tempfile
import time

import numpy as np

from . import npgrad as _ng
from . import safetensors_np as _st
from . import vocab as _vocab
from smaulbrain import CHECKPOINT_VERSION

LEGACY_VERSION = "smaulbrain-v2:1"
LEGACY_TORCH_VERSION = "smaulbrain-pytorch-v1:1"

_CKPT_TMP_PREFIX = ".ckpt."
_CKPT_STALE_SECONDS = 60.0


def sweep_stale_checkpoints(target_dir, grace=None):
    """Remove publication debris left around ``target_dir``.

    A trainer killed mid-save can leave ``<parent>/.ckpt.*`` build trees from
    before the atomic publish and a ``<target>.bak`` tree from the
    old-target rename.  A later save sweeps them so interrupted runs do not
    accumulate debris.  Entries newer than ``grace`` seconds (default 60)
    are never touched, which protects a concurrent writer's fresh temporary.
    A ``.bak`` tree is always kept while the target checkpoint has no
    manifest of its own: in the crash-mid-publish state that fallback tree
    may be the only complete checkpoint on disk, and
    ``_resolve_checkpoint_dir`` must still be able to find it.  It becomes
    debris the moment the target is valid again, exactly like the tree the
    normal publish path deletes after a successful rename.

    Returns the list of removed paths.
    """
    grace = _CKPT_STALE_SECONDS if grace is None else float(grace)
    target_dir = os.path.abspath(target_dir)
    parent = os.path.dirname(target_dir)
    now = time.time()
    removed = []
    try:
        entries = os.listdir(parent)
    except OSError:
        return removed
    for name in entries:
        if not name.startswith(_CKPT_TMP_PREFIX):
            continue
        path = os.path.join(parent, name)
        if os.path.islink(path) or not os.path.isdir(path):
            continue
        try:
            cold = now - os.path.getmtime(path) >= grace
        except OSError:
            cold = False
        if cold:
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                removed.append(path)
    bak = target_dir + ".bak"
    if os.path.isdir(bak) and not os.path.islink(bak):
        have_target = os.path.isfile(os.path.join(target_dir, "manifest.json"))
        bak_manifest = os.path.isfile(os.path.join(bak, "manifest.json"))
        try:
            cold = now - os.path.getmtime(bak) >= grace
        except OSError:
            cold = False
        # Reap a bak once the new target stands on its own, or once a
        # manifest-less bak is cold enough not to belong to an in-flight
        # publisher.  A bak without a valid target is kept unconditionally.
        if have_target or (not bak_manifest and cold):
            shutil.rmtree(bak, ignore_errors=True)
            if not os.path.exists(bak):
                removed.append(bak)
    return removed



def _sha256(path):
    return _st.file_sha256(path)


def blank_scheduler_state(growth_every_steps=10000, growth_loss_nats=0.5, loss_beta=0.98, min_eval_samples=256):
    interval = int(growth_every_steps)
    return {"global_step": 0, "growth_every_steps": interval,
            "next_periodic_step": interval if interval > 0 else None,
            "loss_threshold": growth_loss_nats, "loss_beta": float(loss_beta),
            "min_eval_samples": int(min_eval_samples),
            "smooth_loss": None, "loss_armed": True,
            "event_count": 0, "last_growth_step": None, "history": [], "pending": None}


def build_manifest(preset_name, arch, extra, sched, data_cursor, rng_state, n_experts_current,
                   expert_registry, opt_meta):
    return {"version": CHECKPOINT_VERSION, "preset": preset_name,
            "preset_expert_count_init": arch.get("experts_init"),
            "n_experts_current": n_experts_current, "expert_registry": expert_registry,
            "vocab": _vocab.vocab_meta(),
            "arch": arch, "config": extra, "scheduler": sched,
            "data_cursor": int(data_cursor), "rng_state": rng_state,
            "optimizer": opt_meta, "files": {}}


def _array(value):
    if isinstance(value, _ng.Tensor):
        value = value.data
    if np.isscalar(value):
        return np.asarray(value)
    return np.ascontiguousarray(np.asarray(value))


def _cpu_tree(obj):
    if isinstance(obj, _ng.Tensor):
        return _array(obj)
    if isinstance(obj, np.ndarray):
        return _array(obj)
    if isinstance(obj, dict):
        return {k: _cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_tree(v) for v in obj)
    return obj


def _flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}::{key}" if prefix else str(key), value, out)
    else:
        out[prefix] = _array(obj)


def _write_tree(target_dir, manifest, params, opt_state):
    parent = os.path.dirname(os.path.abspath(target_dir))
    os.makedirs(parent, exist_ok=True)
    sweep_stale_checkpoints(target_dir)
    tmp = tempfile.mkdtemp(prefix=_CKPT_TMP_PREFIX, dir=parent)
    try:
        pp = os.path.join(tmp, "params.safetensors")
        op = os.path.join(tmp, "optim.safetensors")
        params_flat = {str(k): _array(v) for k, v in params.items()}
        optim_flat = {}
        _flatten("", _cpu_tree(opt_state), optim_flat)
        _st.save_file(params_flat, pp, metadata={"format": "smaulbrain-params"})
        _st.save_file(optim_flat, op, metadata={"format": "smaulbrain-optim"})
        man = dict(manifest)
        man["version"] = CHECKPOINT_VERSION
        man["files"] = {"params.safetensors": _sha256(pp), "optim.safetensors": _sha256(op)}
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(man, f, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        _publish_dir(tmp, target_dir)
        _sync_parent(target_dir)
        return target_dir
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _sync_parent(path):
    """Flush a directory entry so a rename survives an actual power loss."""
    try:
        fd = os.open(os.path.dirname(os.path.normpath(path)), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _publish_dir(tmp, target_dir):
    target_dir = os.path.normpath(target_dir)
    _sync_parent(tmp)
    if not os.path.exists(target_dir):
        os.replace(tmp, target_dir)
        _sync_parent(target_dir)
        return
    bak = target_dir + ".bak"
    if os.path.exists(bak):
        shutil.rmtree(bak)
    os.rename(target_dir, bak)
    try:
        os.rename(tmp, target_dir)
        _sync_parent(target_dir)
    except BaseException:
        os.rename(bak, target_dir)
        raise
    shutil.rmtree(bak, ignore_errors=True)


def _resolve_checkpoint_dir(ckpt_dir):
    if os.path.exists(os.path.join(ckpt_dir, "manifest.json")):
        return ckpt_dir
    bak = os.path.normpath(ckpt_dir) + ".bak"
    return bak if os.path.exists(os.path.join(bak, "manifest.json")) else ckpt_dir


def _json_default(o):
    if isinstance(o, _ng.Tensor):
        return o.detach().tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"unserializable {type(o)}")


def save_checkpoint(target_dir, manifest, params, opt_state):
    sched = manifest.get("scheduler") or {}
    if not isinstance(sched, dict):
        raise ValueError("scheduler state must be an object")
    hist = sched.get("history", [])
    if not isinstance(hist, list):
        raise ValueError("scheduler history must be a list")
    if any(not isinstance(c, dict) or not isinstance(c.get("id"), int) for c in hist):
        raise ValueError("committed history entries require integer ids")
    ids = [c["id"] for c in hist]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate committed event ids")
    pending = sched.get("pending")
    if pending is not None and not isinstance(pending, dict):
        raise ValueError("scheduler pending must be an object")
    if pending is not None and pending.get("id") in ids:
        raise ValueError("pending event id already committed")
    _check_consistency(manifest, params)
    return _write_tree(target_dir, manifest, params, opt_state)


def _check_consistency(manifest, params):
    n = manifest.get("n_experts_current")
    reg = manifest.get("expert_registry", [])
    if n is not None and reg and len(reg) != n:
        raise ValueError("expert registry length disagrees with n_experts_current")
    if n is not None and manifest.get("preset", "").startswith("moe"):
        eids = sorted({int(k.split(".")[1]) for k in params
                       if k.startswith("expert.") and k.endswith(".Win")})
        if eids != list(range(len(eids))):
            raise ValueError("expert params are not a dense 0..E-1 registry")
        if len(eids) != n:
            raise ValueError("param expert count disagrees with manifest")


def _unflatten(flat):
    opt_state = {}
    for key, value in flat.items():
        name, _, sub = key.partition("::")
        opt_state.setdefault(name, {})[sub] = value
    return opt_state


def load_checkpoint(ckpt_dir):
    ckpt_dir = _resolve_checkpoint_dir(ckpt_dir)
    mp = os.path.join(ckpt_dir, "manifest.json")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"no manifest at {mp}")
    with open(mp) as f:
        man = json.load(f)
    version = man.get("version")
    if version == LEGACY_TORCH_VERSION:
        raise ValueError(
            "checkpoint is a legacy PyTorch checkpoint; convert it with "
            "tests/_reference/convert_torch_checkpoint.py before loading")
    if version not in (CHECKPOINT_VERSION, LEGACY_VERSION):
        raise ValueError(f"incompatible checkpoint version {version!r}")
    if version == CHECKPOINT_VERSION:
        names = ("params.safetensors", "optim.safetensors")
        for name in names:
            path = os.path.join(ckpt_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"checkpoint missing {name} (incomplete write?)")
            if man.get("files", {}).get(name) not in (None, _sha256(path)):
                raise ValueError(f"integrity mismatch for {name}")
        params = _st.load(os.path.join(ckpt_dir, "params.safetensors"))
        optim_flat = _st.load(os.path.join(ckpt_dir, "optim.safetensors"))
        opt_state = _unflatten(optim_flat)
    else:
        for name in ("params.npz", "optim.npz"):
            path = os.path.join(ckpt_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"checkpoint missing {name}")
            if man.get("files", {}).get(name) not in (None, _sha256(path)):
                raise ValueError(f"integrity mismatch for {name}")
        with np.load(os.path.join(ckpt_dir, "params.npz")) as z:
            params = {k: z[k].copy() for k in z.files}
        with np.load(os.path.join(ckpt_dir, "optim.npz")) as z:
            flat = {k: z[k].copy() for k in z.files}
        opt_state = _unflatten(flat)
    _check_consistency(man, params)
    return {"manifest": man, "params": params, "opt_state": opt_state, "dir": ckpt_dir}
