"""Inference / speculative-decoding audit tests (critic A).

Covers the README claims about MTP speculative decoding:

* ``temperature=0`` speculative output is token-identical to plain decoding for
  any depth / draft count / prompt length, with and without stop ids;
* ``Model.verify_tokens`` reproduces ``step_token`` row by row (logits, hidden,
  per-position states), including the no-drop MoE routing path;
* resuming after a stop inside the draft loop (and after a plain generation
  call) matches the uninterrupted greedy stream and leaves the main recurrent
  state identical to a plain engine's;
* ``export_states`` / ``import_states`` round-trip exactly, including across
  engines with a different ``speculative`` depth and blobs without a draft
  pipeline;
* chunked prefill equals unchunked prefill for dense models;
* long generations do not drift or stall the MTP catch-up;
* rejected drafts never pollute the exported MTP pipeline;
* engine argument validation.
"""
import torch
import pytest

from smaulbrain import config as cfg, data as D, inference as I, model as M


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def build_model(preset, seed, depth):
    spec = cfg.PRESETS[preset]
    params = M.build_params(spec, torch.Generator().manual_seed(seed), spec.n_experts_init,
                            mtp_depth=depth)
    return M.Model(spec, params, mtp_depth=depth)


def build_engine(preset, seed, depth, **kw):
    kw.setdefault("temperature", 0.0)
    return I.InferenceEngine(build_model(preset, seed, depth), seed=3, **kw)


def prompt_of(n, seed=7):
    return list(D.synth_bytes(n, seed=seed))


def state_maxdiff(a, b):
    assert len(a) == len(b)
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def assert_tokens_eq(a, b, label=""):
    assert a == b, f"{label}: got={a[:8]} want={b[:8]}"


class GreedyDraftEngine(I.InferenceEngine):
    """Engine whose drafts are the true greedy continuation.

    Every draft the verifier sees is an accepted draft, which stresses the
    accept-all / stop-among-accepted-drafts paths that untrained MTP modules
    (nearly all rejects) never reach.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.draft_calls = 0
        self.nonempty_drafts = 0
        self.accepted = 0

    def _mtp_draft(self, pos):
        st = [s.clone() for s in self.states]
        drafts = []
        qs = []
        cur = self._tok_at(pos + 1)
        for _ in range(self._mtp_K):
            logits = self.model.step_token(cur, st)
            p = self._probs(logits)
            drafts.append(int(torch.argmax(p).item()))
            qs.append(p)
            cur = drafts[-1]
        self.draft_calls += 1
        if drafts:
            self.nonempty_drafts += 1
        return drafts, qs

    def _accept(self, p, q, draft):
        ok, repl = super()._accept(p, q, draft)
        if ok:
            self.accepted += 1
        return ok, repl


class DraftlessEngine(I.InferenceEngine):
    """Speculative engine whose draft pass always comes back empty."""

    def _mtp_draft(self, pos):
        return [], []


def first_occurrence_stops(stream):
    """(index, token) for every token at its first occurrence in `stream`."""
    seen = set()
    out = []
    for i, tok in enumerate(stream):
        if tok in seen:
            continue
        seen.add(tok)
        out.append((i, tok))
    return out


# --------------------------------------------------------------------------
# 1. greedy identity (README)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("preset,seed", [("dense-tiny", 0), ("moe-tiny", 0), ("dense-tiny", 1)])
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("plen", [1, 16])
def test_greedy_identity_no_stop(preset, seed, depth, plen):
    model = build_model(preset, seed, depth)
    prompt = prompt_of(plen)
    plain = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=0)
    ref = plain.generate(prompt, max_new_tokens=50)
    for K in sorted({1, min(2, depth), min(3, depth)}):
        spec = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=K)
        got = spec.generate(prompt, max_new_tokens=50)
        assert_tokens_eq(got, ref, f"{preset} d={depth} K={K} plen={plen}")
        assert state_maxdiff(spec.states, plain.states) < 1e-5


@pytest.mark.parametrize("preset,seed,depth,K", [
    ("dense-tiny", 0, 4, 2), ("dense-tiny", 1, 3, 1), ("moe-tiny", 0, 4, 3), ("moe-tiny", 2, 2, 2)])
def test_greedy_identity_with_stop_ids(preset, seed, depth, K):
    prompt = prompt_of(7)
    plain = build_engine(preset, seed, depth, speculative=0)
    ref = plain.generate(prompt, max_new_tokens=50)
    s, stop = next((i, t) for i, t in first_occurrence_stops(ref) if i >= 1)
    for cls in (I.InferenceEngine, GreedyDraftEngine):
        spec = cls(build_model(preset, seed, depth), seed=3, temperature=0.0, speculative=K)
        stopped = spec.generate(prompt, max_new_tokens=50, stop_ids={stop})
        assert stopped == ref[:s + 1]
        resumed = spec.generate([], max_new_tokens=50 - s - 1)
        assert_tokens_eq(resumed, ref[s + 1:], f"resume {cls.__name__} {preset} d={depth} K={K}")


@pytest.mark.parametrize("preset,seed,depth,K", [
    ("dense-tiny", 0, 3, 3), ("moe-tiny", 1, 2, 2)])
def test_greedy_identity_forced_correct_drafts(preset, seed, depth, K):
    model = build_model(preset, seed, depth)
    prompt = prompt_of(11)
    ref = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=0).generate(
        prompt, max_new_tokens=40)
    spec = GreedyDraftEngine(model, seed=3, temperature=0.0, speculative=K)
    got = spec.generate(prompt, max_new_tokens=40)
    assert_tokens_eq(got, ref, f"forced drafts {preset} d={depth} K={K}")
    assert spec.accepted > 0, "forced-draft coverage requires accepted drafts"
    assert spec.nonempty_drafts == spec.draft_calls


# --------------------------------------------------------------------------
# 2. verify_tokens == sequential step_token
# --------------------------------------------------------------------------
@pytest.mark.parametrize("preset", ["dense-tiny", "moe-tiny"])
@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6])
def test_verify_tokens_matches_step_token(preset, T):
    model = build_model(preset, 4, 3)
    toks = prompt_of(40, seed=11)
    engine = I.InferenceEngine(model, temperature=0.0, speculative=2)
    engine.prefill(toks[:9])
    base = [s.clone() for s in engine.states]
    seq = toks[9:9 + T]

    st = [s.clone() for s in base]
    want_logits = torch.stack([model.step_token(t, st) for t in seq])
    want_hidden = []
    st = [s.clone() for s in base]
    for t in seq:
        _lg, h = model.hidden_forward([t], st)
        want_hidden.append(h[0])

    st2 = [s.clone() for s in base]
    got_logits, got_hidden, hist = model.verify_tokens(seq, st2)
    maxd = (got_logits - want_logits).abs().max().item()
    assert maxd < 1e-6, f"{preset} T={T}: max|dlogits|={maxd:.3e}"
    assert all(int(torch.argmax(got_logits[i])) == int(torch.argmax(want_logits[i]))
               for i in range(T))
    assert state_maxdiff(got_hidden, torch.stack(want_hidden)) < 1e-6
    # per-position recurrent states must match the sequential walk (rollback basis)
    walk = [s.clone() for s in base]
    for j, t in enumerate(seq):
        model.step_token(t, walk)
        for layer in range(len(walk)):
            assert (hist[layer][j] - walk[layer]).abs().max().item() < 1e-6, \
                f"hist layer {layer} row {j} mismatch"
    assert state_maxdiff(st2, walk) < 1e-6


# --------------------------------------------------------------------------
# 3. stop inside the draft loop: resumable, matches the uninterrupted stream
# --------------------------------------------------------------------------
@pytest.mark.parametrize("preset,seed,depth,K", [
    ("dense-tiny", 0, 3, 3), ("dense-tiny", 1, 1, 1), ("moe-tiny", 0, 2, 2)])
def test_stop_mid_draft_resume_is_exact(preset, seed, depth, K):
    model = build_model(preset, seed, depth)
    prompt = prompt_of(7)
    N = 24
    ref = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=0).generate(
        prompt, max_new_tokens=N)
    for s, tok in first_occurrence_stops(ref):
        for cls in (I.InferenceEngine, GreedyDraftEngine):
            eng = cls(model, seed=3, temperature=0.0, speculative=K)
            stopped = eng.generate(prompt, max_new_tokens=N, stop_ids={tok})
            assert stopped == ref[:s + 1], (cls.__name__, s, len(stopped))
            resumed = eng.generate([], max_new_tokens=N - s - 1)
            assert_tokens_eq(resumed, ref[s + 1:], f"{cls.__name__} stop@{s}")


@pytest.mark.parametrize("preset,seed,depth,K", [
    ("dense-tiny", 0, 3, 3), ("moe-tiny", 0, 2, 2)])
def test_stop_mid_draft_state_equals_plain_trajectory(preset, seed, depth, K):
    model = build_model(preset, seed, depth)
    prompt = prompt_of(7)
    N = 20
    plain = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=0)
    ref = plain.generate(prompt, max_new_tokens=N)
    s, tok = next((i, t) for i, t in first_occurrence_stops(ref) if i >= 1)
    spec = GreedyDraftEngine(model, seed=3, temperature=0.0, speculative=K)
    spec.generate(prompt, max_new_tokens=N, stop_ids={tok})
    spec.generate([], max_new_tokens=N - s - 1)
    assert state_maxdiff(spec.states, plain.states) < 1e-5


def test_resume_keeps_draft_pipeline_aligned():
    """`generate([], ...)` after a finished call must not stall the draft loop.

    Regression: the empty prefill used to clear the pending-token flag, the
    next round inserted a duplicate token record one position ahead of the real
    frontier, the hidden-state ring ended up with a hole and ``_mtp_catch_up``
    froze forever (no drafts were ever produced again).
    """
    for preset, seed, depth, K in (("dense-tiny", 0, 3, 3), ("moe-tiny", 0, 2, 2)):
        model = build_model(preset, seed, depth)
        eng = GreedyDraftEngine(model, seed=3, temperature=0.0, speculative=K)
        eng.generate(prompt_of(7), max_new_tokens=8)
        eng.generate([], max_new_tokens=12)
        frontier = eng._tok_base + len(eng._toks)
        eng._mtp_catch_up()  # a healthy pipeline reaches its frontier
        assert min(eng._mtp_pos) >= frontier - K, (
            f"{preset}: draft pipeline stalled at {eng._mtp_pos} (frontier {frontier})")
        before = eng.nonempty_drafts
        eng.generate([], max_new_tokens=4)
        assert eng.nonempty_drafts > before, f"{preset}: no drafts after resume"


# --------------------------------------------------------------------------
# 4. export / import
# --------------------------------------------------------------------------
def test_export_import_roundtrip_same_depth():
    for preset, seed, depth, K in (("dense-tiny", 0, 4, 3), ("moe-tiny", 0, 4, 2)):
        model = build_model(preset, seed, depth)
        src = I.InferenceEngine(model, seed=5, temperature=0.0, speculative=K)
        src.generate(prompt_of(9, seed=17), max_new_tokens=7)
        blob = src.export_states()
        assert all(isinstance(v, list) for k, v in blob.items() if k != "mtp")
        assert "mtp" in blob and blob["mtp"]["pos"]
        dst = I.InferenceEngine(model, seed=0, temperature=0.0, speculative=K)
        dst.import_states(blob)
        assert state_maxdiff(src.states, dst.states) == 0.0
        assert dst._mtp_pos == src._mtp_pos
        assert dst._toks == src._toks and dst._tok_base == src._tok_base
        for a, b in zip([src._h0] + src._h, [dst._h0] + dst._h):
            assert len(a) == len(b)
            for (pa, va), (pb, vb) in zip(a, b):
                assert pa == pb and torch.equal(va, vb)
        for a, b in zip(src._mtp_states, dst._mtp_states):
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.equal(a, b)
        ca = src.generate([], max_new_tokens=12)
        cb = dst.generate([], max_new_tokens=12)
        assert_tokens_eq(cb, ca, f"{preset} import same depth")
        # importing twice is idempotent
        once = I.InferenceEngine(model, seed=0, temperature=0.0, speculative=K)
        once.import_states(blob)
        twice = I.InferenceEngine(model, seed=0, temperature=0.0, speculative=K)
        twice.import_states(blob)
        twice.import_states(blob)
        assert twice._mtp_pos == once._mtp_pos
        assert state_maxdiff(twice.states, once.states) == 0.0
        assert_tokens_eq(twice.generate([], max_new_tokens=8),
                         once.generate([], max_new_tokens=8), f"{preset} double import")


def test_export_import_cross_depth():
    """A blob from a small draft depth must import into a larger engine."""
    for preset, seed, depth in (("dense-tiny", 0, 4), ("moe-tiny", 1, 4)):
        model = build_model(preset, seed, depth)
        prompt = prompt_of(9, seed=17)
        src = I.InferenceEngine(model, seed=5, temperature=0.0, speculative=2)
        src.generate(prompt, max_new_tokens=7)
        blob = src.export_states()
        want = src.generate([], max_new_tokens=10)
        for eng_k in (2, 4):
            dst = I.InferenceEngine(model, seed=0, temperature=0.0, speculative=eng_k)
            dst.import_states(blob)
            got = dst.generate([], max_new_tokens=10)
            assert_tokens_eq(got, want, f"{preset} blobK=2 engK={eng_k}")
            assert state_maxdiff(src.states, dst.states) < 1e-5
            assert len(dst._mtp_pos) == dst.draft_depth, f"{preset} padded to {eng_k}"
    # a deeper exporter's blob into a shallower engine: extra depths are
    # dropped and the continuation still matches the source's own resume
    model = build_model("dense-tiny", 0, 4)
    prompt = prompt_of(9, seed=17)
    src = I.InferenceEngine(model, seed=5, temperature=0.0, speculative=4)
    src.generate(prompt, max_new_tokens=7)
    blob = src.export_states()
    want = src.generate([], max_new_tokens=10)
    dst = I.InferenceEngine(model, seed=0, temperature=0.0, speculative=2)
    dst.import_states(blob)
    assert len(dst._mtp_pos) == 2
    assert_tokens_eq(dst.generate([], max_new_tokens=10), want, "blobK=4 engK=2")


def test_export_import_plain_blob_re_enables_drafting_on_prefill():
    """Blobs without a draft pipeline stay on the plain path until a prefill."""
    preset, seed, depth = "dense-tiny", 3, 3
    model = build_model(preset, seed, depth)
    prompt = prompt_of(13, seed=5)
    plain = I.InferenceEngine(model, seed=4, temperature=0.0, speculative=0)
    plain.generate(prompt, max_new_tokens=6)
    blob = plain.export_states()
    assert "mtp" not in blob

    src = I.InferenceEngine(model, seed=4, temperature=0.0, speculative=3)
    src.import_states(blob)
    src.prefill(prompt_of(6, seed=6))
    assert src._mtp_ready is True
    # Drafting re-enabled: the continuation must equal the draftless reference
    # on the same imported context (greedy identity), not a fresh engine's
    # resume of the new prompt alone.
    ref = DraftlessEngine(model, seed=4, temperature=0.0, speculative=3)
    ref.import_states(blob)
    ref.prefill(prompt_of(6, seed=6))
    want = ref.generate([], max_new_tokens=10)
    got = src.generate([], max_new_tokens=10)
    assert_tokens_eq(got, want, "plain blob -> prefill -> generate([])")
    assert state_maxdiff(src.states, ref.states) < 1e-5

    spec = GreedyDraftEngine(model, seed=4, temperature=0.0, speculative=3)
    spec.import_states(blob)
    spec.prefill(prompt_of(6, seed=6))
    spec.generate([], max_new_tokens=4)
    assert spec.nonempty_drafts > 0


def test_export_after_prefill_roundtrip():
    """Exporting right after a prefill must resume exactly as the source does."""
    for preset, seed, depth, K in (("dense-tiny", 0, 3, 3), ("moe-tiny", 0, 2, 2)):
        model = build_model(preset, seed, depth)
        src = I.InferenceEngine(model, seed=5, temperature=0.0, speculative=K)
        src.prefill(prompt_of(13, seed=17))
        blob = src.export_states()
        cs = src.generate([], max_new_tokens=6)
        dst = I.InferenceEngine(model, seed=0, temperature=0.0, speculative=K)
        dst.import_states(blob)
        cd = dst.generate([], max_new_tokens=6)
        assert_tokens_eq(cd, cs, f"{preset} export-after-prefill")
        assert state_maxdiff(src.states, dst.states) < 1e-5


def test_import_fresh_blob_then_empty_resume():
    """Imports of an empty fresh pipeline must not crash on generate([])."""
    model = build_model("dense-tiny", 0, 2)
    src = I.InferenceEngine(model, seed=1, temperature=0.0, speculative=2)
    blob = src.export_states()
    dst = I.InferenceEngine(model, seed=1, temperature=0.0, speculative=2)
    dst.import_states(blob)
    assert dst.generate([], max_new_tokens=4) == src.generate([], max_new_tokens=4)


# --------------------------------------------------------------------------
# 5. chunked prefill
# --------------------------------------------------------------------------
@pytest.mark.parametrize("chunk", [1, 5, 8, 37, 64])
def test_chunked_prefill_dense_matches_unchunked(chunk):
    model = build_model("dense-tiny", 0, 3)
    toks = prompt_of(37)
    refs = {}
    for K in (0, 2):
        ref = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=K)
        ref.prefill(toks, chunk=0)
        st0 = [s.clone() for s in ref.states]
        refs[K] = (st0, ref.generate([], max_new_tokens=6))
    for K in (0, 2):
        ref_states, ref_next = refs[K]
        eng = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=K)
        eng.prefill(toks, chunk=chunk)
        assert state_maxdiff(eng.states, ref_states) == 0.0, f"chunk={chunk} K={K}"
        assert_tokens_eq(eng.generate([], max_new_tokens=6), ref_next, f"chunk={chunk} K={K}")


def test_chunked_prefill_dense_long_prompt():
    """Prompts beyond the 256-position catch-up window and non-dividing chunks."""
    model = build_model("dense-tiny", 1, 2)
    toks = prompt_of(600, seed=31)
    ref = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=2)
    ref.prefill(toks, chunk=0)
    want = ref.generate([], max_new_tokens=6)
    for chunk in (1, 257, 256):
        eng = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=2)
        eng.prefill(toks, chunk=chunk)
        got = eng.generate([], max_new_tokens=6)
        assert_tokens_eq(got, want, f"long prompt chunk={chunk}")


# --------------------------------------------------------------------------
# 6. long generation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("preset,seed,depth,K,N", [
    ("dense-tiny", 0, 4, 3, 320), ("moe-tiny", 0, 4, 2, 320)])
def test_long_generation_matches_plain(preset, seed, depth, K, N):
    model = build_model(preset, seed, depth)
    prompt = prompt_of(23, seed=9)
    plain = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=0)
    ref = plain.generate(prompt, max_new_tokens=N)
    spec = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=K)
    got = spec.generate(prompt, max_new_tokens=N)
    assert_tokens_eq(got, ref, f"long gen {preset} d={depth} K={K}")
    assert state_maxdiff(spec.states, plain.states) < 1e-5
    frontier = spec._tok_base + len(spec._toks)
    # the deepest depth may legitimately lag by up to ~K rows; a stall freezes
    # it far behind the frontier (regression test_resume_keeps_draft_pipeline_aligned)
    assert max(spec._mtp_pos) >= frontier - 8, "MTP catch-up stalled"
    assert min(spec._mtp_pos) >= frontier - 8, "MTP catch-up drifted behind"


# --------------------------------------------------------------------------
# 7. rejected drafts must not pollute the verified pipeline
# --------------------------------------------------------------------------
@pytest.mark.parametrize("preset,seed,depth,K", [
    ("dense-tiny", 0, 3, 3), ("moe-tiny", 0, 4, 2)])
def test_rejected_drafts_do_not_pollute_mtp_state(preset, seed, depth, K):
    model = build_model(preset, seed, depth)
    prompt = prompt_of(19, seed=21)
    drafting = I.InferenceEngine(model, seed=3, temperature=0.0, speculative=K)
    got = drafting.generate(prompt, max_new_tokens=60)
    nodraft = DraftlessEngine(model, seed=3, temperature=0.0, speculative=K)
    ref = nodraft.generate(prompt, max_new_tokens=60)
    assert_tokens_eq(got, ref, f"pollution {preset}")

    # full catch-up on both engines must agree: the draft pass persisted nothing
    drafting._mtp_catch_up()
    nodraft._mtp_catch_up()
    assert drafting._mtp_pos == nodraft._mtp_pos
    for a, b in zip(drafting._mtp_states, nodraft._mtp_states):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b)
    for ra, rb in zip([drafting._h0] + drafting._h, [nodraft._h0] + nodraft._h):
        da = dict(ra)
        db = dict(rb)
        for p in set(da) & set(db):
            assert torch.equal(da[p], db[p])

    # export after the rejection-heavy round, import, continue: exact
    blob = drafting.export_states()
    dst = I.InferenceEngine(model, seed=0, temperature=0.0, speculative=K)
    dst.import_states(blob)
    ca = drafting.generate([], max_new_tokens=20)
    cb = dst.generate([], max_new_tokens=20)
    assert_tokens_eq(cb, ca, f"export after rejections {preset}")


# --------------------------------------------------------------------------
# 8. argument validation
# --------------------------------------------------------------------------
def test_engine_argument_validation():
    model = build_model("dense-tiny", 0, 2)
    bad = [
        (dict(top_p=0.0), "--top-p"),
        (dict(top_p=1.5), "--top-p"),
        (dict(top_p=-1.0), "--top-p"),
        (dict(top_k=-1), "--top-k"),
        (dict(temperature=-0.5), "--temperature"),
        (dict(temperature=float("nan")), "--temperature"),
        (dict(speculative=-1), "speculative"),
        (dict(speculative=3), "mtp_depth"),
    ]
    for kw, needle in bad:
        with pytest.raises(ValueError) as err:
            I.InferenceEngine(model, seed=0, **kw)
        assert needle in str(err.value), (kw, str(err.value))
    ok = I.InferenceEngine(model, seed=0, temperature=0.0, top_p=1.0, top_k=0, speculative=2)
    assert ok.speculative == 2 and ok.draft_depth == 2
    assert I.InferenceEngine(model, seed=0, temperature=1e-9, top_p=0.999).temperature > 0
