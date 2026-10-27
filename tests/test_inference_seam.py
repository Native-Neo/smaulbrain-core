"""Regression tests for the prefill/generate seam.

Before the fix, ``generate_stream`` re-stepped the prompt's last token after
``prefill`` had already consumed it, so ``prefill(p) + generate([])``
double-consumed the prompt tail (or fed literal byte 0) and diverged from a
single ``generate(p)`` call. All forms must now agree token-for-token with an
independent manual reference loop.
"""
import numpy as np

import pytest

from smaulbrain import config as cfg, model as M
from smaulbrain.inference import InferenceEngine

PROMPT = [104, 101, 108, 108, 111]


def _greedy_reference(model, prompt, n):
    """Manual greedy loop through model.forward/step_token (no engine logic)."""
    eng = InferenceEngine(model, temperature=0.0)
    logits = np.asarray(model.forward([prompt], states=eng.states)[0])
    pending = logits[-1]
    out = []
    for _ in range(n):
        nxt = int(np.argmax(pending))
        out.append(nxt)
        pending = np.asarray(model.step_token(nxt, eng.states))
    return out


def _dense_model():
    spec = cfg.PRESETS["dense-tiny"]
    params = M.build_params(spec, np.random.default_rng(7), mtp_depth=2)
    return M.Model(spec, params, mtp_depth=2)


def test_seam_matches_reference_dense_and_speculative():
    model = _dense_model()
    ref = _greedy_reference(model, PROMPT, 16)
    assert list(InferenceEngine(model, temperature=0.0).generate(PROMPT, 16)) == ref
    assert list(InferenceEngine(model, temperature=0.0, speculative=2).generate(PROMPT, 16)) == ref


def test_prefill_then_empty_generate_is_identical_to_oneshot():
    model = _dense_model()
    ref = _greedy_reference(model, PROMPT, 16)
    plain = InferenceEngine(model, temperature=0.0)
    plain.prefill(PROMPT)
    seam_plain = list(plain.generate([], 16))
    spec = InferenceEngine(model, temperature=0.0, speculative=2)
    spec.prefill(PROMPT)
    seam_spec = list(spec.generate([], 16))
    assert seam_plain == ref
    assert seam_spec == ref
    # the delivered states must be identical too, not just the tokens
    for eng in (plain,):
        other = InferenceEngine(model, temperature=0.0)
        other.generate(PROMPT, 16)
        assert all(np.array_equal(a, b) for a, b in zip(eng.states, other.states))


def test_chunked_prefill_is_seam_exact():
    model = _dense_model()
    ref = _greedy_reference(model, PROMPT, 16)
    eng = InferenceEngine(model, temperature=0.0)
    eng.prefill(PROMPT, chunk=2)
    assert list(eng.generate([], 16)) == ref


def test_continuation_generate_calls_match_single_call():
    model = _dense_model()
    ref = _greedy_reference(model, PROMPT, 16)
    for engine in (InferenceEngine(model, temperature=0.0),
                   InferenceEngine(model, temperature=0.0, speculative=2)):
        first = engine.generate(PROMPT, 8)
        rest = engine.generate([], 8)
        assert first + rest == ref


def test_stop_ids_halt_gold_recurrent_stream():
    model = _dense_model()
    ref = _greedy_reference(model, PROMPT, 16)
    cut = InferenceEngine(model, temperature=0.0).generate(PROMPT, 64, stop_ids=(ref[5],))
    assert cut == ref[:6]


def test_fresh_engine_empty_prompt_keeps_byte0_seed():
    model = _dense_model()
    # Documented degenerate behavior: no context anywhere, first token comes
    # from a byte-0 step, identical across engines and equal to a manual loop.
    a = InferenceEngine(model, temperature=0.0).generate([], 8)
    b = InferenceEngine(model, temperature=0.0).generate([], 8)
    manual = InferenceEngine(model, temperature=0.0)
    pending = np.asarray(model.step_token(0, manual.states))
    ref = []
    for _ in range(8):
        nxt = int(np.argmax(pending))
        ref.append(nxt)
        pending = np.asarray(model.step_token(nxt, manual.states))
    assert a == b == ref


def test_seam_on_moe_model_with_speculation():
    spec = cfg.PRESETS["moe-tiny"]
    params = M.build_params(spec, np.random.default_rng(3), mtp_depth=1)
    model = M.Model(spec, params, mtp_depth=1)
    ref = _greedy_reference(model, PROMPT, 12)
    assert list(InferenceEngine(model, temperature=0.0, speculative=1).generate(PROMPT, 12)) == ref
    eng = InferenceEngine(model, temperature=0.0, speculative=1)
    eng.prefill(PROMPT)
    assert list(eng.generate([], 12)) == ref
