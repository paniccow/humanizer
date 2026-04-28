"""Rejection-sampling humanizer — covers accept / escalate / exhaust /
similarity-filter paths with fully in-memory stubs (no model loads, no
network).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from humanizer.detectors.base import Detector
from humanizer.humanizers.base import HumanizeResult, Humanizer
from humanizer.humanizers.rejection import (
    RejectionConfig,
    RejectionSamplingHumanizer,
    _temp_for_round,
)


class _StubBase(Humanizer):
    """Returns a hardcoded list-of-rounds. round k yields candidates[k]."""

    name = "stub-base"

    def __init__(self, candidates_by_round: List[List[str]]):
        self.candidates_by_round = candidates_by_round
        # (n, temperature, system_prompt) tuples — captures kwargs the rejection
        # sampler passed in. system_prompt is None unless strategy carousel set.
        self.calls: list[tuple[int, float | None, str | None]] = []
        self._round = 0

    def humanize(self, text: str, **_) -> HumanizeResult:
        return HumanizeResult(original=text, text=self.candidates_by_round[0][0])

    def sample(self, text: str, n: int, *, temperature=None, top_p=None,
               system_prompt=None, user_template=None) -> List[str]:
        self.calls.append((n, temperature, system_prompt))
        cands = self.candidates_by_round[self._round]
        self._round += 1
        return list(cands[:n])


class _StubJudge(Detector):
    """Returns fixed p_ai for each text via a dict; default 0.99."""

    name = "stub-judge"

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls = 0

    def score(self, text: str) -> float:
        self.calls += 1
        return self.scores.get(text, 0.99)


def _patch_similarity(monkeypatch, sim_value: float = 1.0):
    """All similarity comparisons return sim_value — bypass the embedding model."""
    def fake_sim(a, b, *, model_id=None):
        return np.array([sim_value] * len(a), dtype=np.float32)

    monkeypatch.setattr(
        "humanizer.humanizers.rejection.embedding_similarity", fake_sim
    )


def test_accepts_first_passing_candidate(monkeypatch):
    _patch_similarity(monkeypatch)
    base = _StubBase([["A", "B", "C"]])  # one round, three cands
    judge = _StubJudge({"A": 0.9, "B": 0.02, "C": 0.5})
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(candidates_per_round=3, max_rounds=2, p_ai_threshold=0.05),
    )
    out = h.humanize("source")
    assert out.text == "B"
    assert out.metadata["passed"] is True
    assert out.metadata["rounds_used"] == 1
    # Early exit: A scored, B scored & accepted; C never scored.
    assert judge.calls == 2


def test_escalates_when_first_round_fails(monkeypatch):
    _patch_similarity(monkeypatch)
    base = _StubBase([
        ["round0_a", "round0_b"],   # both fail
        ["round1_a", "round1_b"],   # round1_a passes
    ])
    judge = _StubJudge({
        "round0_a": 0.8, "round0_b": 0.7,
        "round1_a": 0.01, "round1_b": 0.6,
    })
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=2, max_rounds=4, p_ai_threshold=0.05,
            temperature_ramp=[0.85, 1.05, 1.2, 1.3],
        ),
    )
    out = h.humanize("source")
    assert out.text == "round1_a"
    assert out.metadata["passed"] is True
    assert out.metadata["rounds_used"] == 2
    assert out.metadata["best_round"] == 1
    # Temperature ramp applied: round 0 = 0.85, round 1 = 1.05.
    assert base.calls == [(2, 0.85, None), (2, 1.05, None)]


def test_exhausted_returns_best_with_passed_false(monkeypatch):
    _patch_similarity(monkeypatch)
    base = _StubBase([
        ["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"],
    ])
    judge = _StubJudge({
        "a": 0.8, "b": 0.6, "c": 0.5, "d": 0.4,
        "e": 0.3, "f": 0.35, "g": 0.2, "h": 0.25,
    })
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=2, max_rounds=4,
            p_ai_threshold=0.05, fallback_to_best=True,
        ),
    )
    out = h.humanize("source")
    assert out.metadata["passed"] is False
    assert out.metadata["rounds_used"] == 4
    # Best across all rounds: g with p_ai=0.2.
    assert out.text == "g"
    assert out.score == pytest.approx(0.2)


def test_similarity_filter_drops_off_topic_candidates(monkeypatch):
    # Every candidate from round 0 is "off-topic" (sim 0.5 < 0.78);
    # round 1 candidates are on-topic. Rejection sampler should skip
    # judging the round-0 batch entirely.
    sim_pool = {0: 0.5, 1: 1.0}

    def fake_sim(a, b, *, model_id=None):
        # First call (round 0) returns low sim; subsequent calls return high.
        out = sim_pool[fake_sim.calls]
        fake_sim.calls += 1
        return np.array([out] * len(a), dtype=np.float32)
    fake_sim.calls = 0
    import humanizer.humanizers.rejection as rej
    monkeypatch.setattr(rej, "embedding_similarity", fake_sim)

    base = _StubBase([
        ["off1", "off2"],                # round 0: dropped by sim filter
        ["on_passes", "on_fails"],       # round 1: on_passes passes
    ])
    judge = _StubJudge({
        "off1": 0.01, "off2": 0.01,      # would pass — but should never be scored
        "on_passes": 0.01, "on_fails": 0.5,
    })
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=2, max_rounds=2, p_ai_threshold=0.05,
        ),
    )
    out = h.humanize("source")
    assert out.text == "on_passes"
    assert out.metadata["rounds_used"] == 2
    # Crucially: judge was NOT called on the off-topic round 0 batch.
    assert judge.calls == 1  # only on_passes was scored (early exit)


def test_temp_ramp_index_clamps_to_last_value():
    cfg = RejectionConfig(temperature_ramp=[0.85, 1.0])
    assert _temp_for_round(cfg, 0) == 0.85
    assert _temp_for_round(cfg, 1) == 1.0
    assert _temp_for_round(cfg, 5) == 1.0  # clamps to last


def test_no_temp_ramp_returns_none():
    cfg = RejectionConfig(temperature_ramp=None)
    assert _temp_for_round(cfg, 0) is None


def test_concurrent_judge_scores_all_candidates_in_parallel(monkeypatch):
    """When concurrent_judge_calls > 1, all candidates get scored (no
    early-exit). The best is still chosen by lowest p_ai."""
    _patch_similarity(monkeypatch)
    base = _StubBase([["a", "b", "c", "d"]])
    # b is the best, but a would early-exit if we ran serially. With
    # concurrent_judge_calls=4 we score all four and pick b.
    judge = _StubJudge({"a": 0.04, "b": 0.001, "c": 0.5, "d": 0.6})
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=4, max_rounds=1,
            p_ai_threshold=0.05, early_exit_p_ai=0.05,
            concurrent_judge_calls=4,
        ),
    )
    out = h.humanize("source")
    # All 4 candidates were judged (no early-exit in concurrent mode).
    assert judge.calls == 4
    # Best (lowest p_ai) was chosen — b at 0.001, not a at 0.04.
    assert out.text == "b"
    assert out.score == pytest.approx(0.001)
    assert out.metadata["passed"] is True


def test_concurrent_judge_falls_back_to_serial_when_one_candidate(monkeypatch):
    """concurrent_judge_calls > 1 but only one candidate -> serial path."""
    _patch_similarity(monkeypatch)
    base = _StubBase([["only_one"]])
    judge = _StubJudge({"only_one": 0.01})
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=1, max_rounds=1,
            p_ai_threshold=0.05, concurrent_judge_calls=8,
        ),
    )
    out = h.humanize("source")
    assert out.text == "only_one"
    assert judge.calls == 1


def test_preservation_threshold_drops_fact_losing_candidates(monkeypatch):
    """When preservation_threshold > 0, candidates that drop facts are
    filtered out before the judge sees them."""
    _patch_similarity(monkeypatch)
    src = "Founded in 1947 with $5,000."
    base = _StubBase([[
        "Established back in 1947 with five thousand dollars.",   # keeps year + bare $5000-ish
        "It started a long time ago with no money.",              # drops both facts
    ]])
    # Judge would prefer the second one (lower fake p_ai), but with the
    # preservation gate we should reject it before scoring.
    judge = _StubJudge({
        "Established back in 1947 with five thousand dollars.": 0.5,
        "It started a long time ago with no money.": 0.01,
    })
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=2, max_rounds=1,
            p_ai_threshold=0.5,                # only need < 0.5 to pass
            preservation_threshold=0.5,        # require >= 50% facts preserved
        ),
    )
    out = h.humanize(src)
    # The "no money" version should have been dropped, leaving only the
    # 1947-preserving one.
    assert "1947" in out.text


def test_preservation_threshold_zero_is_disabled(monkeypatch):
    """Default 0.0 -> no fact filtering (back-compat)."""
    _patch_similarity(monkeypatch)
    src = "Founded in 1947."
    base = _StubBase([["Founded in 1948."]])  # year changed
    judge = _StubJudge({"Founded in 1948.": 0.01})
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=1, max_rounds=1,
            p_ai_threshold=0.05,
            # preservation_threshold defaults to 0.0
        ),
    )
    out = h.humanize(src)
    # With no preservation gate, the wrong-year candidate is accepted.
    assert out.text == "Founded in 1948."


def test_strategy_carousel_overrides_temperature_ramp(monkeypatch):
    """When prompt_strategies is set, each round uses a different system
    prompt + temperature; temperature_ramp is ignored."""
    from humanizer.humanizers.strategies import Strategy

    _patch_similarity(monkeypatch)
    s1 = Strategy(name="s1", system_prompt="prompt one", temperature=0.7)
    s2 = Strategy(name="s2", system_prompt="prompt two", temperature=1.3)
    base = _StubBase([
        ["round1_a"],   # s1 round
        ["round2_a"],   # s2 round (failed first; succeeds second by p_ai)
    ])
    judge = _StubJudge({"round1_a": 0.5, "round2_a": 0.01})
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=1, max_rounds=2, p_ai_threshold=0.05,
            prompt_strategies=[s1, s2],
            temperature_ramp=[99.0, 99.0],   # should be ignored
        ),
    )
    out = h.humanize("source")
    assert out.text == "round2_a"
    # Round 1: temp=0.7, system="prompt one"
    # Round 2: temp=1.3, system="prompt two"
    assert base.calls == [
        (1, 0.7, "prompt one"),
        (1, 1.3, "prompt two"),
    ]


def test_strategy_carousel_repeats_last_strategy_when_more_rounds(monkeypatch):
    """If max_rounds > len(strategies), extra rounds repeat the last one
    (not crash with IndexError)."""
    from humanizer.humanizers.strategies import Strategy

    _patch_similarity(monkeypatch)
    only = Strategy(name="only", system_prompt="prompt", temperature=1.0)
    base = _StubBase([["a"], ["b"], ["c"]])  # 3 rounds asked, 1 strategy provided
    judge = _StubJudge({"a": 0.5, "b": 0.5, "c": 0.5})
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=1, max_rounds=3, p_ai_threshold=0.05,
            prompt_strategies=[only],
        ),
    )
    h.humanize("source")
    # All 3 rounds used the same (only) strategy
    assert all(c == (1, 1.0, "prompt") for c in base.calls)


def test_no_strategy_carousel_falls_back_to_temperature_ramp(monkeypatch):
    """Default: no carousel -> temperature_ramp drives diversity, no
    system_prompt override."""
    _patch_similarity(monkeypatch)
    base = _StubBase([["a"], ["b"]])
    judge = _StubJudge({"a": 0.5, "b": 0.5})
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(
            candidates_per_round=1, max_rounds=2, p_ai_threshold=0.05,
            temperature_ramp=[0.8, 1.1],
            # prompt_strategies left at None default
        ),
    )
    h.humanize("source")
    # Temperature varies; system_prompt is None on every call.
    assert base.calls == [(1, 0.8, None), (1, 1.1, None)]


def test_metadata_contains_telemetry_fields(monkeypatch):
    _patch_similarity(monkeypatch)
    base = _StubBase([["good"]])
    judge = _StubJudge({"good": 0.01})
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(candidates_per_round=1, max_rounds=1, p_ai_threshold=0.05),
    )
    out = h.humanize("source")
    meta = out.metadata
    assert meta["passed"] is True
    assert meta["judge"] == "stub-judge"
    assert meta["judge_calls"] == 1
    assert meta["best_p_ai"] == pytest.approx(0.01)
    assert meta["best_round"] == 0
    assert meta["threshold"] == 0.05
