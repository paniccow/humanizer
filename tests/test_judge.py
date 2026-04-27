"""Judge factory + EnsembleJudge adapter — fully in-memory tests.

Builds detector ensembles without loading torch/transformers. The
factory's _build_paid_detector and _build_local_fallback paths are
monkeypatched so we never actually instantiate a real RoBERTa or hit
a paid API.
"""
from __future__ import annotations

import pytest

from humanizer.detectors.base import Detector
from humanizer.detectors.ensemble import DetectorEnsemble
from humanizer.detectors.judge import (
    EnsembleJudge,
    available_paid_detectors,
    judge_from_env,
)


class _FakeDetector(Detector):
    def __init__(self, name: str, fixed_p_ai: float):
        self.name = name
        self._p = fixed_p_ai

    def score(self, text: str) -> float:
        return self._p


def test_ensemble_judge_returns_aggregate():
    ens = DetectorEnsemble([
        _FakeDetector("a", 0.2),
        _FakeDetector("b", 0.8),
    ])
    j = EnsembleJudge(ens, name="ab")
    assert j.score("hello") == pytest.approx(0.5)  # equal weights, mean of 0.2 and 0.8
    assert j.last_breakdown == {"a": 0.2, "b": 0.8}
    assert j.name == "ab"


def test_ensemble_judge_weighted():
    ens = DetectorEnsemble(
        [_FakeDetector("a", 0.0), _FakeDetector("b", 1.0)],
        weights=[1.0, 3.0],
    )
    j = EnsembleJudge(ens)
    # Weighted: (0*1 + 1*3) / 4 = 0.75
    assert j.score("x") == pytest.approx(0.75)


def test_available_paid_detectors_filters_by_env():
    avail = available_paid_detectors(env={"ORIGINALITY_API_KEY": "x"})
    assert [s.name for s in avail] == ["originality"]
    avail2 = available_paid_detectors(env={
        "ORIGINALITY_API_KEY": "x",
        "PANGRAM_API_KEY": "y",
    })
    assert [s.name for s in avail2] == ["originality", "pangram"]
    avail3 = available_paid_detectors(env={})
    assert avail3 == []


def test_judge_from_env_falls_back_to_local(monkeypatch):
    sentinel = _FakeDetector("local-roberta", 0.5)
    import humanizer.detectors.judge as jmod
    monkeypatch.setattr(jmod, "_build_local_fallback", lambda: sentinel)

    j = judge_from_env(env={}, fallback_to_local=True)
    assert j is sentinel


def test_judge_from_env_raises_when_no_keys_and_no_fallback():
    with pytest.raises(RuntimeError, match="no paid-detector API keys"):
        judge_from_env(env={}, fallback_to_local=False)


def test_judge_from_env_single_key_returns_bare_detector(monkeypatch):
    sentinel = _FakeDetector("originality", 0.7)
    import humanizer.detectors.judge as jmod
    monkeypatch.setattr(
        jmod, "_build_paid_detector",
        lambda name: sentinel if name == "originality" else None,
    )

    j = judge_from_env(env={"ORIGINALITY_API_KEY": "k"})
    assert j is sentinel  # bare detector, no ensemble overhead


def test_judge_from_env_multiple_keys_returns_ensemble(monkeypatch):
    fakes = {
        "originality": _FakeDetector("originality", 0.4),
        "pangram": _FakeDetector("pangram", 0.6),
        "gptzero": _FakeDetector("gptzero", 0.8),
    }
    import humanizer.detectors.judge as jmod
    monkeypatch.setattr(jmod, "_build_paid_detector", lambda name: fakes[name])

    j = judge_from_env(env={
        "ORIGINALITY_API_KEY": "a",
        "PANGRAM_API_KEY": "b",
        "GPTZERO_API_KEY": "c",
    })
    assert isinstance(j, EnsembleJudge)
    # Mean of 0.4, 0.6, 0.8 = 0.6
    assert j.score("x") == pytest.approx(0.6)
    assert set(j.last_breakdown.keys()) == {"originality", "pangram", "gptzero"}
    assert "ensemble" in j.name


def test_judge_from_env_prefer_filters(monkeypatch):
    fakes = {
        "originality": _FakeDetector("originality", 0.4),
        "pangram": _FakeDetector("pangram", 0.6),
    }
    import humanizer.detectors.judge as jmod
    monkeypatch.setattr(jmod, "_build_paid_detector", lambda name: fakes[name])

    # Both keys present, but prefer only "pangram" → bare pangram detector.
    j = judge_from_env(
        env={"ORIGINALITY_API_KEY": "a", "PANGRAM_API_KEY": "b"},
        prefer=["pangram"],
    )
    assert j.name == "pangram"
    assert j.score("x") == pytest.approx(0.6)


def test_ensemble_judge_works_inside_rejection_sampler(monkeypatch):
    """End-to-end: ensemble judge integrates with the rejection sampler
    via the standard Detector contract — no rejection-sampler change
    needed."""
    import numpy as np
    from humanizer.humanizers.base import HumanizeResult, Humanizer
    from humanizer.humanizers.rejection import (
        RejectionConfig, RejectionSamplingHumanizer,
    )

    class _StubBase(Humanizer):
        name = "stub"
        def humanize(self, text, **_): return HumanizeResult(original=text, text="x")
        def sample(self, text, n, *, temperature=None, top_p=None):
            return ["good"] * n

    monkeypatch.setattr(
        "humanizer.humanizers.rejection.embedding_similarity",
        lambda a, b, *, model_id=None: np.array([1.0] * len(a), dtype=np.float32),
    )

    ens = DetectorEnsemble([
        _FakeDetector("d1", 0.01),
        _FakeDetector("d2", 0.02),
    ])
    judge = EnsembleJudge(ens, name="ens")

    base = _StubBase()
    h = RejectionSamplingHumanizer(
        base, judge,
        RejectionConfig(candidates_per_round=2, max_rounds=1, p_ai_threshold=0.05),
    )
    out = h.humanize("source")
    assert out.metadata["passed"] is True
    assert out.metadata["judge"] == "ens"
    # Aggregate of 0.01 and 0.02 = 0.015
    assert out.score == pytest.approx(0.015)
