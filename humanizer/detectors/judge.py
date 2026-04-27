"""Judge factory: build the inference-time target detector from whatever
the operator has actually paid for.

Detector evasion is mostly an arms race against specific models. If you
optimize against only one detector, you overfit to it. The right move
at inference time is a small ensemble of REAL detectors — whatever
paid APIs you have keys for. This module:

  - Wraps a DetectorEnsemble as a single Detector (EnsembleJudge), so
    RejectionSamplingHumanizer's single-judge contract still works.
  - Provides judge_from_env(): inspects environment variables for paid-
    API keys (GPTZERO_API_KEY, ORIGINALITY_API_KEY, PANGRAM_API_KEY),
    constructs whichever paid detectors are available, falls back to
    the local RoBERTa-large open detector if none are configured.

Cost-aware ordering: the ensemble queries detectors in a defined order
so cheaper / faster ones run first if the rejection sampler short-
circuits on first failure. (The current rejection sampler scores one
candidate at a time and returns the aggregate, so order matters less,
but it's still a useful invariant.)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .base import Detector
from .ensemble import DetectorEnsemble


@dataclass(frozen=True)
class JudgeSpec:
    """Description of one detector slot in a judge ensemble."""

    name: str
    weight: float
    api_key_env: str | None  # None = local (no env key required)


# Order: cheapest-to-most-expensive paid first, local last as fallback.
_PAID_SPECS: list[JudgeSpec] = [
    JudgeSpec("originality", weight=1.0, api_key_env="ORIGINALITY_API_KEY"),
    JudgeSpec("pangram", weight=1.0, api_key_env="PANGRAM_API_KEY"),
    JudgeSpec("gptzero", weight=1.0, api_key_env="GPTZERO_API_KEY"),
]
_LOCAL_FALLBACK = JudgeSpec("roberta-large", weight=1.0, api_key_env=None)


class EnsembleJudge(Detector):
    """Adapter: presents a DetectorEnsemble as a single Detector, returning
    the weighted aggregate p_ai. After each .score() call, .last_breakdown
    holds the per-detector p_ai map. Single-threaded use only.
    """

    def __init__(self, ensemble: DetectorEnsemble, name: str = "ensemble"):
        self.ensemble = ensemble
        self.name = name
        self.last_breakdown: dict[str, float] = {}

    def score(self, text: str) -> float:
        result = self.ensemble.score(text)
        self.last_breakdown = result.by_name
        return float(result.aggregate)


def _build_paid_detector(name: str) -> Detector:
    if name == "originality":
        from .originality import OriginalityDetector
        return OriginalityDetector()
    if name == "pangram":
        from .pangram import PangramDetector
        return PangramDetector()
    if name == "gptzero":
        from .gptzero import GPTZeroDetector
        return GPTZeroDetector()
    raise ValueError(f"unknown paid detector: {name!r}")


def _build_local_fallback() -> Detector:
    from .roberta import RoBERTaDetector
    return RoBERTaDetector("roberta-large-openai-detector")


def available_paid_detectors(env: dict[str, str] | None = None) -> list[JudgeSpec]:
    """Return the subset of paid detectors whose API keys are present in env."""
    env = env or dict(os.environ)
    return [s for s in _PAID_SPECS if s.api_key_env and env.get(s.api_key_env)]


def judge_from_env(
    *,
    prefer: list[str] | None = None,
    fallback_to_local: bool = True,
    env: dict[str, str] | None = None,
) -> Detector:
    """Construct an inference-time judge from whatever's available.

    Returns:
      - EnsembleJudge wrapping all paid detectors with valid keys, OR
      - a single paid detector if exactly one key is present, OR
      - the local RoBERTa-large detector if nothing else is configured
        (and fallback_to_local is True), OR
      - raises RuntimeError if nothing is configured and fallback_to_local
        is False.

    Args:
      prefer: optional ordered subset of detector names to consider. If
        provided, only these detectors are checked. Useful to scope down
        to e.g. ["originality"] when you have multiple keys but want one.
      fallback_to_local: if True, return a roberta-large local detector
        when no paid keys are present.
      env: dict to use instead of os.environ (testing).
    """
    env = env or dict(os.environ)
    specs = _PAID_SPECS
    if prefer is not None:
        prefer_set = set(prefer)
        specs = [s for s in _PAID_SPECS if s.name in prefer_set]

    available = [s for s in specs if s.api_key_env and env.get(s.api_key_env)]

    if not available:
        if fallback_to_local:
            return _build_local_fallback()
        raise RuntimeError(
            "judge_from_env: no paid-detector API keys are set "
            f"(checked {[s.api_key_env for s in specs]}). "
            "Set ORIGINALITY_API_KEY, PANGRAM_API_KEY, or GPTZERO_API_KEY, "
            "or pass fallback_to_local=True for the free local judge."
        )

    if len(available) == 1:
        # Single judge — return the bare Detector (cheaper, no ensemble overhead).
        return _build_paid_detector(available[0].name)

    # Multi-detector ensemble.
    detectors = [_build_paid_detector(s.name) for s in available]
    weights = [s.weight for s in available]
    ensemble = DetectorEnsemble(detectors, weights=weights)
    label = "+".join(s.name for s in available)
    return EnsembleJudge(ensemble, name=f"ensemble({label})")
