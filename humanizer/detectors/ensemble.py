from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .base import Detector, DetectorScore


@dataclass
class EnsembleResult:
    """Aggregated output from a DetectorEnsemble."""

    scores: list[DetectorScore] = field(default_factory=list)
    aggregate: float = 0.0  # mean p_ai across detectors

    @property
    def by_name(self) -> dict[str, float]:
        return {s.name: s.p_ai for s in self.scores}

    def __repr__(self) -> str:
        parts = ", ".join(f"{s.name}={s.p_ai:.3f}" for s in self.scores)
        return f"EnsembleResult(mean_p_ai={self.aggregate:.3f}, {parts})"


class DetectorEnsemble:
    """Run multiple AI-text detectors and aggregate their probabilities.

    Used for two purposes:
      1. Reward signal during GRPO training (lower mean p_ai => higher reward).
      2. Evaluation: report attack-success-rate (ASR) per detector.
    """

    def __init__(
        self,
        detectors: Iterable[Detector],
        weights: list[float] | None = None,
        threshold: float = 0.5,
    ):
        self.detectors: list[Detector] = list(detectors)
        if not self.detectors:
            raise ValueError("DetectorEnsemble needs at least one Detector")
        if weights is None:
            self.weights = [1.0] * len(self.detectors)
        else:
            if len(weights) != len(self.detectors):
                raise ValueError("len(weights) must equal len(detectors)")
            self.weights = list(weights)
        self.threshold = threshold

    def score(self, text: str) -> EnsembleResult:
        scores = [d.detect(text) for d in self.detectors]
        total_w = sum(self.weights) or 1.0
        agg = sum(s.p_ai * w for s, w in zip(scores, self.weights)) / total_w
        return EnsembleResult(scores=scores, aggregate=agg)

    def score_batch(self, texts: list[str]) -> list[EnsembleResult]:
        # Each detector batches internally; we transpose at the end.
        per_detector: list[list[float]] = [d.score_batch(texts) for d in self.detectors]
        total_w = sum(self.weights) or 1.0
        results: list[EnsembleResult] = []
        for i in range(len(texts)):
            scores = [
                DetectorScore(name=d.name, p_ai=per_detector[j][i])
                for j, d in enumerate(self.detectors)
            ]
            agg = sum(s.p_ai * w for s, w in zip(scores, self.weights)) / total_w
            results.append(EnsembleResult(scores=scores, aggregate=agg))
        return results

    def reward(self, text: str) -> float:
        """Reward = 1 - mean(p_ai). Higher when text reads as more human."""
        return 1.0 - self.score(text).aggregate

    def reward_batch(self, texts: list[str]) -> list[float]:
        return [1.0 - r.aggregate for r in self.score_batch(texts)]

    def asr(self, texts: list[str]) -> dict[str, float]:
        """Attack-success-rate per detector. Fraction of texts where p_ai < threshold."""
        per_detector: list[list[float]] = [d.score_batch(texts) for d in self.detectors]
        out: dict[str, float] = {}
        n = max(len(texts), 1)
        for d, ps in zip(self.detectors, per_detector):
            out[d.name] = sum(1 for p in ps if p < self.threshold) / n
        return out
