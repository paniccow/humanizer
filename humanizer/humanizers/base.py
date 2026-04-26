from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class HumanizeResult:
    """Output of a Humanizer.

    `original`  — input text
    `text`      — humanized output
    `score`     — final detector-ensemble p_ai (lower = more human)
    `attempts`  — for best-of-N humanizers, how many candidates were sampled
    `metadata`  — anything else worth keeping (per-detector scores, similarity, etc.)
    """

    original: str
    text: str
    score: float | None = None
    attempts: int = 1
    metadata: dict | None = None


class Humanizer(ABC):
    name: str = "base"

    @abstractmethod
    def humanize(self, text: str, **kwargs) -> HumanizeResult: ...

    def humanize_batch(self, texts: list[str], **kwargs) -> list[HumanizeResult]:
        return [self.humanize(t, **kwargs) for t in texts]
