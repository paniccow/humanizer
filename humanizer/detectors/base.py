from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DetectorScore:
    """A single detector's verdict on a piece of text.

    `p_ai` is the probability the detector assigns to the text being AI-generated,
    in [0, 1]. Lower is better for a humanizer.
    """

    name: str
    p_ai: float

    @property
    def p_human(self) -> float:
        return 1.0 - self.p_ai


class Detector(ABC):
    """Abstract AI-text detector. Implementations return p(AI | text)."""

    name: str = "base"

    @abstractmethod
    def score(self, text: str) -> float:
        """Return p(AI | text) in [0, 1]."""

    def score_batch(self, texts: list[str]) -> list[float]:
        return [self.score(t) for t in texts]

    def detect(self, text: str) -> DetectorScore:
        return DetectorScore(name=self.name, p_ai=float(self.score(text)))


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    import torch  # deferred so the rest of this module is torch-free
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
