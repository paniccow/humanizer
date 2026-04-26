"""Factory for the default detector ensemble. Imported lazily to keep `detectors`
importable on machines without transformers."""
from __future__ import annotations

from .ensemble import DetectorEnsemble
from .roberta import RoBERTaDetector


def default_ensemble(device: str | None = None, lite: bool = False) -> DetectorEnsemble:
    """A reasonable default detector ensemble.

    `lite=True` returns a single small detector — fits on a CPU/Mac for quick eval.
    `lite=False` loads three detectors with diverse architectures (RoBERTa-base,
    RoBERTa-large, DeBERTa-v3-large) so the reward signal is harder to game.
    """
    if lite:
        return DetectorEnsemble(
            [RoBERTaDetector("openai-community/roberta-base-openai-detector", device=device)]
        )
    return DetectorEnsemble(
        [
            RoBERTaDetector("openai-community/roberta-base-openai-detector", device=device),
            RoBERTaDetector("openai-community/roberta-large-openai-detector", device=device),
            RoBERTaDetector("desklib/ai-text-detector-v1.01", device=device),
        ]
    )
