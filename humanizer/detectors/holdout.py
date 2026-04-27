"""Held-out detectors — used ONLY for evaluation, NEVER for training reward.

The point: if our trained policy fools the detectors it was trained against
(roberta-base, roberta-large, etc.) but a fresh detector still flags it as AI,
we know we're overfitting to the training ensemble — not actually producing
human-sounding text. The held-out set is the truth check.

These detectors are intentionally distinct from the training ensemble:
  - SuperAnnotate/roberta-large-llm-content-detector (different training data)
  - coai/roberta-ai-detector-v2 (different team, different corpus)
  - Hello-SimpleAI/chatgpt-detector-roberta (HC3-trained, different distribution)

Binoculars (zero-shot, ICML 2024) is the gold standard for held-out — but it
needs two ~7B LLMs and only runs on a real GPU; loaded separately via
humanizer.detectors.binoculars.Binoculars when CUDA is available.
"""
from __future__ import annotations

from .ensemble import DetectorEnsemble

HOLDOUT_DETECTOR_IDS: tuple[str, ...] = (
    "SuperAnnotate/roberta-large-llm-content-detector",
    "coai/roberta-ai-detector-v2",
    "Hello-SimpleAI/chatgpt-detector-roberta",
)


def holdout_ensemble(
    device: str | None = None,
    detector_ids: tuple[str, ...] = HOLDOUT_DETECTOR_IDS,
    skip_failed: bool = True,
) -> DetectorEnsemble:
    """Construct a held-out detector ensemble.

    Args:
        device: cpu / cuda / mps. Auto-detected if None.
        detector_ids: override the default set.
        skip_failed: if True, log and skip detectors that fail to load
            (state-dict mismatches etc.) rather than crashing. Returns the
            partial ensemble. Useful when the Hub mirror has a half-broken
            checkpoint and you'd rather get *some* signal than none.
    """
    from .roberta import RoBERTaDetector  # deferred — keeps torch out of import-time
    detectors = []
    failed: list[tuple[str, str]] = []
    for det_id in detector_ids:
        try:
            detectors.append(RoBERTaDetector(det_id, device=device))
        except Exception as e:  # noqa: BLE001
            if not skip_failed:
                raise
            failed.append((det_id, str(e)))
    if not detectors:
        msg = "no held-out detectors loaded"
        if failed:
            msg += "; failures: " + "; ".join(f"{d}: {e}" for d, e in failed)
        raise RuntimeError(msg)
    if failed:
        import sys
        for d, e in failed:
            print(f"[holdout] skipped {d}: {e}", file=sys.stderr)
    return DetectorEnsemble(detectors)
