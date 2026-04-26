"""Adversarial humanizer — training-free best-of-N with detector-as-judge.

Reproduces the recipe from "Adversarial Paraphrasing: A Universal Attack for
Humanizing AI-Generated Text" (arXiv:2506.07001, 2025) and the inference
sampling used in AuthorMist (arXiv:2503.08716, 2025).

Procedure:
  1. Sample N candidate paraphrases from a base humanizer (LLM with prompt).
  2. Score each with the detector ensemble.
  3. Filter for semantic similarity ≥ threshold against the original.
  4. Return the candidate with the lowest mean detector p_ai.

This works without any training and is the recommended baseline before
spinning up GRPO. It also serves as the inference-time wrapper around the
trained model.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..detectors.ensemble import DetectorEnsemble
from ..metrics.semantic import embedding_similarity
from .base import HumanizeResult, Humanizer
from .prompt import PromptHumanizer


@dataclass
class AdversarialConfig:
    n_candidates: int = 8
    similarity_threshold: float = 0.78          # cosine sim with MiniLM, AuthorMist uses 0.94 with E5
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_to_best: bool = True               # if no candidate passes sim threshold, return lowest-p_ai anyway
    max_passes: int = 1                         # iterate humanization (each pass uses prior best as input)


class AdversarialHumanizer(Humanizer):
    """Best-of-N humanizer guided by an ensemble of AI-text detectors."""

    name = "adversarial"

    def __init__(
        self,
        base: Humanizer,
        detectors: DetectorEnsemble,
        config: AdversarialConfig | None = None,
    ):
        if not hasattr(base, "sample"):
            raise TypeError(
                f"{type(base).__name__} must implement .sample(text, n) -> list[str]"
            )
        self.base = base
        self.detectors = detectors
        self.config = config or AdversarialConfig()

    def _select(self, original: str, candidates: list[str]) -> tuple[str, float, dict]:
        """Score candidates, filter by semantic sim, return (best_text, p_ai, meta)."""
        ensemble_results = self.detectors.score_batch(candidates)
        sims = embedding_similarity(
            [original] * len(candidates),
            candidates,
            model_id=self.config.similarity_model,
        ).tolist()
        scored = list(zip(candidates, [r.aggregate for r in ensemble_results], sims, ensemble_results))
        # First, only consider candidates that preserve meaning.
        kept = [c for c in scored if c[2] >= self.config.similarity_threshold]
        pool = kept or (scored if self.config.fallback_to_best else [])
        if not pool:
            raise RuntimeError("All candidates failed the semantic similarity threshold.")
        # Choose lowest p_ai among the kept set.
        best = min(pool, key=lambda c: c[1])
        text, p_ai, sim, ens = best
        return text, p_ai, {
            "similarity": sim,
            "candidates_kept": len(kept),
            "candidates_total": len(candidates),
            "per_detector": ens.by_name,
        }

    def humanize(self, text: str, **_) -> HumanizeResult:
        current = text
        attempts = 0
        last_p_ai = None
        last_meta: dict = {}
        for _ in range(self.config.max_passes):
            cands = self.base.sample(current, n=self.config.n_candidates)
            attempts += len(cands)
            best_text, p_ai, meta = self._select(text, cands)
            current = best_text
            last_p_ai = p_ai
            last_meta = meta
            # Early-out if we've already convinced the ensemble.
            if p_ai < 0.1:
                break
        return HumanizeResult(
            original=text,
            text=current,
            score=last_p_ai,
            attempts=attempts,
            metadata=last_meta,
        )
