"""End-to-end evaluation harness.

For a list of (source) AI texts, runs a Humanizer and reports:
  - per-detector ASR (attack-success-rate)  — fraction of outputs that fool each detector
  - mean p_ai across the ensemble
  - semantic similarity to the source (MiniLM cosine)
  - perplexity (GPT-2)
  - sentence burstiness (CV of sentence lengths)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .detectors.ensemble import DetectorEnsemble
from .humanizers.base import Humanizer
from .metrics.burstiness import sentence_length_stats
from .metrics.quality import perplexity
from .metrics.semantic import embedding_similarity


@dataclass
class EvalReport:
    n: int
    asr_per_detector: dict[str, float]
    asr_ensemble: float                # fraction with mean p_ai < 0.5
    mean_p_ai: float
    mean_similarity: float
    mean_perplexity: float
    mean_burstiness_cv: float
    samples: list[dict]                # per-example detail for inspection

    def summary(self) -> str:
        lines = [
            f"n={self.n}",
            f"mean_p_ai={self.mean_p_ai:.3f}",
            f"asr_ensemble={self.asr_ensemble:.2%}",
            "asr_per_detector:",
        ]
        for name, val in self.asr_per_detector.items():
            lines.append(f"  {name:<50s} {val:.2%}")
        lines.extend(
            [
                f"mean_similarity={self.mean_similarity:.3f}",
                f"mean_perplexity={self.mean_perplexity:.1f}",
                f"mean_burstiness_cv={self.mean_burstiness_cv:.2f}",
            ]
        )
        return "\n".join(lines)


def evaluate(
    humanizer: Humanizer,
    sources: list[str],
    detectors: DetectorEnsemble,
    similarity_threshold: float = 0.78,
    compute_perplexity: bool = True,
) -> EvalReport:
    results = humanizer.humanize_batch(sources)
    outputs = [r.text for r in results]

    ensemble_results = detectors.score_batch(outputs)
    mean_p_ai = statistics.fmean(r.aggregate for r in ensemble_results)
    asr_ensemble = sum(1 for r in ensemble_results if r.aggregate < 0.5) / max(len(outputs), 1)
    asr_per = detectors.asr(outputs)

    sims = embedding_similarity(sources, outputs).tolist()
    mean_sim = statistics.fmean(sims) if sims else 0.0

    if compute_perplexity:
        perps = [perplexity(t) for t in outputs]
        mean_perp = statistics.fmean(perps) if perps else 0.0
    else:
        mean_perp = 0.0

    cvs = [sentence_length_stats(t).cv_words for t in outputs]
    mean_cv = statistics.fmean(cvs) if cvs else 0.0

    samples = []
    for src, out, er, sim in zip(sources, outputs, ensemble_results, sims):
        samples.append(
            {
                "source": src,
                "humanized": out,
                "p_ai": er.aggregate,
                "per_detector": er.by_name,
                "similarity": sim,
                "burstiness_cv": sentence_length_stats(out).cv_words,
            }
        )

    return EvalReport(
        n=len(outputs),
        asr_per_detector=asr_per,
        asr_ensemble=asr_ensemble,
        mean_p_ai=mean_p_ai,
        mean_similarity=mean_sim,
        mean_perplexity=mean_perp,
        mean_burstiness_cv=mean_cv,
        samples=samples,
    )
