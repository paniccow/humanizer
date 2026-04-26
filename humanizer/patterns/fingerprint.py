"""Aggregate fingerprint — runs every signal and returns a structured report.

Use cases:
  1. Explain *why* the detector ensemble flagged something.
  2. As an auxiliary reward channel: penalize text that lights up many signals.
  3. As a teaching tool — print the fingerprint to learn what AI looks like.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import signals as S


@dataclass
class Fingerprint:
    """Bag of AI-tell scores. Each in [0, 1]; higher = more AI-like."""

    burstiness: float
    stiff_transitions: float
    favorite_words: float
    em_dash_density: float
    hedging: float
    tricolons: float
    contraction_deficit: float
    ngram_repetition: float
    type_token: float
    sentence_start_uniformity: float
    aggregate: float = 0.0
    flagged: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "aggregate": self.aggregate,
            "burstiness": self.burstiness,
            "stiff_transitions": self.stiff_transitions,
            "favorite_words": self.favorite_words,
            "em_dash_density": self.em_dash_density,
            "hedging": self.hedging,
            "tricolons": self.tricolons,
            "contraction_deficit": self.contraction_deficit,
            "ngram_repetition": self.ngram_repetition,
            "type_token": self.type_token,
            "sentence_start_uniformity": self.sentence_start_uniformity,
            "flagged": self.flagged,
        }

    def explain(self) -> str:
        order = [
            ("burstiness", "uniform sentence lengths"),
            ("stiff_transitions", "stiff transitional phrases (Furthermore, Moreover, ...)"),
            ("favorite_words", "AI-favorite vocabulary (delve, leverage, intricate, ...)"),
            ("em_dash_density", "em-dash overuse"),
            ("hedging", "hedging boilerplate (It's important to note ...)"),
            ("tricolons", "tricolons (X, Y, and Z)"),
            ("contraction_deficit", "lacks contractions (uses 'do not' instead of 'don't')"),
            ("ngram_repetition", "repeats 4-grams"),
            ("type_token", "low vocabulary diversity"),
            ("sentence_start_uniformity", "sentences start the same way"),
        ]
        lines = [f"Aggregate AI-likeness: {self.aggregate:.2f}"]
        for key, desc in order:
            v = getattr(self, key)
            bar = "█" * int(v * 20)
            lines.append(f"  {v:.2f} {bar:<20s}  {desc}")
        if self.flagged:
            lines.append("Flagged: " + ", ".join(self.flagged))
        return "\n".join(lines)


# Weighting reflects how strongly each signal correlates with detector verdicts
# in our HC3 calibration (higher = more important).
_WEIGHTS = {
    "burstiness": 1.5,
    "stiff_transitions": 1.3,
    "favorite_words": 1.6,
    "em_dash_density": 0.8,
    "hedging": 1.2,
    "tricolons": 0.7,
    "contraction_deficit": 1.0,
    "ngram_repetition": 0.7,
    "type_token": 0.6,
    "sentence_start_uniformity": 0.7,
}
_FLAG_THRESHOLD = 0.6


def analyze(text: str) -> Fingerprint:
    fp = Fingerprint(
        burstiness=S.burstiness_score(text),
        stiff_transitions=S.stiff_transition_score(text),
        favorite_words=S.favorite_word_density(text),
        em_dash_density=S.em_dash_density_score(text),
        hedging=S.hedging_phrase_score(text),
        tricolons=S.tricolon_density_score(text),
        contraction_deficit=S.contraction_deficit_score(text),
        ngram_repetition=S.ngram_repetition_score(text),
        type_token=S.type_token_ratio_score(text),
        sentence_start_uniformity=S.sentence_start_uniformity_score(text),
    )
    total_w = sum(_WEIGHTS.values())
    agg = sum(getattr(fp, k) * w for k, w in _WEIGHTS.items()) / total_w
    fp.aggregate = float(agg)
    fp.flagged = [k for k in _WEIGHTS if getattr(fp, k) >= _FLAG_THRESHOLD]
    return fp
