"""AI-pattern fingerprints — what every detector actually looks for.

This module encodes, in code, the *signals* that real AI-text detectors use:

  - Perplexity (GPTZero, Originality, every classifier-based detector)
  - Burstiness (GPTZero — sentence-length variance)
  - AI-favorite vocabulary (Turnitin, Copyleaks)
  - Stiff transitional phrases ("Furthermore,", "Moreover,", ...)
  - Em-dash overuse (recent ChatGPT/Claude tell)
  - Hedging boilerplate ("It is important to note that ...")
  - Tricolons (X, Y, and Z — AI overuses)
  - Lack of contractions / personal voice
  - Predictable conclusion phrases
  - N-gram repetition + low type-token ratio
  - Sentence-start uniformity
  - Watermark-friendly token patterns (heuristic only)

Each pattern is exposed as a checker `score(text) -> [0, 1]` (higher = more AI-like
on that axis). `analyze(text)` runs the full suite and returns a Fingerprint.

These can be wired into a reward function or used post-hoc to explain WHY a
piece of text reads as AI.
"""
from .fingerprint import Fingerprint, analyze
from .signals import (
    AI_FAVORITE_WORDS,
    AI_HEDGING_PHRASES,
    AI_TRANSITIONS,
    burstiness_score,
    contraction_deficit_score,
    em_dash_density_score,
    favorite_word_density,
    hedging_phrase_score,
    ngram_repetition_score,
    sentence_start_uniformity_score,
    stiff_transition_score,
    tricolon_density_score,
    type_token_ratio_score,
)

__all__ = [
    "Fingerprint",
    "analyze",
    "AI_FAVORITE_WORDS",
    "AI_TRANSITIONS",
    "AI_HEDGING_PHRASES",
    "burstiness_score",
    "stiff_transition_score",
    "favorite_word_density",
    "em_dash_density_score",
    "hedging_phrase_score",
    "tricolon_density_score",
    "contraction_deficit_score",
    "ngram_repetition_score",
    "type_token_ratio_score",
    "sentence_start_uniformity_score",
]
