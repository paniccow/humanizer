"""Burstiness statistics — the 'shape' of human writing detectors look for.

Two ingredients in the standard detector recipe (GPTZero etc.):
  - perplexity: how typical the text is under a reference LM (low ≈ AI)
  - burstiness: variance of *sentence-level* perplexity / length (low ≈ AI)

Humans mix short and long sentences; AI tends to produce ~15-word sentences
of consistent surprise. We measure both so post-processing can target them.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class BurstinessStats:
    n_sentences: int
    mean_words: float
    std_words: float          # higher = more burstiness in length
    cv_words: float           # coefficient of variation = std/mean
    min_words: int
    max_words: int

    def as_dict(self) -> dict[str, float]:
        return {
            "n_sentences": self.n_sentences,
            "mean_words": self.mean_words,
            "std_words": self.std_words,
            "cv_words": self.cv_words,
            "min_words": self.min_words,
            "max_words": self.max_words,
        }


def sentence_length_stats(text: str) -> BurstinessStats:
    sents = split_sentences(text)
    if not sents:
        return BurstinessStats(0, 0.0, 0.0, 0.0, 0, 0)
    counts = [len(s.split()) for s in sents]
    n = len(counts)
    mean = sum(counts) / n
    var = sum((c - mean) ** 2 for c in counts) / n
    std = math.sqrt(var)
    cv = std / mean if mean > 0 else 0.0
    return BurstinessStats(
        n_sentences=n,
        mean_words=mean,
        std_words=std,
        cv_words=cv,
        min_words=min(counts),
        max_words=max(counts),
    )
