"""Fact-preservation metric — pure-python, no models.

Extracts a "verifiable surface" from source text:
  - numbers (123, 4.5, 1/2)
  - currency ($5,000, USD 5000, €500)
  - percentages (25%, 1.4 percent)
  - years (1947, 2026 — distinct from generic numbers)
  - dates (Jan 5, 2024 — month-day-year fragments)
  - capitalized multi-word tokens (proper-noun-ish: "World War II", "OpenAI",
    "Sarah Chen") — heuristic, not a real NER

Then `entity_overlap(original, candidate)` returns the fraction of
extracted tokens that survive verbatim (or near-verbatim, for numbers).
A humanization that drops "$5,000" or changes "1947" to "1948" gets
flagged.

Used as an optional gate in RejectionSamplingHumanizer: candidates
with overlap below `preservation_threshold` are dropped before judge
scoring. False positives are common (the humanizer might paraphrase
"World War II" as "the second world war") — that's why this is
opt-in, not on by default. For high-stakes contexts (academic essays,
news articles) where number/date integrity matters, turn it on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Order matters — higher-priority kinds run first. Their spans are
# claimed and lower-priority kinds skip overlapping matches. Date BEFORE
# year/number so "January 5, 2024" wins over "5" + "2024".
_PATTERNS = [
    # Currency with symbol prefix or ISO code
    ("currency", re.compile(r"(?:\$|€|£|¥|USD|EUR|GBP|JPY)\s?\d[\d,]*(?:\.\d+)?", re.I)),
    # Percentages: % without \b, "percent" with \b
    ("pct", re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*percent\b", re.I)),
    # Dates like "Jan 5, 2024" / "January 5" — claim before year/number split it.
    ("date", re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|"
        r"Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2}(?:,\s+\d{4})?\b",
        re.I,
    )),
    # Years standalone — 1700-2099
    ("year", re.compile(r"\b(?:1[7-9]\d{2}|20\d{2})\b")),
    # Numbers (integers + decimals + comma-separated thousands)
    ("number", re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")),
    # Capitalized words/phrases (heuristic proper noun). Two passes:
    # 1. Multi-word: "World War II", "Sarah Chen", "New York"
    # 2. Single-word with internal caps or all-caps: "OpenAI", "GPT", "Anthropic"
    # Single sentence-initial words ("The", "When") will false-positive — that's
    # the cost of a regex-only NER. Acceptable for an opt-in preservation gate.
    ("proper_noun", re.compile(
        r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+"  # multi-word
        r"|\b(?:[A-Z]{2,}|[A-Z][a-z]+[A-Z][a-zA-Z]*|[A-Z][a-z]{2,})\b"  # single
    )),
]


@dataclass
class FactSet:
    """Bag of canonicalized facts extracted from text. Comparable across
    humanizations to detect drops/changes."""

    by_kind: dict[str, set[str]]

    @property
    def all(self) -> set[str]:
        out: set[str] = set()
        for s in self.by_kind.values():
            out |= s
        return out

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_kind.values())


def _normalize_number(s: str) -> str:
    """Strip thousands commas, lower-case currency prefix."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower().replace(",", "")


def extract_facts(text: str) -> FactSet:
    """Return a FactSet of all matched entities from `text`.

    Numbers are normalized (commas stripped, lowercased) so "$5,000" and
    "$5000" count as the same fact.
    """
    out: dict[str, set[str]] = {k: set() for k, _ in _PATTERNS}
    consumed: set[tuple[int, int]] = set()
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            span = m.span()
            # Skip overlaps with already-matched higher-priority kinds
            if any(_overlaps(span, c) for c in consumed):
                continue
            consumed.add(span)
            tok = m.group(0)
            if kind in ("number", "currency", "pct", "year", "date"):
                out[kind].add(_normalize_number(tok))
            else:
                out[kind].add(tok)
    return FactSet(by_kind=out)


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def entity_overlap(original: str, candidate: str) -> float:
    """Fraction of facts in `original` that also appear in `candidate`.

    Returns 1.0 if the original has no extractable facts (nothing to
    preserve, so trivially "preserved"). Numbers compared after
    normalization so commas/case don't trip us. Proper nouns compared
    case-insensitively.
    """
    src = extract_facts(original)
    if not src.all:
        return 1.0
    cand_text_lower = candidate.lower()
    # Strip thousands-comma separators in the candidate so "5,000" matches "5000".
    cand_text_normalized = re.sub(r"(\d),(\d)", r"\1\2", cand_text_lower)
    cand = extract_facts(candidate)

    # For each source fact, check it appears either as an extracted entity
    # in the candidate OR substring-matches in the candidate text. The latter
    # catches cases where the candidate's regex match shape differs (e.g.
    # "$5,000" -> "5,000 dollars") but the value is still present.
    preserved = 0
    for kind, facts in src.by_kind.items():
        for f in facts:
            cand_facts = cand.by_kind.get(kind, set())
            if f in cand_facts:
                preserved += 1
                continue
            # substring fallback — strip the currency prefix etc. and look
            # for the bare number in the candidate text (also comma-stripped).
            if kind in ("currency", "number", "year", "pct"):
                bare = re.sub(r"[^\d.]+", "", f)
                if bare and bare in cand_text_normalized:
                    preserved += 1
                    continue
            if kind == "proper_noun" and f.lower() in cand_text_lower:
                preserved += 1

    return preserved / max(len(src.all), 1)
