"""Individual AI-tell signals.

Each `*_score` function returns a value in roughly [0, 1] where higher = more
AI-like on that axis. Magnitudes are calibrated against typical human writing
samples (news, Reddit, fiction) and ChatGPT/Claude outputs.

References:
  - GPTZero perplexity & burstiness primer:
    https://gptzero.me/news/perplexity-and-burstiness-what-is-it/
  - Originality.ai detector breakdowns
  - Empirical word-frequency studies of GPT-3.5/4 outputs
    (e.g. "Has ChatGPT been gradually changing the way you write?")
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..metrics.burstiness import split_sentences

# ---------- Canonical AI-tell vocabularies ----------

# Words that appear at 5-30x the human-baseline rate in ChatGPT/Claude outputs
# across blog, essay, and answer-style text.
AI_FAVORITE_WORDS: frozenset[str] = frozenset(
    {
        # Archaic-formal markers (essentially never in casual human prose)
        "whereby", "therein", "hereby", "thereof", "thereto",
        "henceforth", "hitherto", "wherein", "whereupon",
        # Verbs
        "delve", "delves", "delved", "delving",
        "leverage", "leverages", "leveraged", "leveraging",
        "navigate", "navigates", "navigated", "navigating",
        "embark", "embarks", "embarked", "embarking",
        "foster", "fosters", "fostered", "fostering",
        "underscore", "underscores", "underscored", "underscoring",
        "showcase", "showcases", "showcased", "showcasing",
        "garner", "garners", "garnered", "garnering",
        # Adjectives
        "intricate", "multifaceted", "paramount", "crucial", "pivotal",
        "robust", "seamless", "comprehensive", "holistic", "nuanced",
        "innovative", "cutting-edge", "transformative", "groundbreaking",
        "myriad", "vibrant", "bustling", "remarkable",
        # Nouns
        "tapestry", "realm", "plethora", "landscape", "ecosystem",
        "paradigm", "synergy", "endeavor", "endeavour",
        "testament", "cornerstone", "framework",
        # Adverbs
        "moreover", "furthermore", "additionally", "consequently",
        "indeed", "essentially", "fundamentally", "particularly",
    }
)

# Stiff sentence-initial transitions — humans rarely string several together.
AI_TRANSITIONS: frozenset[str] = frozenset(
    {
        "Furthermore,", "Moreover,", "Additionally,", "In addition,",
        "Consequently,", "Thus,", "Therefore,", "Hence,",
        "However,", "Nevertheless,", "Nonetheless,",
        "In conclusion,", "To conclude,", "In summary,", "Overall,",
        "To summarize,", "Ultimately,", "All in all,",
    }
)

# Hedging boilerplate that AI loves. Each match counts as one signal.
AI_HEDGING_PHRASES: tuple[str, ...] = (
    "it is important to note that",
    "it's important to note that",
    "it is worth noting that",
    "it's worth noting that",
    "it is essential to",
    "it should be noted that",
    "in today's world",
    "in today's society",
    "in today's rapidly evolving",
    "navigating the complexities of",
    "the ever-changing landscape",
    "play a crucial role",
    "a wide range of",
    "a variety of",
    "delve into",
    "shed light on",
    "stand the test of time",
    "in the realm of",
    "when it comes to",
    "at the end of the day",
    "it's worth mentioning",
    "it is worth mentioning",
    "it bears repeating",
    "it's important to remember",
    "it is important to remember",
    "needless to say",
    "as previously mentioned",
    "as we delve",
    "in the grand scheme",
    "with that being said",
    "that being said",
)

_CONTRACTIONS_RE = re.compile(
    r"\b("
    r"don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"can't|won't|wouldn't|couldn't|shouldn't|"
    r"hasn't|haven't|hadn't|"
    r"it's|that's|what's|there's|here's|"
    r"you're|they're|we're|i'm|i've|i'd|i'll|"
    r"you've|you'd|you'll|they've|they'd|they'll|we've|we'd|we'll"
    r")\b",
    re.IGNORECASE,
)
_EXPANDABLE_PAIRS_RE = re.compile(
    r"\b(do|does|did|is|are|was|were|can|will|would|could|should|"
    r"has|have|had|it|that|what|there|here|you|they|we|I) "
    r"(not|is|are|am|have|has|had|will|would|d)\b",
    re.IGNORECASE,
)


def _logistic(x: float, midpoint: float, steep: float = 8.0) -> float:
    """Squash a raw count into [0, 1]; midpoint = "moderately suspicious"."""
    return 1.0 / (1.0 + math.exp(-steep * (x - midpoint)))


# ---------- Signals ----------

def burstiness_score(text: str) -> float:
    """Lower sentence-length variance => more AI. Returns 1.0 for very flat text."""
    sents = split_sentences(text)
    if len(sents) < 3:
        return 0.5  # too short to judge
    counts = [len(s.split()) for s in sents]
    n = len(counts)
    mean = sum(counts) / n
    var = sum((c - mean) ** 2 for c in counts) / n
    cv = math.sqrt(var) / mean if mean > 0 else 0.0
    # Human writing CV ~ 0.5-0.9. AI ~ 0.1-0.35. Map: low CV -> high score.
    return float(max(0.0, min(1.0, 1.0 - (cv / 0.7))))


def stiff_transition_score(text: str) -> float:
    """Density of stiff sentence-initial transitions per 100 words."""
    words = max(len(text.split()), 1)
    hits = sum(text.count(t) for t in AI_TRANSITIONS)
    per_100 = hits * 100.0 / words
    return _logistic(per_100, midpoint=0.8)


def favorite_word_density(text: str) -> float:
    """Density of AI-favorite vocabulary per 100 words."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in AI_FAVORITE_WORDS)
    per_100 = hits * 100.0 / len(tokens)
    # Human baseline ~0.1-0.3 per 100; ChatGPT often 1-3 per 100.
    return _logistic(per_100, midpoint=1.0)


def em_dash_density_score(text: str) -> float:
    """Em-dashes per 100 words. Recent ChatGPT/Claude tell."""
    words = max(len(text.split()), 1)
    em = text.count("—") + text.count("--")
    per_100 = em * 100.0 / words
    return _logistic(per_100, midpoint=0.7)


def hedging_phrase_score(text: str) -> float:
    """Hits per 100 words of AI hedging boilerplate."""
    words = max(len(text.split()), 1)
    lower = text.lower()
    hits = sum(lower.count(p) for p in AI_HEDGING_PHRASES)
    per_100 = hits * 100.0 / words
    return _logistic(per_100, midpoint=0.4)


def tricolon_density_score(text: str) -> float:
    """`X, Y, and Z` constructions per 100 words. AI overuses."""
    words = max(len(text.split()), 1)
    # Match "<word>, <word>, and <word>"
    hits = len(re.findall(r"\b\w+,\s+\w+,\s+(?:and|or)\s+\w+", text))
    per_100 = hits * 100.0 / words
    return _logistic(per_100, midpoint=0.8)


def contraction_deficit_score(text: str) -> float:
    """Higher when text has contraction-eligible phrases but doesn't contract.
    Requires a minimum sample (≥3 total) so a single 'is not' doesn't max
    out the score on short text — that was a false-positive issue in eval."""
    contracted = len(_CONTRACTIONS_RE.findall(text))
    expandable = len(_EXPANDABLE_PAIRS_RE.findall(text))
    total = contracted + expandable
    if total < 2:                        # min-sample guard (2 instances minimum)
        return 0.0
    rate = expandable / total            # high => AI-style "do not"/"is not"
    return float(rate)


def ngram_repetition_score(text: str, n: int = 4) -> float:
    """Fraction of n-grams that repeat. Above ~0.05 starts to look templated."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())
    if len(tokens) < n + 5:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counter = Counter(grams)
    repeats = sum(c for c in counter.values() if c > 1)
    return min(1.0, repeats / max(len(grams), 1) / 0.15)


def type_token_ratio_score(text: str) -> float:
    """Inverted TTR — lower vocabulary diversity => more AI-like.
    Human prose typically has TTR 0.55-0.75 at this length; AI ~0.4-0.55."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())
    if len(tokens) < 50:
        return 0.5
    ttr = len(set(tokens)) / len(tokens)
    return float(max(0.0, min(1.0, (0.65 - ttr) / 0.25)))


def sentence_start_uniformity_score(text: str) -> float:
    """Repeated sentence-start patterns => AI."""
    sents = split_sentences(text)
    if len(sents) < 4:
        return 0.0
    starts = [tuple(s.split()[:2]) for s in sents if s.split()]
    if not starts:
        return 0.0
    counter = Counter(starts)
    most = counter.most_common(1)[0][1]
    return min(1.0, (most - 1) / max(len(starts) - 1, 1))


# ---- Newer / more specific AI-tell signals ----

# Abstract noun subjects AI loves opening sentences with.
_ABSTRACT_SUBJECTS = (
    "the system", "the framework", "the ecosystem", "the platform",
    "the architecture", "the infrastructure", "the integration",
    "the implementation", "the approach", "the methodology",
    "the foundation", "the cornerstone", "the underlying",
    "the introduction of", "the emergence of", "the proliferation of",
    "the advent of", "the rise of", "the development of",
    "the importance of", "the significance of", "the role of",
    "the evolution of", "the growth of", "the impact of",
)


def abstract_subject_score(text: str) -> float:
    """How often sentences start with an abstract noun subject like 'The system',
    'The framework' — strongly AI-marked, especially in tech/business prose."""
    sents = split_sentences(text)
    if len(sents) < 3:
        return 0.0
    hits = 0
    for s in sents:
        lower = s.lower()
        for sub in _ABSTRACT_SUBJECTS:
            if lower.startswith(sub):
                hits += 1
                break
    rate = hits / len(sents)
    # Human baseline ~0.05-0.10; AI essays often hit 0.30+.
    return _logistic(rate, midpoint=0.20, steep=10)


# AI loves these formulaic enumeration shapes.
_ENUMERATION_PATTERNS = (
    re.compile(r"\b(?:not only|both)\s+\w+(?:\s+\w+){0,3}\s+but\s+also\s+", re.IGNORECASE),
    re.compile(r"\bwhether\s+it'?s\s+\w+(?:\s+\w+){0,4}\s+(?:or|and)\s+", re.IGNORECASE),
    re.compile(r"\bfrom\s+\w+(?:\s+\w+){0,4}\s+to\s+\w+(?:\s+\w+){0,4}\b,", re.IGNORECASE),
    re.compile(r"\bit'?s\s+not\s+(?:just|only)\s+about\s+", re.IGNORECASE),
)


def enumeration_shape_score(text: str) -> float:
    """Density per 100 words of formulaic AI enumeration shapes."""
    words = max(len(text.split()), 1)
    hits = sum(len(p.findall(text)) for p in _ENUMERATION_PATTERNS)
    per_100 = hits * 100.0 / words
    # 1+ hit per 100 words is suspicious; saturate at 2+/100.
    return _logistic(per_100, midpoint=0.7)


_MODAL_VERBS = re.compile(
    r"\b(?:must|should|ought to|needs? to|have to|has to|should not|must not|"
    r"shouldn't|mustn't|need to)\b",
    re.IGNORECASE,
)


def modality_overload_score(text: str) -> float:
    """Density of modal/normative verbs ('must', 'should', 'ought to', 'needs to').
    AI argumentative writing loads up on these; human writing is more declarative."""
    words = max(len(text.split()), 1)
    hits = len(_MODAL_VERBS.findall(text))
    per_100 = hits * 100.0 / words
    # Human baseline ~0.5-1.0/100; AI essays often 2-4/100.
    return _logistic(per_100, midpoint=2.0, steep=4)
