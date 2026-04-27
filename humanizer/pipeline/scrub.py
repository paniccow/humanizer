"""Stage 1: deterministic AI-tell scrubber.

Reverse rules of the patterns module — for every signal that catches AI text,
we have a substitution that removes it. Pure regex, no model call. Runs in
microseconds. Composes safely with any later stage.

Categories:
  - Stiff transitional phrases (Furthermore/Moreover/...) → drop or soft variant
  - AI-favorite vocabulary (delve/leverage/intricate/...) → plain synonyms
  - Hedging boilerplate (It is important to note that...) → drop
  - Expand-only-by-AI phrases (cannot/do not/will not...) → contractions
  - "In the realm of X" / "navigating the complexities of X" → simpler equivalents

The substitutions are deliberately conservative. We never change a fact, never
swap an entity, never alter sentence count. The output reads as the same person
but with the AI surface tics removed.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# ---------- Substitution tables ----------

_TRANSITIONS_DROP = {
    # Drop entirely; capitalize the following word.
    r"^Furthermore,\s+",
    r"\s+Furthermore,\s+",
    r"^Moreover,\s+",
    r"\s+Moreover,\s+",
    r"^Additionally,\s+",
    r"\s+Additionally,\s+",
    r"^In addition,\s+",
    r"\s+In addition,\s+",
    r"^In conclusion,\s+",
    r"^To conclude,\s+",
    r"^In summary,\s+",
    r"^To summarize,\s+",
    r"^Overall,\s+",
    r"^Ultimately,\s+",
    r"^All in all,\s+",
}

_TRANSITIONS_SOFT = {
    # Replace with a softer/shorter human variant. Picked at random per match.
    "However,": ["But", "Still,", "That said,", "Though"],
    "Nevertheless,": ["Still,", "Even so,"],
    "Nonetheless,": ["Even so,", "Still,"],
    "Consequently,": ["So", "Which means"],
    "Therefore,": ["So"],
    "Thus,": ["So"],
    "Hence,": ["So"],
    "Indeed,": [""],
}

_FAVORITE_WORDS = {
    # AI-overused → plain alternatives. Word-boundary replacement, case-preserving.
    r"\bdelve into\b": "look at",
    r"\bdelve\b": "explore",
    r"\bdelves\b": "explores",
    r"\bdelved\b": "explored",
    r"\bdelving\b": "exploring",
    r"\bleverage\b": "use",
    r"\bleverages\b": "uses",
    r"\bleveraged\b": "used",
    r"\bleveraging\b": "using",
    r"\bnavigate\b": "handle",
    r"\bnavigates\b": "handles",
    r"\bnavigated\b": "handled",
    r"\bnavigating\b": "handling",
    r"\bembark on\b": "start",
    r"\bembark upon\b": "start",
    r"\bembark\b": "start",
    r"\bembarks\b": "starts",
    r"\bembarked\b": "started",
    r"\bembarking\b": "starting",
    r"\bfoster\b": "build",
    r"\bfosters\b": "builds",
    r"\bunderscore\b": "highlight",
    r"\bunderscores\b": "highlights",
    r"\bshowcase\b": "show",
    r"\bshowcases\b": "shows",
    r"\bshowcased\b": "showed",
    r"\bgarner\b": "earn",
    r"\bgarners\b": "earns",
    r"\bgarnered\b": "earned",
    r"\bintricate\b": "complex",
    r"\bmultifaceted\b": "complex",
    r"\bparamount\b": "essential",
    r"\bcrucial\b": "important",
    r"\bpivotal\b": "central",
    r"\brobust\b": "solid",
    r"\bseamless\b": "smooth",
    r"\bcomprehensive\b": "thorough",
    r"\bholistic\b": "broad",
    r"\bnuanced\b": "subtle",
    r"\binnovative\b": "new",
    r"\bcutting-edge\b": "new",
    r"\btransformative\b": "significant",
    r"\bgroundbreaking\b": "major",
    r"\bmyriad\b": "many",
    r"\bvibrant\b": "lively",
    r"\bbustling\b": "busy",
    r"\bremarkable\b": "notable",
    r"\btapestry\b": "mix",
    r"\bplethora\b": "lots of",
    r"\bparadigm\b": "approach",
    r"\bsynergy\b": "fit",
    r"\bendeavor\b": "effort",
    r"\bendeavour\b": "effort",
    r"\btestament\b": "sign",
    r"\bcornerstone\b": "foundation",
    # Note: "landscape", "ecosystem", "realm" — handled by PHRASE_SWAPS in
    # context, NOT by blanket word swap (creates worse output otherwise).
}

_HEDGING_DROP = (
    "it is important to note that ",
    "it's important to note that ",
    "it is worth noting that ",
    "it's worth noting that ",
    "it should be noted that ",
    "it is essential to note that ",
    "it is crucial to note that ",
    "in today's rapidly evolving ",
    "in today's fast-paced ",
    "in today's world, ",
    "in today's society, ",
)

_PHRASE_SWAPS = {
    # Run BEFORE word-level swaps so we don't get "complex complexities".
    "intricate complexities": "complexities",
    "intricate complexity": "complexity",
    "multifaceted complexities": "complexities",
    "multifaceted nature of": "scope of",
    "the realm of": "",
    "in the realm of": "in",
    "when it comes to": "for",
    "navigating the complexities of": "dealing with",
    "the ever-changing landscape of": "",
    "the rapidly evolving landscape of": "",
    "the digital landscape": "the digital world",
    "the modern landscape of": "modern",
    "the broader landscape of": "",
    "shed light on": "explain",
    "play a crucial role in": "drive",
    "play a vital role in": "drive",
    "plays a crucial role in": "drives",
    "plays a vital role in": "drives",
    "a wide range of": "many",
    "a variety of": "many",
    "stand the test of time": "last",
    "at the end of the day": "ultimately",
    "due to the fact that": "because",
    "in order to": "to",
    "a large number of": "many",
    "at this point in time": "now",
    "in today's rapidly evolving digital ": "in the digital ",
    "in today's rapidly evolving ": "in the modern ",
    "in today's fast-paced ": "in the modern ",
}

_CONTRACTIONS = {
    r"\bdo not\b": "don't",
    r"\bdoes not\b": "doesn't",
    r"\bdid not\b": "didn't",
    r"\bis not\b": "isn't",
    r"\bare not\b": "aren't",
    r"\bwas not\b": "wasn't",
    r"\bwere not\b": "weren't",
    r"\bcannot\b": "can't",
    r"\bwill not\b": "won't",
    r"\bwould not\b": "wouldn't",
    r"\bcould not\b": "couldn't",
    r"\bshould not\b": "shouldn't",
    r"\bhas not\b": "hasn't",
    r"\bhave not\b": "haven't",
    r"\bhad not\b": "hadn't",
    r"\bit is\b": "it's",
    r"\bthat is\b": "that's",
    r"\bthere is\b": "there's",
    r"\byou are\b": "you're",
    r"\bthey are\b": "they're",
    r"\bwe are\b": "we're",
    r"\bI am\b": "I'm",
    r"\bI have\b": "I've",
    r"\bI will\b": "I'll",
}


# ---------- Public API ----------

@dataclass
class ScrubConfig:
    drop_stiff_transitions: bool = True
    soften_transitions: bool = True
    swap_favorite_words: bool = True
    drop_hedging: bool = True
    swap_phrases: bool = True
    apply_contractions: bool = True
    seed: int | None = None
    # If True, preserve original casing when swapping words (Leverage → Use, not use).
    case_preserve: bool = True


@dataclass
class ScrubResult:
    text: str
    edits: int = 0
    edits_by_kind: dict[str, int] = field(default_factory=dict)


# ---------- Helpers ----------

def _case_preserve_swap(replacement: str, matched: str) -> str:
    if not matched:
        return replacement
    if matched.isupper():
        return replacement.upper()
    if matched[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _drop_pattern(text: str, pattern: str) -> tuple[str, int]:
    new_text, n = re.subn(pattern, " ", text)
    new_text = re.sub(r" +", " ", new_text).strip()
    # Capitalize what now starts the sentence (after any leading punctuation).
    new_text = re.sub(
        r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), new_text
    )
    return new_text, n


def _swap_table(text: str, table: dict, *, case_preserve: bool, rng: random.Random) -> tuple[str, int]:
    edits = 0
    for pat, repl in table.items():
        # If repl is a list, pick one per match.
        if isinstance(repl, list):
            def _sub(m, opts=repl, r=rng):
                return r.choice(opts)
            new_text, n = re.subn(pat, _sub, text)
        elif case_preserve:
            def _sub(m, r=repl):
                return _case_preserve_swap(r, m.group(0))
            new_text, n = re.subn(pat, _sub, text, flags=re.IGNORECASE)
        else:
            new_text, n = re.subn(pat, repl, text, flags=re.IGNORECASE)
        text = new_text
        edits += n
    return text, edits


def scrub(text: str, cfg: ScrubConfig | None = None) -> ScrubResult:
    cfg = cfg or ScrubConfig()
    rng = random.Random(cfg.seed)
    edits = 0
    by_kind: dict[str, int] = {}

    def _bump(kind: str, n: int) -> None:
        nonlocal edits
        if n:
            edits += n
            by_kind[kind] = by_kind.get(kind, 0) + n

    if cfg.drop_stiff_transitions:
        for pat in _TRANSITIONS_DROP:
            text, n = _drop_pattern(text, pat)
            _bump("transition_drop", n)

    if cfg.soften_transitions:
        text, n = _swap_table(text, _TRANSITIONS_SOFT, case_preserve=False, rng=rng)
        _bump("transition_soften", n)

    if cfg.drop_hedging:
        for phrase in _HEDGING_DROP:
            new_text, n = re.subn(re.escape(phrase), "", text, flags=re.IGNORECASE)
            text = new_text
            _bump("hedging_drop", n)
        text = re.sub(r"  +", " ", text).strip()
        text = re.sub(
            r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
        )

    if cfg.swap_phrases:
        for phrase, repl in _PHRASE_SWAPS.items():
            new_text, n = re.subn(re.escape(phrase), repl, text, flags=re.IGNORECASE)
            text = new_text
            _bump("phrase_swap", n)
        text = re.sub(r"  +", " ", text).strip()

    if cfg.swap_favorite_words:
        text, n = _swap_table(text, _FAVORITE_WORDS, case_preserve=cfg.case_preserve, rng=rng)
        _bump("favorite_word", n)

    if cfg.apply_contractions:
        text, n = _swap_table(text, _CONTRACTIONS, case_preserve=cfg.case_preserve, rng=rng)
        _bump("contraction", n)

    # Redundancy collapse: kill adjacent same-stem words like "complex complexities"
    text, n = re.subn(
        r"\b(\w{4,})\w*\s+\1\w*\b", lambda m: m.group(0).split()[0], text, flags=re.IGNORECASE
    )
    _bump("redundancy_collapse", n)

    # Final: drop double spaces and capitalize each sentence-start.
    text = re.sub(r"  +", " ", text).strip()
    text = re.sub(
        r"(^|(?<=[.!?])\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )
    # Drop dangling commas right after a period (artifact of mid-clause drops).
    text = re.sub(r"\.\s*,\s*", ". ", text)

    return ScrubResult(text=text, edits=edits, edits_by_kind=by_kind)
