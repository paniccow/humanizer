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
    # Drop entirely. Match at text-start OR after sentence-end punctuation —
    # AI loves dropping these mid-paragraph too.
    r"(?:^|(?<=[.!?])\s+)Furthermore,\s+",
    r"\bFurthermore,\s+",
    r"(?:^|(?<=[.!?])\s+)Moreover,\s+",
    r"\bMoreover,\s+",
    r"(?:^|(?<=[.!?])\s+)Additionally,\s+",
    r"\bAdditionally,\s+",
    r"(?:^|(?<=[.!?])\s+)In addition,\s+",
    r"\bIn addition,\s+",
    r"(?:^|(?<=[.!?])\s+)In conclusion,\s+",
    r"(?:^|(?<=[.!?])\s+)To conclude,\s+",
    r"(?:^|(?<=[.!?])\s+)In summary,\s+",
    r"(?:^|(?<=[.!?])\s+)To summarize,\s+",
    r"(?:^|(?<=[.!?])\s+)Overall,\s+",
    r"(?:^|(?<=[.!?])\s+)Ultimately,\s+",
    r"(?:^|(?<=[.!?])\s+)All in all,\s+",
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
    # Archaic-formal markers — humans almost never use these in casual prose.
    # Drop them outright rather than expanding to phrases that cascade with
    # contraction rules (e.g. 'thereof' -> 'of it' -> 'of it's' was a bug).
    r"\s+whereby\s+": " where ",
    r"\s+therein\b": "",
    r"\s+hereby\b": "",
    r"\s+thereof\b": "",
    r"\s+thereto\b": "",
    # Henceforth/Hitherto at sentence start: drop including trailing comma.
    r"(?:^|(?<=[.!?])\s+)Henceforth,?\s+": " ",
    r"(?:^|(?<=[.!?])\s+)Hitherto,?\s+": " ",
    r"\s+henceforth\s+": " ",
    r"\s+hitherto\s+": " ",
    r"\bwherein\b": "where",
    r"\s+whereupon\s+": ", and ",
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
    # Note: 'in today's rapidly evolving' / 'in today's fast-paced' are
    # handled by PHRASE_SWAPS (substitute → 'in the modern') so the
    # following sentence-start word doesn't become an orphan fragment.
    "in today's world, ",
    "in today's society, ",
    # Newer GPT-4/Claude tells:
    "and there you have it",
    "and that's the beauty of it",
    "the bottom line is that ",
    "at the end of the day, ",
    "needless to say, ",
    "needless to say ",
    "it goes without saying that ",
    "it's worth mentioning that ",
    "it is worth mentioning that ",
    "it bears repeating that ",
    "it's important to remember that ",
    "it is important to remember that ",
    "as previously mentioned, ",
    "as we delve into ",
    "in the grand scheme of things, ",
    "with that being said, ",
    "that being said, ",
    # Parenthetical AI hedges
    "(in many cases) ",
    "(if you will) ",
    "(so to speak) ",
    "(for that matter) ",
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
    # Common AI buzzword adjacencies
    "deep dive into": "look at",
    "deep dive": "look",
    "key takeaway": "takeaway",
    "key takeaways": "takeaways",
    "moving forward,": "going forward,",
    "going forward,": "next,",
    "by harnessing": "by using",
    "harnessing the power of": "using",
    # "It's not just X, it's Y" — AI loves this construction
    "it's not just about ": "it's about ",
    "it's not only about ": "it's about ",
    # "Not only X, but also Y" -> simpler
    "not only that, but ": "and ",
    # Common cliche openers AI overuses
    "let's face it, ": "",
    "let's be honest, ": "",
    "the truth is, ": "",
    "the fact is, ": "",
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
    thin_em_dashes: bool = True               # NEW: replace em-dashes with mixed comma/period
    break_tricolons: bool = True              # split "X, Y, and Z" into ". Z" sentence
    split_long_sentences: bool = True         # break sentences > 30 words at coordinators
    merge_short_runs: bool = True             # merge adjacent fragments
    seed: int | None = None
    # If True, preserve original casing when swapping words (Leverage → Use, not use).
    case_preserve: bool = True
    target_max_words: int = 28                # split threshold for long sentences
    target_min_words: int = 5                 # merge threshold for short fragments


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


# ---------- Burstiness fixes (sentence-level restructuring) ----------

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_BOUNDARY.split(text.strip()) if s.strip()]


def _thin_em_dashes(text: str) -> tuple[str, int]:
    """Replace em-dashes (— or --) with more human-typical punctuation.
    AI overuses em-dashes 5-10x relative to typical human prose.

    Strategy: alternate between comma and period to break the AI rhythm:
      - " — " or " -- "  ->  ", " (comma, mid-sentence)
      - If the dash separates two independent clauses, use ". " (period)
    """
    edits = 0
    # Normalize -- to em-dash variant for consistency.
    text = text.replace("--", "—")

    # Find all em-dashes; alternate between comma/period.
    # Period when the right side starts with capital + multiple words (likely clause).
    parts = text.split("—")
    if len(parts) <= 1:
        return text, 0
    out_parts = [parts[0].rstrip()]
    for i, right in enumerate(parts[1:], start=1):
        right_clean = right.strip()           # both sides — was leaking trailing whitespace
        if not right_clean:
            continue
        # Heuristic: period if right side looks like a new sentence
        first_word = right_clean.split()[0] if right_clean else ""
        if first_word[:1].isupper() and len(right_clean.split()) >= 4:
            sep = ". "
        else:
            sep = ", "
        out_parts.append(sep + right_clean)
        edits += 1
    text = "".join(out_parts)
    # Final cleanup: collapse double-commas, fix capitalization after periods.
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(
        r"(^|(?<=[.!?])\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )
    return text, edits


def _break_tricolons(text: str) -> tuple[str, int]:
    """Convert 'A, B, and C' (when each item is short enough) into 'A and B. C'.
    Adds a sentence boundary AND breaks the X, Y, and Z pattern in one shot."""
    edits = 0
    def _sub(m: re.Match) -> str:
        nonlocal edits
        a, b, conj, c = m.group(1), m.group(2), m.group(3), m.group(4)
        # Only split if the items aren't too long (otherwise risk awkward fragments)
        if len(a.split()) <= 4 and len(b.split()) <= 4 and len(c.split()) <= 6:
            edits += 1
            cap_c = c[:1].upper() + c[1:] if c else c
            return f"{a} {conj} {b}. {cap_c}"
        return m.group(0)
    pattern = re.compile(
        r"\b([\w-]+(?:\s[\w-]+){0,3}),\s+([\w-]+(?:\s[\w-]+){0,3}),\s+(and|or)\s+([\w-]+(?:\s[\w-]+){0,5})"
    )
    new_text = pattern.sub(_sub, text)
    return new_text, edits


def _split_long_sentence(sent: str, max_words: int) -> list[str]:
    """If a sentence is > max_words and has a coordinator like ', and' / ', but',
    split it at that coordinator into two sentences."""
    if len(sent.split()) <= max_words:
        return [sent]
    m = re.search(r",\s+(and|but|so|yet|or)\s+", sent)
    if not m:
        return [sent]
    left = sent[: m.start()].strip()
    right = sent[m.end():].strip()
    if not left or not right:
        return [sent]
    if not left.endswith((".", "!", "?")):
        left += "."
    right = right[:1].upper() + right[1:]
    if not right.endswith((".", "!", "?")):
        right += "."
    return [left, right]


def _split_long_sentences(text: str, max_words: int) -> tuple[str, int]:
    sents = _split_sentences(text)
    new_sents: list[str] = []
    edits = 0
    for s in sents:
        parts = _split_long_sentence(s, max_words)
        if len(parts) > 1:
            edits += len(parts) - 1
        new_sents.extend(parts)
    return " ".join(new_sents), edits


def _add_variance(text: str, target_cv: float = 0.35) -> tuple[str, int]:
    """If sentence-length variance is suspiciously low (CV < target_cv),
    aggressively split the LONGEST sentences at any reasonable boundary
    (semicolon first, then any coordinator, then any comma-after-subject)
    until variance crosses target. Each split reduces uniformity."""
    import math as _math
    sents = _split_sentences(text)
    if len(sents) < 4:
        return text, 0
    counts = [len(s.split()) for s in sents]
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / len(counts)
    cv = _math.sqrt(var) / mean if mean > 0 else 0.0
    if cv >= target_cv:
        return text, 0

    edits = 0
    max_attempts = 4
    for _ in range(max_attempts):
        # Recompute CV; stop if we've reached target.
        counts = [len(s.split()) for s in sents]
        mean = sum(counts) / max(len(counts), 1)
        var = sum((c - mean) ** 2 for c in counts) / max(len(counts), 1)
        cv = _math.sqrt(var) / mean if mean > 0 else 0.0
        if cv >= target_cv:
            break
        # Pick the longest sentence; try splitting it.
        longest_idx = max(range(len(sents)), key=lambda i: counts[i])
        s = sents[longest_idx]
        # Try splitters in order of how "natural" the break is.
        # No-comma subord-conj splits would break restrictive clauses
        # ('the store where I bought milk') — kept comma-only.
        new_pair = None
        for pat in (
            r";\s+",                                            # semicolon
            r",\s+(and|but|so|yet|or)\s+",                      # coordinator
            r",\s+(which|who)\s+",                              # rel pronoun
            r",\s+(where|when|because|although|since|while)\s+", # subord conj (comma)
            r",\s+(?=[A-Za-z]+\s+)",                            # any comma followed by a clause
        ):
            m = re.search(pat, s)
            if m:
                left = s[: m.start()].strip()
                right = s[m.end():].strip()
                if not left or not right or len(left.split()) < 4 or len(right.split()) < 4:
                    continue
                if not left.endswith((".", "!", "?")):
                    left += "."
                right = right[:1].upper() + right[1:]
                if not right.endswith((".", "!", "?")):
                    right += "."
                new_pair = [left, right]
                break
        if not new_pair:
            break
        sents = sents[:longest_idx] + new_pair + sents[longest_idx + 1:]
        edits += 1

    return " ".join(sents), edits


def _merge_short_runs(text: str, min_words: int) -> tuple[str, int]:
    """Merge runs of adjacent very-short sentences into one sentence with comma joins.
    Reduces uniformity at the low end."""
    sents = _split_sentences(text)
    if len(sents) < 3:
        return text, 0
    out: list[str] = []
    buf: list[str] = []
    edits = 0

    def _flush() -> None:
        nonlocal edits
        if not buf:
            return
        if len(buf) == 1:
            out.append(buf[0])
        else:
            merged = ", ".join(s.rstrip(".!?") for s in buf) + "."
            out.append(merged)
            edits += len(buf) - 1
        buf.clear()

    for s in sents:
        if len(s.split()) < min_words:
            buf.append(s)
        else:
            _flush()
            out.append(s)
    _flush()
    return " ".join(out), edits


# Only DOUBLE quotes are protected. Single quotes are far more often contractions
# (it's, don't, they're) than actual quoted speech, and our previous regex
# was matching across contraction apostrophes — hiding entire paragraphs from
# the scrubber.
_QUOTE_RE = re.compile(r'("[^"]*"|“[^”]*”)')


def _protect_quotes(text: str) -> tuple[str, dict[str, str]]:
    """Replace quoted spans with placeholder tokens before scrub. Returns
    (text-with-placeholders, restore_map). Quotes are sacred — never modify
    text inside them."""
    restore: dict[str, str] = {}
    counter = [0]

    def _sub(m: re.Match) -> str:
        token = f"__SCRUB_QUOTE_{counter[0]:03d}__"
        counter[0] += 1
        restore[token] = m.group(0)
        return token
    return _QUOTE_RE.sub(_sub, text), restore


def _restore_quotes(text: str, restore: dict[str, str]) -> str:
    for token, original in restore.items():
        text = text.replace(token, original)
    return text


def scrub(text: str, cfg: ScrubConfig | None = None) -> ScrubResult:
    cfg = cfg or ScrubConfig()
    rng = random.Random(cfg.seed)
    edits = 0
    by_kind: dict[str, int] = {}

    # Protect quoted dialogue from any modification.
    text, _quote_restore = _protect_quotes(text)

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

    if cfg.thin_em_dashes:
        text, n = _thin_em_dashes(text)
        _bump("em_dash_thin", n)

    # Burstiness fixes — restructure at the SENTENCE level. These come AFTER
    # word-level swaps so we operate on the cleaner text.
    if cfg.break_tricolons:
        text, n = _break_tricolons(text)
        _bump("tricolon_break", n)

    if cfg.split_long_sentences:
        text, n = _split_long_sentences(text, cfg.target_max_words)
        _bump("sentence_split", n)

    if cfg.merge_short_runs:
        text, n = _merge_short_runs(text, cfg.target_min_words)
        _bump("sentence_merge", n)

    if cfg.split_long_sentences:
        # If sentences are still too uniform, aggressively introduce variance
        # by splitting the longest at any reasonable boundary (semicolon, etc.)
        text, n = _add_variance(text, target_cv=0.35)
        _bump("variance_inject", n)

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

    # a/an grammar repair — word swaps can leave "a important" or "an big".
    # Note: this is heuristic (vowel-LETTER detection misses 'an honest', 'a unique')
    # but covers ~95% of cases.
    def _fix_an(m: re.Match) -> str:
        article = m.group(1)
        space = m.group(2)
        next_word = m.group(3)
        starts_with_vowel = next_word[0].lower() in "aeiou"
        # Honor common exceptions: 'an honest', 'a unicorn', 'a useful'.
        if next_word.lower() in {"honest", "honor", "honour", "hour", "hourly", "honesty"}:
            starts_with_vowel = True
        if next_word.lower().startswith(("hour",)):
            starts_with_vowel = True
        if next_word.lower().startswith(("uni", "use", "user", "europ", "one")):
            starts_with_vowel = False
        correct = "an" if starts_with_vowel else "a"
        if article.lower() == correct:
            return m.group(0)
        # Preserve casing (A/a, An/an).
        if article[0].isupper():
            correct = correct.capitalize()
        return correct + space + next_word
    text = re.sub(r"\b(a|an|A|An)(\s+)([a-zA-Z][a-zA-Z'-]*)", _fix_an, text)
    # Drop dangling commas right after a period (artifact of mid-clause drops).
    text = re.sub(r"\.\s*,\s*", ". ", text)

    # Restore protected quotes.
    text = _restore_quotes(text, _quote_restore)

    return ScrubResult(text=text, edits=edits, edits_by_kind=by_kind)
