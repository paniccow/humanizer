"""Burstiness post-processor.

After model generation, apply lightweight surface edits that bump perplexity and
sentence-length variance — the two main signals every commercial AI detector uses.
The edits are deterministic + reversible (no semantic drift): merging adjacent
short sentences, splitting overly even ones at coordinator clauses, swapping in
contractions, varying connector vocabulary.

This is a complement, not a replacement, for a trained policy. Use it as a
post-processing pass on the policy's output.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from ..metrics.burstiness import BurstinessStats, sentence_length_stats, split_sentences

# Mild, semantics-preserving substitutions human writers actually use.
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
    r"\bit is\b": "it's",
    r"\bthat is\b": "that's",
    r"\bwhat is\b": "what's",
    r"\bthere is\b": "there's",
    r"\byou are\b": "you're",
    r"\bthey are\b": "they're",
    r"\bwe are\b": "we're",
    r"\bI am\b": "I'm",
    r"\bI have\b": "I've",
}

# Stiff transitional phrases AI overuses → natural variants. Picked at random.
_TRANSITION_VARIANTS = {
    "Furthermore,": ["Also,", "On top of that,", "And", "Plus,"],
    "Moreover,": ["Also,", "Beyond that,", "And"],
    "Additionally,": ["Also,", "Plus,", "And"],
    "However,": ["But", "That said,", "Still,", "Though"],
    "Therefore,": ["So", "Which means", "So,"],
    "Thus,": ["So", "Which means"],
    "In conclusion,": ["So in the end,", "All told,", "Bottom line:"],
    "It is important to note that": ["Worth noting:", "One thing —", "Note that"],
    "In order to": ["To"],
    "Due to the fact that": ["Because"],
    "A large number of": ["Many"],
    "At this point in time": ["Now"],
}


@dataclass
class BurstinessConfig:
    apply_contractions: bool = True
    vary_transitions: bool = True
    split_long: bool = True              # split sentences > target_max_words
    merge_short: bool = True             # merge runs of sentences < target_min_words
    target_min_words: int = 6
    target_max_words: int = 28
    seed: int | None = None


def _maybe_contractions(text: str) -> str:
    for pat, repl in _CONTRACTIONS.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def _vary_transitions(text: str, rng: random.Random) -> str:
    for stiff, variants in _TRANSITION_VARIANTS.items():
        # Replace each occurrence independently, choose a variant per match.
        def _sub(_m, vs=variants, r=rng):
            return r.choice(vs)
        text = re.sub(re.escape(stiff), _sub, text)
    return text


def _split_at_coordinator(sent: str) -> list[str]:
    # Split at ", and"/", but"/", so" if the sentence is long enough on each side.
    m = re.search(r",\s+(and|but|so|yet|or)\s+", sent)
    if not m:
        return [sent]
    left = sent[: m.start()].strip()
    right = sent[m.end():].strip()
    if not left or not right:
        return [sent]
    # Capitalize first letter of right; left already capitalized.
    right = right[0].upper() + right[1:]
    if not left.endswith((".", "!", "?")):
        left += "."
    if not right.endswith((".", "!", "?")):
        right += "."
    return [left, right]


def _merge_short_runs(sents: list[str], min_words: int) -> list[str]:
    out: list[str] = []
    buf: list[str] = []

    def flush():
        if not buf:
            return
        merged = " ".join(s.rstrip(".") for s in buf) + "."
        out.append(merged)
        buf.clear()

    for s in sents:
        if len(s.split()) < min_words:
            buf.append(s)
            if len(" ".join(buf).split()) >= min_words * 2:
                flush()
        else:
            flush()
            out.append(s)
    flush()
    return out


def apply_burstiness(text: str, cfg: BurstinessConfig | None = None) -> str:
    cfg = cfg or BurstinessConfig()
    rng = random.Random(cfg.seed)
    if cfg.vary_transitions:
        text = _vary_transitions(text, rng)
    sents = split_sentences(text)
    if cfg.split_long:
        new_sents: list[str] = []
        for s in sents:
            if len(s.split()) > cfg.target_max_words:
                new_sents.extend(_split_at_coordinator(s))
            else:
                new_sents.append(s)
        sents = new_sents
    if cfg.merge_short:
        sents = _merge_short_runs(sents, cfg.target_min_words)
    text = " ".join(sents)
    if cfg.apply_contractions:
        text = _maybe_contractions(text)
    return text


def report(text: str) -> BurstinessStats:
    return sentence_length_stats(text)
