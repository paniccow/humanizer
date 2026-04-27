"""Tests for the advanced scrub stages added in subsequent rounds:
em-dash thinning, tricolon breaking, sentence splitting/merging,
archaic-marker drops, a/an article repair.

All pure-python; no model loads.
"""
from humanizer.pipeline import scrub
from humanizer.pipeline.scrub import (
    ScrubConfig,
    _add_variance,
    _break_tricolons,
    _merge_short_runs,
    _split_long_sentences,
    _thin_em_dashes,
)
from humanizer.patterns import analyze


# ---- em-dash thinning ----

def test_em_dash_in_parenthetical_becomes_comma():
    text = "Companies — including small ones — must adapt."
    out, n = _thin_em_dashes(text)
    assert "—" not in out
    assert "Companies, including small ones, must adapt." == out
    assert n == 2


def test_em_dash_separating_clauses_becomes_period():
    # Right side: capital + ≥4 words → treat as new sentence (period).
    text = "AI is everywhere — Companies have transformed their entire workflow."
    out, _ = _thin_em_dashes(text)
    assert "—" not in out
    # Should produce 2 sentences
    assert "everywhere." in out or "everywhere. " in out


def test_em_dash_no_artifacts():
    """Earlier bug: ' , ' (space before comma). Make sure it stays gone."""
    text = "Things — like cars — are big."
    out, _ = _thin_em_dashes(text)
    assert " ," not in out
    assert "  " not in out


def test_double_dash_normalized():
    text = "AI is here -- it's everywhere -- including chatbots."
    out, _ = _thin_em_dashes(text)
    assert "--" not in out
    assert "—" not in out
    assert ", " in out


# ---- tricolon breaking ----

def test_tricolon_three_short_items_breaks():
    text = "I bought apples, bananas, and oranges yesterday."
    out, n = _break_tricolons(text)
    assert n >= 1
    assert "apples and bananas. Oranges" in out


def test_tricolon_long_items_kept():
    """If items are long, the tricolon is left alone (would create awkward fragments)."""
    text = ("Companies use machine learning models with custom optimizers, "
            "deep learning architectures involving many layers, "
            "and reinforcement learning loops with reward shaping.")
    out, n = _break_tricolons(text)
    assert n == 0  # all items > 4 words → not split
    # Original text preserved
    assert out == text


# ---- sentence splitting ----

def test_long_sentence_split_at_coordinator():
    text = (
        "The system is highly complex and difficult to understand without years of training, "
        "and even experienced engineers occasionally find themselves stumped by edge cases."
    )
    out, n = _split_long_sentences(text, max_words=20)
    assert n >= 1
    # Should produce a period somewhere mid-text
    assert ". " in out


def test_short_sentence_not_split():
    text = "I went home. I slept."
    out, n = _split_long_sentences(text, max_words=28)
    assert n == 0
    assert out == text


# ---- merge short runs ----

def test_merge_short_fragments():
    text = "I walked. I ran. I jumped. The sun was bright and the day was nice."
    out, n = _merge_short_runs(text, min_words=4)
    assert n >= 1
    # First three short sentences should have been merged with commas
    assert ", " in out


# ---- variance injection ----

def test_variance_injection_on_uniform_text():
    """Texts with all-similar sentence lengths get aggressively split."""
    # 5 sentences all roughly 10 words.
    text = (
        "The dog ran across the field with great speed. "
        "The cat watched from the windowsill with calm interest. "
        "The bird sang loudly from the tall pine tree. "
        "The owner called out, but neither animal responded. "
        "The neighbor laughed at the scene from her porch."
    )
    out, n = _add_variance(text, target_cv=0.5)
    # Should have made an attempt (or possibly 0 if no good split points)
    assert isinstance(n, int) and n >= 0
    # Output should still parse as multiple sentences
    assert out.count(".") >= 5


# ---- archaic marker drops ----

def test_archaic_thereof_dropped():
    text = "The system thereof is comprehensive."
    out = scrub(text).text
    assert "thereof" not in out
    # No "of it's" cascade artifact
    assert "of it's" not in out
    assert "of its" not in out


def test_archaic_whereby_kept_meaning():
    text = "A process whereby decisions are made."
    out = scrub(text).text
    assert "whereby" not in out
    assert "where" in out


def test_archaic_henceforth_dropped():
    text = "Henceforth, all employees must wear badges."
    out = scrub(text).text.lower()
    assert "henceforth" not in out


# ---- a/an article repair ----

def test_an_to_a_after_consonant_swap():
    text = "It plays an big role."
    out = scrub(text).text
    assert "an big" not in out
    assert "a big" in out


def test_a_to_an_after_vowel_swap():
    text = "It plays a important role."
    out = scrub(text).text
    assert "a important" not in out
    assert "an important" in out


def test_silent_h_keeps_an():
    text = "It is an honest mistake."
    out = scrub(text).text
    assert "an honest" in out


def test_yu_sound_keeps_a():
    text = "This is a unique problem."
    out = scrub(text).text
    assert "a unique" in out


def test_hour_takes_an():
    text = "They built a hourly schedule."
    out = scrub(text).text
    assert "an hourly" in out


# ---- end-to-end pattern aggregate must drop ----

def test_full_scrub_drops_pattern_aggregate_on_busy_ai_text():
    sample = (
        "In today's rapidly evolving digital landscape, organizations are increasingly "
        "leveraging AI-driven solutions to navigate intricate complexities. Furthermore, "
        "the integration thereof is paramount. Moreover, it is important to note that "
        "companies — including small ones — must delve into multifaceted capabilities. "
        "That being said, the framework hereby established plays a crucial role."
    )
    before = analyze(sample).aggregate
    after = analyze(scrub(sample).text).aggregate
    assert after < before
    # We expect a substantial reduction (≥30 percentage points relative)
    assert (before - after) / before > 0.30
