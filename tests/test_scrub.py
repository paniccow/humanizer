"""Tests for the deterministic AI-tell scrubber. Pure-python, no model loads."""
from humanizer.pipeline import scrub
from humanizer.pipeline.scrub import ScrubConfig
from humanizer.patterns import analyze


_AI_SAMPLE = (
    "In today's rapidly evolving digital landscape, artificial intelligence has "
    "emerged as a transformative force. Furthermore, organizations are leveraging "
    "AI to navigate the intricate complexities of this multifaceted ecosystem. "
    "Moreover, the integration of machine learning is paramount. Additionally, "
    "this paradigm shift is fundamentally altering the realm of business. "
    "It is important to note that companies must delve into these capabilities "
    "to remain competitive."
)


def test_scrub_lowers_pattern_aggregate():
    before = analyze(_AI_SAMPLE).aggregate
    scrubbed = scrub(_AI_SAMPLE).text
    after = analyze(scrubbed).aggregate
    assert after < before, f"scrub didn't reduce AI-likeness: {before:.2f} -> {after:.2f}"
    # Should drop pattern aggregate by ≥ 25% relative (more robust to weight changes
    # as new signals are added).
    rel_drop = (before - after) / before
    assert rel_drop > 0.25, f"scrub barely helped: {before:.2f} -> {after:.2f} ({rel_drop:.0%} relative)"


def test_scrub_kills_named_offenders():
    out = scrub(_AI_SAMPLE).text
    for offender in ["Furthermore", "Moreover", "Additionally",
                      "leverage", "navigate", "intricate", "multifaceted",
                      "paramount", "delve",
                      "in today's rapidly evolving", "It is important to note"]:
        assert offender.lower() not in out.lower(), f"scrub left '{offender}' in output"


def test_scrub_preserves_facts_and_length_roughly():
    out = scrub(_AI_SAMPLE).text
    # Should preserve key entities like "machine learning"
    assert "machine learning" in out.lower()
    # Length should be within ±35% of original (we drop some boilerplate).
    ratio = len(out.split()) / len(_AI_SAMPLE.split())
    assert 0.65 < ratio < 1.35, f"length ratio off: {ratio:.2f}"


def test_scrub_reports_edit_count():
    result = scrub(_AI_SAMPLE)
    assert result.edits > 5
    assert "transition_drop" in result.edits_by_kind or "transition_soften" in result.edits_by_kind
    assert "favorite_word" in result.edits_by_kind


def test_scrub_idempotent_on_clean_text():
    clean = "I think the model works pretty well. There are still rough edges, but it's progress."
    result = scrub(clean)
    # No changes — text is already human.
    assert result.edits == 0
    assert result.text == clean


def test_scrub_disable_individual_stages():
    out = scrub(_AI_SAMPLE, ScrubConfig(swap_favorite_words=False)).text
    # leverage should still be there since we disabled that stage
    assert "leverage" in out.lower() or "leveraging" in out.lower()
