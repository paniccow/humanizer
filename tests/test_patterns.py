"""Tests for the AI-pattern fingerprint — pure-python, no model downloads."""
from humanizer.patterns import analyze
from humanizer.patterns.signals import (
    burstiness_score,
    contraction_deficit_score,
    favorite_word_density,
    hedging_phrase_score,
    stiff_transition_score,
)


_AI_SAMPLE = (
    "In today's rapidly evolving digital landscape, artificial intelligence has "
    "emerged as a transformative force that is reshaping numerous industries. "
    "Furthermore, organizations are increasingly leveraging AI-driven solutions "
    "to enhance operational efficiency, streamline workflows, and deliver "
    "personalized experiences. Moreover, the integration of machine learning "
    "algorithms with existing infrastructure enables companies to derive "
    "actionable insights from vast amounts of data. Additionally, this paradigm "
    "shift is fundamentally altering the competitive dynamics across multiple "
    "sectors. It is important to note that organizations must navigate the "
    "intricate complexities of this multifaceted landscape."
)

_HUMAN_SAMPLE = (
    "AI is everywhere now. Every company I talk to is trying to bolt it onto "
    "something. Sometimes that pays off, sometimes it just adds latency nobody "
    "asked for. The honest answer is most teams don't have the data pipeline to "
    "make ML useful yet, and that's the part nobody wants to fix because it "
    "isn't glamorous. Still, when it lands, it lands big. The shift is real. "
    "The timeline is just slower than the conference talks make it sound."
)


def test_ai_sample_scores_higher_than_human():
    ai = analyze(_AI_SAMPLE)
    human = analyze(_HUMAN_SAMPLE)
    assert ai.aggregate > human.aggregate
    # The gap should be meaningful, not just rounding.
    assert ai.aggregate - human.aggregate > 0.3


def test_individual_signals_fire():
    assert stiff_transition_score(_AI_SAMPLE) > 0.5
    assert favorite_word_density(_AI_SAMPLE) > 0.5
    assert hedging_phrase_score(_AI_SAMPLE) > 0.5
    assert contraction_deficit_score(_AI_SAMPLE) > 0.7  # all "do not"/"is not" expanded
    # Burstiness depends on sentence-length variance — the AI sample is uniform.
    assert burstiness_score(_AI_SAMPLE) > 0.5


def test_human_sample_doesnt_fire_most_signals():
    fp = analyze(_HUMAN_SAMPLE)
    # Human sample uses contractions and varies sentence length.
    assert contraction_deficit_score(_HUMAN_SAMPLE) < 0.5
    assert favorite_word_density(_HUMAN_SAMPLE) < 0.3
    assert stiff_transition_score(_HUMAN_SAMPLE) < 0.3
    # At most a couple of flags lit; not the AI-sample's 5+
    assert len(fp.flagged) <= 2


def test_explain_renders_without_error():
    out = analyze(_AI_SAMPLE).explain()
    assert "Aggregate AI-likeness" in out
    assert "Flagged" in out
