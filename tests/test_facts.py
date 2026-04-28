"""Fact-preservation metric tests — pure-python, no models, no network."""
from __future__ import annotations

import pytest

from humanizer.metrics.facts import entity_overlap, extract_facts


def test_extract_currency():
    f = extract_facts("Sales hit $5,000 and €1.5 million in Q3.")
    assert "$5000" in f.by_kind["currency"]
    assert "€1.5" in f.by_kind["currency"]


def test_extract_percentages():
    f = extract_facts("Up 25% from last year, down 1.4 percent overall.")
    assert "25%" in f.by_kind["pct"]
    assert "1.4 percent" in f.by_kind["pct"]


def test_extract_years_distinct_from_numbers():
    f = extract_facts("In 1947, India gained independence; 75 nations supported it.")
    assert "1947" in f.by_kind["year"]
    assert "75" in f.by_kind["number"]
    assert "1947" not in f.by_kind["number"]


def test_extract_dates():
    f = extract_facts("The deadline is January 5, 2024 — see the schedule.")
    matches = {x.lower() for x in f.by_kind["date"]}
    assert any("january 5" in m for m in matches)


def test_extract_proper_nouns():
    f = extract_facts("OpenAI released GPT-4 last year, after Anthropic's Claude.")
    nouns = {n for n in f.by_kind["proper_noun"]}
    # Heuristic — at minimum we want OpenAI, Anthropic, Claude
    assert "OpenAI" in nouns
    assert "Anthropic" in nouns


def test_overlap_perfect_preservation():
    src = "In 1947 the budget was $5,000."
    cand = "In 1947 the budget was $5,000."
    assert entity_overlap(src, cand) == pytest.approx(1.0)


def test_overlap_paraphrased_but_facts_kept():
    src = "Revenue grew 25% to $1.2 million in 2024."
    cand = "Revenue jumped 25% in 2024, reaching $1.2 million."  # same facts, different prose
    assert entity_overlap(src, cand) >= 0.9


def test_overlap_drops_a_number():
    src = "We saw a 25% increase to $1,000 this quarter."
    cand = "We saw a big increase this quarter."  # all numbers dropped
    assert entity_overlap(src, cand) < 0.5


def test_overlap_changes_a_year():
    # "Founded" gets matched as proper-noun-ish (sentence-initial single
    # capitalized word — a known heuristic false-positive), so two facts
    # are extracted: {1947, Founded}. The candidate keeps "Founded" but
    # changes the year, so overlap = 1/2 = 0.5. This is the cost of the
    # regex-only NER; acceptable for an opt-in preservation gate.
    src = "Founded in 1947."
    cand = "Founded in 1948."
    overlap = entity_overlap(src, cand)
    assert overlap == pytest.approx(0.5)


def test_overlap_changes_a_year_with_unique_fact():
    """When the year is the ONLY fact, a year change is fully detected."""
    src = "the year was 1947 then."  # all-lowercase prose, no proper-noun bait
    cand = "the year was 1948 then."
    assert entity_overlap(src, cand) == pytest.approx(0.0)


def test_overlap_no_facts_in_source():
    src = "It was a sunny day and the wind was nice."
    cand = "Whatever this is, it works."
    assert entity_overlap(src, cand) == 1.0  # nothing to lose -> trivially preserved


def test_overlap_substring_fallback_for_currency():
    """`$5,000` matched in source as currency; in candidate where the prose
    says `5,000 dollars` — substring fallback should still credit it."""
    src = "Expenses totaled $5,000."
    cand = "Expenses totaled 5,000 dollars."
    assert entity_overlap(src, cand) >= 0.5  # at least the bare number preserved
