"""Safety/preservation tests — make sure scrub doesn't break key text.

Each rule we add could regress on:
  - Factual content (numbers, names, dates, units)
  - Quoted speech / dialogue
  - Negation polarity
  - Code / technical content
  - Specialized vocabulary

These tests catch the worst regressions early. They don't validate that scrub
does anything useful — just that it doesn't *break* things.
"""
from humanizer.pipeline import scrub


def _has_all(text: str, items: list[str]) -> bool:
    """All items must appear (case-insensitive substring)."""
    lower = text.lower()
    return all(item.lower() in lower for item in items)


# ---- factual content preserved ----

def test_numbers_preserved():
    text = "The temperature was 73.5 degrees and the humidity was 42%."
    out = scrub(text).text
    assert "73.5" in out
    assert "42%" in out


def test_dates_preserved():
    text = "On January 15, 2024, the company launched its new product."
    out = scrub(text).text
    assert "January 15, 2024" in out


def test_proper_nouns_preserved():
    text = "Alice met Bob in San Francisco at the Salesforce tower."
    out = scrub(text).text
    assert _has_all(out, ["Alice", "Bob", "San Francisco", "Salesforce"])


def test_units_preserved():
    text = "Each unit weighs 2.3kg and measures 5.1m × 1.7m."
    out = scrub(text).text
    assert "2.3kg" in out
    assert "5.1m" in out


# ---- quoted speech preserved ----

def test_quoted_speech_preserved():
    text = 'She said "I will not go" and turned away.'
    out = scrub(text).text
    # The quote should be intact; contractions outside the quote are fine.
    assert '"I will not go"' in out


def test_single_quotes_preserved():
    text = "He muttered 'this is unbelievable' under his breath."
    out = scrub(text).text
    # Inner content preserved
    assert "'this is unbelievable'" in out or "'this is unbelievable'" in out


# ---- negation polarity ----

def test_simple_negation_preserved():
    """Contractions of 'do not' must stay negative."""
    text = "I do not want to go."
    out = scrub(text).text
    # Either "I don't want to go." or "I do not want to go."  — both negative.
    assert "don't" in out or "do not" in out
    # The verb went, NOT "want to go" affirmatively.
    assert "I want to go." not in out


def test_double_negation_unchanged():
    text = "It is not unfair."
    out = scrub(text).text
    # 'is not' may contract to 'isn't', but the negation polarity must remain.
    # Reasonable outputs: "It's not unfair." OR "It isn't unfair."
    assert ("not unfair" in out) or ("isn't unfair" in out)


# ---- code / technical content ----

def test_code_snippet_preserved():
    text = "Run `import pandas as pd` and then call `pd.read_csv('data.csv')`."
    out = scrub(text).text
    # Code spans should still appear (literal pandas / pd / read_csv)
    assert "pandas" in out
    assert "pd.read_csv" in out


def test_technical_terms_preserved():
    text = "The latency was 12ms p99 and the throughput was 450 req/s."
    out = scrub(text).text
    assert "12ms" in out and "p99" in out
    assert "450 req/s" in out


# ---- domain content stays coherent ----

def test_short_paragraph_round_trip():
    """Short business paragraph must come out as parseable English."""
    text = (
        "The team delivered the project on schedule. We launched on Tuesday "
        "and saw 200 sign-ups by Friday."
    )
    out = scrub(text).text
    # Length within ±30%
    ratio = len(out.split()) / len(text.split())
    assert 0.6 <= ratio <= 1.4
    # Numbers preserved
    assert "200" in out
    assert "Tuesday" in out
    assert "Friday" in out


# ---- empty / trivial ----

def test_empty_string():
    assert scrub("").text == ""


def test_single_word():
    assert scrub("Hello.").text in ("Hello.", "Hello")  # punctuation may be lost


def test_whitespace_only():
    assert scrub("   ").text.strip() == ""
