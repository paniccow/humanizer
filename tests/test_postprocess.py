"""Tests for the burstiness post-processor — pure-python, no model downloads."""
from humanizer.postprocess import BurstinessConfig, apply_burstiness
from humanizer.metrics.burstiness import sentence_length_stats


def test_apply_burstiness_changes_text():
    src = (
        "Furthermore, the model is highly sophisticated. "
        "Moreover, it leverages cutting-edge techniques. "
        "Additionally, the implementation is multifaceted. "
        "It is important to note that performance is paramount."
    )
    out = apply_burstiness(src, BurstinessConfig(seed=0))
    assert out != src
    # transitions should be replaced
    assert "Furthermore" not in out
    assert "Moreover" not in out
    assert "Additionally" not in out


def test_contractions_applied():
    src = "It is not the case that we cannot do this. We will not give up."
    out = apply_burstiness(src, BurstinessConfig(seed=0))
    assert "isn't" in out or "don't" in out or "can't" in out or "won't" in out


def test_burstiness_stats():
    text = "Short sentence. Now here is a much, much longer sentence with many more words in it."
    stats = sentence_length_stats(text)
    assert stats.n_sentences == 2
    assert stats.cv_words > 0  # variance, not flat
