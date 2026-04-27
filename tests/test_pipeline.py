"""Pipeline orchestrator tests with stubbed humanizer + detectors.

We swap in tiny in-memory fakes so the test is fast (no model loads, no
network) but exercises every stage's wiring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from humanizer.detectors.base import Detector, DetectorScore
from humanizer.detectors.ensemble import DetectorEnsemble
from humanizer.humanizers.base import HumanizeResult, Humanizer
from humanizer.pipeline import Pipeline, PipelineConfig


class _StubHumanizer(Humanizer):
    """Returns a fixed list of candidates from `pool` cycling through; humanize() picks first."""

    name = "stub"

    def __init__(self, pool: List[str]):
        self.pool = pool
        self.calls = 0

    def humanize(self, text: str, **_) -> HumanizeResult:
        self.calls += 1
        return HumanizeResult(original=text, text=self.pool[0])

    def sample(self, text: str, n: int) -> List[str]:
        self.calls += 1
        return [self.pool[i % len(self.pool)] for i in range(n)]


class _StubDetector(Detector):
    def __init__(self, name: str, p_for_text: dict):
        self.name = name
        self.p_for_text = p_for_text

    def score(self, text: str) -> float:
        # Default to 0.5 if text isn't in the lookup.
        return float(self.p_for_text.get(text, 0.5))


def test_pipeline_with_no_humanizer_runs_scrub_only():
    """Without a humanizer, the pipeline still runs scrub + burstiness."""
    text = "Furthermore, organizations leverage the intricate complexities of AI to navigate today's landscape."
    p = Pipeline(humanizer=None, detectors=None, config=PipelineConfig(do_qa_gate=False))
    res = p.run(text)
    assert "Furthermore" not in res.text
    assert "leverage" not in res.text.lower()
    assert "intricate" not in res.text.lower()
    # Stages 1 and 5 should have run; stages 2-4 skipped.
    stage_names = [s.name for s in res.stages]
    assert "scrub" in stage_names
    assert "burstiness" in stage_names
    assert not any("paraphrase" in n for n in stage_names)


def test_pipeline_select_picks_lowest_p_ai_candidate():
    """When detectors are wired, best-of-N selection should pick the lowest-p_ai candidate."""
    cands = ["worst output", "medium output", "best output"]
    stub_h = _StubHumanizer(cands)
    detector = _StubDetector("d1", {
        "worst output": 0.95,
        "medium output": 0.50,
        "best output":  0.05,
    })
    ensemble = DetectorEnsemble([detector])

    config = PipelineConfig(
        n_candidates=3, do_scrub=False, do_burstiness=False, do_qa_gate=False,
        do_refine=False, similarity_threshold=0.0,  # accept any candidate
    )
    p = Pipeline(humanizer=stub_h, detectors=ensemble, config=config)
    res = p.run("input text")
    assert res.text == "best output"


def test_pipeline_returns_per_stage_trace():
    text = "Furthermore, the leverage of paradigms is paramount."
    p = Pipeline(humanizer=None, detectors=None,
                 config=PipelineConfig(do_qa_gate=False))
    res = p.run(text)
    assert len(res.stages) >= 2  # scrub + burstiness at minimum
    for stage in res.stages:
        assert stage.text_before is not None
        assert stage.text_after is not None


def test_pipeline_disable_individual_stages():
    """Every stage flag should be respected."""
    text = "Furthermore, intricate complexities."
    config = PipelineConfig(
        do_scrub=False, do_paraphrase=False, do_burstiness=False, do_qa_gate=False,
    )
    p = Pipeline(humanizer=None, detectors=None, config=config)
    res = p.run(text)
    # Nothing did anything.
    assert res.text == text
    assert res.stages == []


def test_pipeline_final_pattern_score_present():
    """final_pattern is computed even without an LLM or detectors."""
    p = Pipeline(humanizer=None, detectors=None,
                 config=PipelineConfig(do_qa_gate=False))
    res = p.run("Furthermore, leverage paradigms.")
    assert res.final_pattern is not None
    assert 0.0 <= res.final_pattern <= 1.0
