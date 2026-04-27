"""Multi-stage humanization pipeline orchestrator.

Composes the deterministic scrub, an LLM paraphrase, detector + pattern
guided best-of-N, optional iterative refinement, and the burstiness
post-process. Returns a `PipelineResult` with the final text and a per-stage
trace so you can debug *which* stage moved the needle.

The pipeline is parameterized: any stage can be disabled. Even with all
LLM stages off, just `scrub` + `burstiness post-process` is a useful tool
that runs in milliseconds without GPU or API.

Composition with the trained adapter:
  - Set `humanizer` to a `TrainedHumanizer` (loads the GRPO LoRA adapter)
  - Stage 3 best-of-N now selects from candidates the trained model produced
  - Stages 1, 5 are deterministic and stack on top
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..humanizers.base import Humanizer
from ..patterns import analyze
from ..postprocess import BurstinessConfig, apply_burstiness
from .scrub import ScrubConfig, scrub


@dataclass
class PipelineConfig:
    # Stage toggles
    do_scrub: bool = True
    do_paraphrase: bool = True
    do_select: bool = True               # best-of-N
    do_refine: bool = True               # iterate paraphrase+select
    do_burstiness: bool = True
    do_qa_gate: bool = True              # abort to prior best on similarity collapse

    # Best-of-N + refinement knobs
    n_candidates: int = 8
    max_refine_passes: int = 2
    target_p_ai: float = 0.20            # stop refining once below this
    target_pattern: float = 0.30
    similarity_threshold: float = 0.70
    detector_weight: float = 0.6
    pattern_weight: float = 0.3
    similarity_weight: float = 0.1

    # Substage configs
    scrub: ScrubConfig = field(default_factory=ScrubConfig)
    burstiness: BurstinessConfig = field(default_factory=BurstinessConfig)


@dataclass
class StageTrace:
    name: str
    text_before: str
    text_after: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    original: str
    text: str                             # final humanized text
    stages: list[StageTrace] = field(default_factory=list)
    final_p_ai: float | None = None
    final_pattern: float | None = None
    final_similarity: float | None = None


class Pipeline:
    """Composable multi-stage humanizer.

    `humanizer` is the LLM paraphraser used for stages 2-4 (any object exposing
    `.sample(text, n)` and `.humanize(text)`). `detectors` is the ensemble
    used for selection rewards and the QA gate. Both are optional — without
    them the pipeline degrades to scrub + post-process.
    """

    def __init__(
        self,
        humanizer: Humanizer | None = None,
        detectors: Any | None = None,         # DetectorEnsemble; loosely typed to avoid eager import
        config: PipelineConfig | None = None,
    ):
        self.humanizer = humanizer
        self.detectors = detectors
        self.config = config or PipelineConfig()

    # ---- main ------------------------------------------------------------

    def run(self, text: str) -> PipelineResult:
        cfg = self.config
        original = text
        result = PipelineResult(original=original, text=text)

        if cfg.do_scrub:
            text = self._stage_scrub(text, result)
        if cfg.do_paraphrase and self.humanizer is not None:
            text = self._stage_paraphrase_and_select(text, original, result)
            if cfg.do_refine:
                text = self._stage_refine(text, original, result)
        if cfg.do_burstiness:
            text = self._stage_burstiness(text, result)
        if cfg.do_qa_gate:
            text = self._stage_qa_gate(text, original, result)

        result.text = text
        result.final_p_ai, result.final_pattern, result.final_similarity = self._final_scores(text, original)
        return result

    # ---- stages ----------------------------------------------------------

    def _stage_scrub(self, text: str, result: PipelineResult) -> str:
        before = text
        sr = scrub(text, self.config.scrub)
        result.stages.append(StageTrace(
            name="scrub",
            text_before=before,
            text_after=sr.text,
            metadata={"edits": sr.edits, "by_kind": sr.edits_by_kind},
        ))
        return sr.text

    def _stage_paraphrase_and_select(self, text: str, original: str, result: PipelineResult) -> str:
        before = text
        # If we don't have detectors, just take a single sample.
        if self.detectors is None or not self.config.do_select:
            chosen = self.humanizer.humanize(text).text
            result.stages.append(StageTrace(
                name="paraphrase",
                text_before=before,
                text_after=chosen,
                metadata={"candidates": 1},
            ))
            return chosen

        cands = self.humanizer.sample(text, n=self.config.n_candidates)
        chosen, score, meta = self._select(original, cands)
        result.stages.append(StageTrace(
            name="paraphrase+select",
            text_before=before,
            text_after=chosen,
            metadata={**meta, "candidates": len(cands), "combined_score": score},
        ))
        return chosen

    def _stage_refine(self, text: str, original: str, result: PipelineResult) -> str:
        for i in range(self.config.max_refine_passes):
            p_ai, pattern, _ = self._final_scores(text, original)
            if p_ai is not None and p_ai <= self.config.target_p_ai and (
                pattern is None or pattern <= self.config.target_pattern
            ):
                break
            before = text
            cands = self.humanizer.sample(text, n=self.config.n_candidates)
            text, score, meta = self._select(original, cands)
            result.stages.append(StageTrace(
                name=f"refine_pass_{i+1}",
                text_before=before,
                text_after=text,
                metadata={**meta, "combined_score": score},
            ))
        return text

    def _stage_burstiness(self, text: str, result: PipelineResult) -> str:
        before = text
        out = apply_burstiness(text, self.config.burstiness)
        result.stages.append(StageTrace(
            name="burstiness",
            text_before=before,
            text_after=out,
        ))
        return out

    def _stage_qa_gate(self, text: str, original: str, result: PipelineResult) -> str:
        # Find the latest stage where similarity was OK; if current dropped, revert.
        sim = self._sim(original, text) if self._can_score() else None
        if sim is None or sim >= self.config.similarity_threshold:
            return text
        # Walk stages backwards looking for a passing candidate.
        for stage in reversed(result.stages):
            cand = stage.text_after
            cand_sim = self._sim(original, cand)
            if cand_sim >= self.config.similarity_threshold:
                result.stages.append(StageTrace(
                    name="qa_gate_revert",
                    text_before=text,
                    text_after=cand,
                    metadata={"reason": "similarity_collapse",
                              "current_sim": float(sim),
                              "reverted_sim": float(cand_sim),
                              "reverted_to": stage.name},
                ))
                return cand
        return text  # nothing passes; ship as-is

    # ---- scoring helpers ------------------------------------------------

    def _select(self, original: str, candidates: list[str]) -> tuple[str, float, dict]:
        cfg = self.config
        if self.detectors is None:
            # Only patterns + similarity to score with.
            scored = []
            for c in candidates:
                fp = analyze(c).aggregate
                sim = self._sim(original, c)
                score = cfg.pattern_weight * (1 - fp) + cfg.similarity_weight * sim
                scored.append((c, score, {"pattern": fp, "similarity": sim}))
        else:
            ensemble_results = self.detectors.score_batch(candidates)
            scored = []
            for c, er in zip(candidates, ensemble_results):
                fp = analyze(c).aggregate
                sim = self._sim(original, c)
                score = (
                    cfg.detector_weight * (1 - er.aggregate)
                    + cfg.pattern_weight * (1 - fp)
                    + cfg.similarity_weight * sim
                )
                scored.append((c, score, {"p_ai": er.aggregate, "pattern": fp, "similarity": sim}))
        # Filter by similarity threshold first.
        kept = [t for t in scored if t[2].get("similarity", 1.0) >= cfg.similarity_threshold]
        pool = kept or scored
        best = max(pool, key=lambda t: t[1])
        return best[0], float(best[1]), {**best[2], "n_kept": len(kept), "n_total": len(candidates)}

    def _can_score(self) -> bool:
        try:
            from ..metrics.semantic import embedding_similarity  # noqa: F401
            return True
        except ImportError:
            return False

    def _sim(self, a: str, b: str) -> float:
        if not self._can_score():
            return 1.0  # if we can't score, don't filter
        from ..metrics.semantic import embedding_similarity
        return float(embedding_similarity(a, b).item())

    def _final_scores(self, text: str, original: str) -> tuple[float | None, float | None, float | None]:
        p_ai = None
        if self.detectors is not None:
            p_ai = float(self.detectors.score(text).aggregate)
        pattern = float(analyze(text).aggregate)
        similarity = self._sim(original, text) if self._can_score() else None
        return p_ai, pattern, similarity
