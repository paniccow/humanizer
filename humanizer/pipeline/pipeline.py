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
    do_reject: bool = False              # if True, replace paraphrase+select+refine
                                         #   with rejection sampling against `judge`
    do_burstiness: bool = True
    do_qa_gate: bool = True              # abort to prior best on similarity collapse

    # Rejection-sampling knobs (used when do_reject=True)
    reject_candidates: int = 8
    reject_max_rounds: int = 4
    reject_p_ai_threshold: float = 0.05

    # Best-of-N + refinement knobs
    n_candidates: int = 16               # bumped from 8 — costs more API calls but typically -0.05 p_ai
    max_refine_passes: int = 3           # bumped from 2 — most outputs converge in 1-2 passes
    target_p_ai: float = 0.20            # stop refining once below this
    target_pattern: float = 0.30
    similarity_threshold: float = 0.70
    detector_weight: float = 0.6
    pattern_weight: float = 0.3
    similarity_weight: float = 0.1
    # Refinement: when iterating, inject the still-flagged pattern signals
    # into the next prompt so the model knows what to fix.
    targeted_refine_prompts: bool = True

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
        judge: Any | None = None,             # single Detector used by do_reject mode
    ):
        self.humanizer = humanizer
        self.detectors = detectors
        self.config = config or PipelineConfig()
        self.judge = judge

    # ---- main ------------------------------------------------------------

    def run(self, text: str) -> PipelineResult:
        cfg = self.config
        original = text
        result = PipelineResult(original=original, text=text)

        if cfg.do_scrub:
            text = self._stage_scrub(text, result)
        if cfg.do_reject and self.humanizer is not None and self.judge is not None:
            text = self._stage_reject(text, original, result)
        elif cfg.do_paraphrase and self.humanizer is not None:
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

    def _stage_reject(self, text: str, original: str, result: PipelineResult) -> str:
        """Replace paraphrase+select+refine with rejection sampling against
        a real-world judge. Used when the operator has paid-detector keys
        and wants strict pass-or-keep-trying semantics."""
        from ..humanizers.rejection import (
            RejectionConfig, RejectionSamplingHumanizer,
        )

        cfg = self.config
        rej = RejectionSamplingHumanizer(
            self.humanizer,
            self.judge,
            RejectionConfig(
                candidates_per_round=cfg.reject_candidates,
                max_rounds=cfg.reject_max_rounds,
                p_ai_threshold=cfg.reject_p_ai_threshold,
                similarity_threshold=cfg.similarity_threshold,
            ),
        )
        before = text
        out = rej.humanize(original)  # rejection sampler operates on the source, not scrubbed text
        result.stages.append(StageTrace(
            name="reject",
            text_before=before,
            text_after=out.text,
            metadata={
                "passed": out.metadata.get("passed", False),
                "rounds_used": out.metadata.get("rounds_used"),
                "judge": out.metadata.get("judge"),
                "judge_calls": out.metadata.get("judge_calls"),
                "best_p_ai": out.metadata.get("best_p_ai"),
                "attempts": out.attempts,
            },
        ))
        return out.text

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
        """Iterate paraphrase+select. Each pass gets a *targeted* prompt that
        names the specific signals still firing (if `targeted_refine_prompts`).
        Stop early when both p_ai and pattern thresholds are met."""
        for i in range(self.config.max_refine_passes):
            p_ai, pattern, _ = self._final_scores(text, original)
            if p_ai is not None and p_ai <= self.config.target_p_ai and (
                pattern is None or pattern <= self.config.target_pattern
            ):
                break
            before = text
            # Targeted refinement: build an input that calls out which AI tells
            # are still firing, so the model knows what to fix this pass.
            target_text = self._build_refine_input(text) if self.config.targeted_refine_prompts else text
            cands = self.humanizer.sample(target_text, n=self.config.n_candidates)
            text, score, meta = self._select(original, cands)
            flagged = self._currently_flagged_signals(before)
            result.stages.append(StageTrace(
                name=f"refine_pass_{i+1}",
                text_before=before,
                text_after=text,
                metadata={**meta, "combined_score": score, "flagged_before": flagged},
            ))
        return text

    def _currently_flagged_signals(self, text: str) -> list[str]:
        """Which pattern signals still fire on the current candidate."""
        return analyze(text).flagged

    def _build_refine_input(self, text: str) -> str:
        """Wrap the text with explicit guidance on which AI tells still fire.
        The base humanizer's system prompt already says "make it human"; this
        gives it the *specific* axes that are still failing on THIS draft."""
        flagged = self._currently_flagged_signals(text)
        if not flagged:
            return text
        # Map signal names -> human-readable instructions for the refine pass.
        guidance = {
            "burstiness": "vary sentence length more — mix short fragments with longer sentences",
            "stiff_transitions": "remove stiff transitional phrases (Furthermore, Moreover, Additionally, In conclusion)",
            "favorite_words": "replace AI-favorite words (delve, leverage, intricate, multifaceted, paramount)",
            "em_dash_density": "reduce em-dash usage",
            "hedging": "remove hedging boilerplate (It's important to note that, In today's...)",
            "tricolons": "break up tricolons (X, Y, and Z patterns)",
            "contraction_deficit": "use contractions (don't / can't / it's / that's)",
            "ngram_repetition": "vary phrasing — you're repeating 4-grams",
            "type_token": "use a wider vocabulary range",
            "sentence_start_uniformity": "vary how sentences start — don't begin them the same way",
            "abstract_subject": "use concrete subjects (people, you, we, real names) instead of abstract nouns ('The system', 'The framework')",
            "enumeration_shape": "avoid formulaic enumerations ('Whether it's X or Y', 'Not only X but also Y', 'From X to Y')",
            "modality_overload": "reduce 'must / should / ought to' — make declarative claims instead",
        }
        instructions = "\n".join(f"- {guidance[f]}" for f in flagged if f in guidance)
        return (
            f"This draft still reads as AI on these axes — fix THESE specifically:\n"
            f"{instructions}\n\n"
            f"Draft:\n{text}"
        )

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
