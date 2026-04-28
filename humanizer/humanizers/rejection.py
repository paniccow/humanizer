"""Rejection-sampling humanizer — the "money-back guarantee" layer.

The math: if our base humanizer's single-shot success rate against the
target detector is p, then best-of-N rejection sampling lifts the per-
request success rate to 1 - (1 - p)^N. So with p = 0.7 and N = 8 we
get 99.99%; with p = 0.5 and N = 8 we get 99.6%.

What it does, per request:
  1. Sample N candidates from the base humanizer.
  2. Filter by semantic similarity to the original (preserves meaning).
  3. Score each survivor through the JUDGE detector (typically the
     real-world target — GPTZero, Originality, etc., not the open
     training detectors).
  4. If any candidate has p_ai < strict_threshold, return it.
  5. Otherwise, escalate (bump temperature, optionally bump model) and
     retry up to max_rounds.
  6. If exhausted, return the best-found-so-far with passed=False so
     callers can decide what to do (refund, return anyway, or escalate
     to a human).

Cost model with GPTZero as judge:
  - 8 OpenRouter completions ≈ $0.008
  - 8 GPTZero scores ≈ $0.024
  - Worst case 4 rounds = $0.13 / request
  - Typical case (passes round 1) = $0.04 / request
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..detectors.base import Detector
from .base import HumanizeResult, Humanizer


def embedding_similarity(*args, **kwargs):
    """Lazy wrapper around metrics.semantic.embedding_similarity.

    The underlying function imports torch + sentence-transformers, which
    are heavy and not required for module-import or for tests that
    monkeypatch this symbol. Production callers pay the import cost on
    first use.
    """
    from ..metrics.semantic import embedding_similarity as _real
    return _real(*args, **kwargs)


@dataclass
class RejectionConfig:
    """Knobs for inference-time rejection sampling.

    candidates_per_round
        How many parallel humanizations to generate per round.
    max_rounds
        Number of escalation rounds before giving up. Worst-case API
        spend = candidates_per_round * max_rounds.
    p_ai_threshold
        A candidate "passes" if judge.score(text) < this. 0.05 = strict
        ("clearly human" verdict). Use 0.5 for "more human than AI".
    similarity_threshold
        Cosine similarity floor against the original. Below this we
        consider the candidate to have lost meaning. AdversarialHumanizer
        uses 0.78 with MiniLM by default.
    similarity_model
        Sentence-transformer model id for similarity. MiniLM is fast.
    temperature_ramp
        Per-round temperature override list. None means use the base
        humanizer's configured temperature. Length must be >= max_rounds
        or it cycles.
    early_exit_p_ai
        If a candidate scores below this, return immediately without
        scoring the rest of the batch. Saves judge calls. None = score
        all candidates, return the best.
    fallback_to_best
        If no round passes the strict threshold, return the lowest-p_ai
        candidate seen across all rounds (with passed=False) instead of
        raising.
    """

    candidates_per_round: int = 8
    max_rounds: int = 4
    p_ai_threshold: float = 0.05
    similarity_threshold: float = 0.78
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    temperature_ramp: list[float] | None = field(
        default_factory=lambda: [0.85, 1.0, 1.15, 1.25]
    )
    # Strategy carousel — list of Strategy objects (system_prompt + temperature)
    # cycled across rounds. When set, OVERRIDES temperature_ramp (each strategy
    # has its own temperature). Forces fundamentally different text shapes per
    # round, not just diversity within the same shape. Higher chance of
    # crossing Pangram's "human" boundary.
    prompt_strategies: list | None = None  # list[Strategy]; None = no carousel
    early_exit_p_ai: float | None = 0.05
    fallback_to_best: bool = True
    # Optional fact-preservation gate. If > 0, candidates whose
    # entity_overlap (numbers, dates, $ amounts, proper nouns) with the
    # original is below this threshold are dropped before judge scoring.
    # 0.0 disables (default — false-positives possible since the metric is
    # heuristic). Turn on for high-stakes inputs (academic essays, news,
    # technical docs) where dropping a number or date is unacceptable.
    preservation_threshold: float = 0.0
    # When > 1, score candidates concurrently via ThreadPoolExecutor.
    # Real win for HTTP-API judges (GPTZero / Originality / Pangram) — an
    # 8-candidate batch drops from 8 × ~2s serial to ~2s parallel. Local
    # RoBERTa judges don't benefit much (already GPU-batched internally).
    # Disables early-exit (we get all scores back at once), so set to 1
    # if your judge is expensive AND you want short-circuit semantics.
    concurrent_judge_calls: int = 1


def _temp_for_round(cfg: RejectionConfig, round_idx: int) -> float | None:
    if not cfg.temperature_ramp:
        return None
    return cfg.temperature_ramp[min(round_idx, len(cfg.temperature_ramp) - 1)]


def _strategy_for_round(cfg: RejectionConfig, round_idx: int):
    """Pick the strategy for this round (or None if no carousel configured).
    Cycles through cfg.prompt_strategies; if max_rounds > len(strategies),
    extra rounds repeat the last strategy."""
    if not cfg.prompt_strategies:
        return None
    return cfg.prompt_strategies[min(round_idx, len(cfg.prompt_strategies) - 1)]


class RejectionSamplingHumanizer(Humanizer):
    """Best-of-N humanizer that judges against the real target detector.

    Unlike AdversarialHumanizer (which always picks lowest-p_ai across a
    fixed candidate pool), this one keeps generating until a candidate
    crosses a STRICT threshold against the live judge — this is the
    operating mode of commercial services that advertise high evasion
    rates. Pair with a paid detector API (GPTZero etc.) for real-world
    reliability.

    Args:
        base: any Humanizer with a `sample(text, n, *, temperature=None,
            top_p=None) -> list[str]` method. PromptHumanizer satisfies
            this.
        judge: a single Detector representing the target you want to
            beat. Typically GPTZeroDetector. For free local validation,
            pass a RoBERTaDetector.
        config: RejectionConfig.
        on_round: optional callback(round_idx, candidates, scores) for
            tracing/telemetry. Not called if absent.
    """

    name = "rejection"

    def __init__(
        self,
        base: Humanizer,
        judge: Detector,
        config: RejectionConfig | None = None,
        *,
        on_round: Callable[[int, list[str], list[float]], None] | None = None,
    ):
        if not hasattr(base, "sample"):
            raise TypeError(
                f"{type(base).__name__} must implement .sample(text, n, *, temperature=...)"
            )
        self.base = base
        self.judge = judge
        self.config = config or RejectionConfig()
        self._on_round = on_round

    def _filter_by_similarity(
        self, original: str, candidates: list[str]
    ) -> tuple[list[str], list[float]]:
        if not candidates:
            return [], []
        sims = embedding_similarity(
            [original] * len(candidates),
            candidates,
            model_id=self.config.similarity_model,
        ).tolist()
        kept_pairs = [
            (c, s) for c, s in zip(candidates, sims) if s >= self.config.similarity_threshold
        ]
        if kept_pairs:
            return [c for c, _ in kept_pairs], [s for _, s in kept_pairs]
        return [], sims  # similarity values still useful for telemetry

    def _filter_by_facts(self, original: str, candidates: list[str]) -> list[str]:
        """Optional: drop candidates that drop too many source facts."""
        if self.config.preservation_threshold <= 0.0 or not candidates:
            return candidates
        from ..metrics.facts import entity_overlap
        return [c for c in candidates if entity_overlap(original, c) >= self.config.preservation_threshold]

    def humanize(self, text: str, **_) -> HumanizeResult:
        cfg = self.config
        best_text: str | None = None
        best_p_ai: float = 1.0
        best_round: int | None = None
        best_breakdown: dict | None = None
        rounds_used = 0
        total_attempts = 0
        total_judge_calls = 0
        passed = False

        for round_idx in range(cfg.max_rounds):
            rounds_used = round_idx + 1
            strategy = _strategy_for_round(cfg, round_idx)
            # Strategy carousel takes precedence: when set, the strategy's
            # system prompt + temperature override the generic temperature ramp.
            if strategy is not None:
                cands = self.base.sample(
                    text, n=cfg.candidates_per_round,
                    temperature=strategy.temperature,
                    system_prompt=strategy.system_prompt,
                )
            else:
                temp = _temp_for_round(cfg, round_idx)
                cands = self.base.sample(text, n=cfg.candidates_per_round, temperature=temp)
            total_attempts += len(cands)

            kept, sims = self._filter_by_similarity(text, cands)
            kept = self._filter_by_facts(text, kept)
            if not kept:
                # Whole batch lost meaning OR dropped facts. Try harder next round.
                if self._on_round:
                    self._on_round(round_idx, cands, [])
                continue

            # Score survivors through the judge.
            # Two paths:
            #   - concurrent_judge_calls > 1: score all candidates in parallel
            #     via ThreadPoolExecutor (no early-exit — we want all scores
            #     back). Best for HTTP-API judges where each call is ~2s.
            #   - concurrent_judge_calls <= 1: serial with early-exit on first
            #     pass. Saves judge calls when typical case passes round 1.
            scores: list[float] = []
            chosen_idx: int | None = None
            chosen_breakdown: dict | None = None
            if cfg.concurrent_judge_calls > 1 and len(kept) > 1:
                from concurrent.futures import ThreadPoolExecutor
                workers = min(cfg.concurrent_judge_calls, len(kept))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    scores = [float(p) for p in pool.map(self.judge.score, kept)]
                total_judge_calls += len(kept)
                # Pick the best (lowest p_ai) candidate. last_breakdown is
                # not safe here (race across threads on EnsembleJudge), so we
                # re-score the chosen one below the loop.
            else:
                for i, cand in enumerate(kept):
                    p = float(self.judge.score(cand))
                    scores.append(p)
                    total_judge_calls += 1
                    if cfg.early_exit_p_ai is not None and p < cfg.early_exit_p_ai:
                        chosen_idx = i
                        # If the judge is an EnsembleJudge, capture per-detector
                        # scores for the chosen candidate. last_breakdown is
                        # populated by EnsembleJudge.score() — single detectors
                        # don't expose this attribute.
                        chosen_breakdown = getattr(self.judge, "last_breakdown", None)
                        if chosen_breakdown is not None:
                            chosen_breakdown = dict(chosen_breakdown)  # snapshot
                        break

            if self._on_round:
                self._on_round(round_idx, kept, scores)

            # Track best across all rounds for fallback.
            round_best_idx = (
                chosen_idx if chosen_idx is not None else min(range(len(scores)), key=lambda i: scores[i])
            )
            round_best_p = scores[round_best_idx]

            # If we DIDN'T early-exit, the chosen candidate is the lowest-p_ai
            # one in this batch — re-score the chosen one to refresh the
            # ensemble's last_breakdown for it (cheaper than tracking through
            # the loop, since the score is already cached above).
            if chosen_breakdown is None:
                # Re-trigger the judge so EnsembleJudge.last_breakdown reflects
                # the chosen candidate. No effect for single detectors.
                if hasattr(self.judge, "last_breakdown"):
                    self.judge.score(kept[round_best_idx])
                    bd = getattr(self.judge, "last_breakdown", None)
                    chosen_breakdown = dict(bd) if bd else None

            if round_best_p < best_p_ai:
                best_p_ai = round_best_p
                best_text = kept[round_best_idx]
                best_round = round_idx
                best_breakdown = chosen_breakdown

            # Strict-threshold pass: return immediately.
            if round_best_p < cfg.p_ai_threshold:
                passed = True
                meta = {
                    "passed": True,
                    "rounds_used": rounds_used,
                    "judge": self.judge.name,
                    "judge_calls": total_judge_calls,
                    "best_p_ai": round_best_p,
                    "best_round": round_idx,
                    "threshold": cfg.p_ai_threshold,
                }
                if chosen_breakdown:
                    meta["per_detector"] = chosen_breakdown
                return HumanizeResult(
                    original=text,
                    text=kept[round_best_idx],
                    score=round_best_p,
                    attempts=total_attempts,
                    metadata=meta,
                )

        # Exhausted without crossing strict threshold.
        if best_text is None:
            if not cfg.fallback_to_best:
                raise RuntimeError(
                    "All candidates failed similarity filter across all rounds; "
                    "no fallback available."
                )
            # Every round lost meaning. Fall back to original text — bad outcome
            # but at least preserves the user's input.
            best_text = text
            best_p_ai = float(self.judge.score(text))
            total_judge_calls += 1

        meta = {
            "passed": passed,
            "rounds_used": rounds_used,
            "judge": self.judge.name,
            "judge_calls": total_judge_calls,
            "best_p_ai": best_p_ai,
            "best_round": best_round,
            "threshold": cfg.p_ai_threshold,
        }
        if best_breakdown:
            meta["per_detector"] = best_breakdown
        return HumanizeResult(
            original=text,
            text=best_text,
            score=best_p_ai,
            attempts=total_attempts,
            metadata=meta,
        )
