"""FastAPI wrapper around RejectionSamplingHumanizer.

The deployable form of the system. Single endpoint, single config
object, lazy initialization so import is cheap and the first request
pays the model-load cost (subsequent requests are fast).

Deploy:

    pip install -e '.[serve,openai]'
    export OPENAI_API_KEY=sk-...                       # OpenRouter or OpenAI
    export ORIGINALITY_API_KEY=...                     # optional, for paid judge
    humanizer serve --port 8000

    # or with uvicorn directly
    uvicorn humanizer.service.app:app --host 0.0.0.0 --port 8000

Endpoints:

    POST /humanize          one humanization, returns text + metadata
    GET  /health            readiness probe + judge name
    GET  /version           build/version info

Auth (optional): set HUMANIZER_API_KEY to require a bearer token on
/humanize. /health and /version stay open so load balancers / monitors
can probe.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

# These imports at module-import time so a missing FastAPI is loud.
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field


@dataclass
class ServiceConfig:
    """Knobs for the deployed service. All env-var reads use default_factory
    so they're evaluated at instantiation time, not class-definition time —
    important so tests (and per-worker configs) see fresh values."""

    model: str = field(default_factory=lambda: os.environ.get("HUMANIZER_MODEL", "gpt-4o-mini"))
    base_url: Optional[str] = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL"))
    judge: str = field(default_factory=lambda: os.environ.get("HUMANIZER_JUDGE", "auto"))
    candidates_per_round: int = field(default_factory=lambda: int(os.environ.get("HUMANIZER_REJECT_N", "8")))
    max_rounds: int = field(default_factory=lambda: int(os.environ.get("HUMANIZER_REJECT_ROUNDS", "4")))
    p_ai_threshold: float = field(default_factory=lambda: float(os.environ.get("HUMANIZER_REJECT_THRESHOLD", "0.05")))
    similarity_threshold: float = field(default_factory=lambda: float(os.environ.get("HUMANIZER_SIM_THRESHOLD", "0.78")))
    api_key_env: str = "HUMANIZER_API_KEY"
    max_chars: int = field(default_factory=lambda: int(os.environ.get("HUMANIZER_MAX_CHARS", "20000")))


class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Source text to humanize.")
    max_rounds: Optional[int] = Field(None, ge=1, le=8, description="Override config max rounds.")
    candidates: Optional[int] = Field(None, ge=1, le=16, description="Override candidates per round.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override pass threshold.")


class HumanizeResponse(BaseModel):
    text: str
    passed: bool
    score: Optional[float] = None
    judge: Optional[str] = None
    rounds_used: Optional[int] = None
    judge_calls: Optional[int] = None
    per_detector: Optional[dict] = None  # populated when judge is EnsembleJudge
    attempts: int
    elapsed_ms: int


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1)


class DetectResponse(BaseModel):
    p_ai: float
    judge: str
    per_detector: Optional[dict] = None
    elapsed_ms: int


class SampleRequest(BaseModel):
    text: str = Field(..., min_length=1)
    n: int = Field(8, ge=1, le=16, description="Number of candidates to generate.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    score: bool = Field(True, description="Score each candidate via the judge.")


class CandidateScore(BaseModel):
    text: str
    p_ai: Optional[float] = None
    per_detector: Optional[dict] = None


class SampleResponse(BaseModel):
    candidates: List[CandidateScore]
    judge: Optional[str] = None
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: str
    model: str
    judge: str
    judge_resolved: Optional[str] = None
    paid_keys_set: List[str]


def _resolve_judge(judge_name: str):
    """Resolve the --judge string (or env) to a concrete Detector instance.
    Lazy — only called once on first request so model loads happen then."""
    if judge_name == "auto":
        from ..detectors import judge_from_env
        return judge_from_env()
    if judge_name == "gptzero":
        from ..detectors import GPTZeroDetector; return GPTZeroDetector()
    if judge_name == "originality":
        from ..detectors import OriginalityDetector; return OriginalityDetector()
    if judge_name == "pangram":
        from ..detectors import PangramDetector; return PangramDetector()
    if judge_name == "roberta":
        from ..detectors import RoBERTaDetector
        return RoBERTaDetector("roberta-large-openai-detector")
    raise ValueError(f"unknown judge: {judge_name!r}")


class _State:
    """Holds lazily-built humanizer + judge so they survive across requests
    in a single uvicorn worker. NOT shared across workers — for that you'd
    need an external cache / model server."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._rejection = None  # type: ignore[assignment]
        self._judge_resolved_name: Optional[str] = None

    def get_humanizer(self):
        if self._rejection is not None:
            return self._rejection
        from ..humanizers import (
            PromptHumanizer, PromptHumanizerConfig,
            RejectionConfig, RejectionSamplingHumanizer,
        )
        base = PromptHumanizer(PromptHumanizerConfig(
            model=self.config.model, base_url=self.config.base_url,
        ))
        judge = _resolve_judge(self.config.judge)
        self._judge_resolved_name = getattr(judge, "name", "?")
        rej_cfg = RejectionConfig(
            candidates_per_round=self.config.candidates_per_round,
            max_rounds=self.config.max_rounds,
            p_ai_threshold=self.config.p_ai_threshold,
            similarity_threshold=self.config.similarity_threshold,
        )
        self._rejection = RejectionSamplingHumanizer(base, judge, rej_cfg)
        return self._rejection

    @property
    def judge_resolved_name(self) -> Optional[str]:
        return self._judge_resolved_name


def _check_auth(state: _State, authorization: Optional[str]) -> None:
    expected = os.environ.get(state.config.api_key_env)
    if not expected:
        return  # auth disabled
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[len("Bearer "):]
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def build_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    """Build a FastAPI app instance. Mostly used for tests so they can
    construct an isolated app with a stubbed humanizer; production
    code uses the module-level `app` exported below."""
    cfg = config or ServiceConfig()
    state = _State(cfg)
    api = FastAPI(title="humanizer", version="0.5.0")
    api.state.svc = state

    # Optional: attach JSONL telemetry middleware if HUMANIZER_TELEMETRY_PATH
    # is set. No-op when unset — zero overhead in dev/test by default.
    if os.environ.get("HUMANIZER_TELEMETRY_PATH"):
        from .telemetry import TelemetryMiddleware
        api.add_middleware(TelemetryMiddleware)

    @api.get("/health", response_model=HealthResponse)
    async def health():
        from ..detectors import available_paid_detectors
        avail = [s.name for s in available_paid_detectors()]
        return HealthResponse(
            status="ready" if state._rejection is not None else "cold",
            model=cfg.model,
            judge=cfg.judge,
            judge_resolved=state.judge_resolved_name,
            paid_keys_set=avail,
        )

    @api.get("/version")
    async def version():
        return {"name": "humanizer", "version": "0.5.0"}

    @api.post("/humanize", response_model=HumanizeResponse)
    async def humanize(req: HumanizeRequest, request: Request, authorization: Optional[str] = Header(None)):
        _check_auth(state, authorization)
        if len(req.text) > cfg.max_chars:
            raise HTTPException(
                status_code=413,
                detail=f"text exceeds max_chars={cfg.max_chars} (got {len(req.text)})",
            )
        rejection = state.get_humanizer()
        # Per-request overrides re-bind cfg fields if provided. Cheaper to
        # mutate the same RejectionConfig in-place than to rebuild the wrapper
        # every request; the rejection sampler reads cfg at call time.
        if req.max_rounds is not None:
            rejection.config.max_rounds = req.max_rounds
        if req.candidates is not None:
            rejection.config.candidates_per_round = req.candidates
        if req.threshold is not None:
            rejection.config.p_ai_threshold = req.threshold
        t0 = time.time()
        result = rejection.humanize(req.text)
        elapsed_ms = int((time.time() - t0) * 1000)
        meta = result.metadata or {}
        return HumanizeResponse(
            text=result.text,
            passed=bool(meta.get("passed", False)),
            score=result.score,
            judge=meta.get("judge"),
            rounds_used=meta.get("rounds_used"),
            judge_calls=meta.get("judge_calls"),
            per_detector=meta.get("per_detector"),
            attempts=result.attempts,
            elapsed_ms=elapsed_ms,
        )

    @api.post("/sample", response_model=SampleResponse)
    async def sample(req: SampleRequest, authorization: Optional[str] = Header(None)):
        """Generate N humanization candidates and (optionally) score each.
        Useful for product UI flows that let the user pick their favorite
        from a slate of options. No rejection threshold applied — caller
        decides what to do with the scores.
        """
        _check_auth(state, authorization)
        if len(req.text) > cfg.max_chars:
            raise HTTPException(
                status_code=413,
                detail=f"text exceeds max_chars={cfg.max_chars}",
            )
        rejection = state.get_humanizer()
        base = rejection.base
        judge = rejection.judge

        t0 = time.time()
        cands = base.sample(req.text, n=req.n, temperature=req.temperature)
        candidates: List[CandidateScore] = []
        if req.score:
            for c in cands:
                p = float(judge.score(c))
                bd = getattr(judge, "last_breakdown", None)
                candidates.append(CandidateScore(
                    text=c, p_ai=p, per_detector=dict(bd) if bd else None,
                ))
        else:
            candidates = [CandidateScore(text=c) for c in cands]
        elapsed_ms = int((time.time() - t0) * 1000)
        return SampleResponse(
            candidates=candidates,
            judge=judge.name if req.score else None,
            elapsed_ms=elapsed_ms,
        )

    @api.post("/detect", response_model=DetectResponse)
    async def detect(req: DetectRequest, authorization: Optional[str] = Header(None)):
        """Score arbitrary text with the configured judge — useful for UI
        feedback ("your AI-likelihood is X") and for testing the judge
        configuration without consuming an LLM generation."""
        _check_auth(state, authorization)
        if len(req.text) > cfg.max_chars:
            raise HTTPException(
                status_code=413,
                detail=f"text exceeds max_chars={cfg.max_chars}",
            )
        # Get the same judge the rejection sampler uses (lazy-init shared state).
        rejection = state.get_humanizer()
        judge = rejection.judge
        t0 = time.time()
        p_ai = float(judge.score(req.text))
        elapsed_ms = int((time.time() - t0) * 1000)
        per_det = getattr(judge, "last_breakdown", None)
        return DetectResponse(
            p_ai=p_ai,
            judge=judge.name,
            per_detector=dict(per_det) if per_det else None,
            elapsed_ms=elapsed_ms,
        )

    return api


# Module-level app for `uvicorn humanizer.service.app:app`.
app = build_app()
