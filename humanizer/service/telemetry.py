"""Per-request telemetry — opt-in JSONL log of /humanize, /detect, /sample
calls. Flat-file by design: cheap to set up, easy to grep, easy to ship
to s3/bigquery later. One line per request:

    {"ts": "2026-04-27T18:01:23Z", "path": "/humanize",
     "status": 200, "elapsed_ms": 1245, "attempts": 8,
     "judge_calls": 8, "passed": true, "score": 0.02,
     "cost_estimate_usd": 0.024, "client_ip": "...",
     "error": null}

Enable by setting HUMANIZER_TELEMETRY_PATH=/var/log/humanizer.jsonl
(directory must be writable by the uvicorn worker). Disabled when
the env var is unset — no overhead, no file created.

Cost estimation is rough: assumes ~$0.001 per LLM generation
(gpt-4o-mini at OpenRouter rates) and ~$0.003 per paid-detector judge
call. Override with HUMANIZER_LLM_COST and HUMANIZER_JUDGE_COST env if
your pricing differs.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Routes whose response bodies hold the metrics we want to log.
_LOGGED_PATHS = {"/humanize", "/detect", "/sample"}


def _llm_cost_per_call() -> float:
    return float(os.environ.get("HUMANIZER_LLM_COST", "0.001"))


def _judge_cost_per_call() -> float:
    return float(os.environ.get("HUMANIZER_JUDGE_COST", "0.003"))


def _estimate_cost_usd(path: str, body: dict) -> float:
    """Rough cost estimate per request. Best-effort — derives from response
    fields that the endpoint already exposes (attempts, judge_calls)."""
    if path == "/humanize":
        attempts = body.get("attempts", 0) or 0
        judge_calls = body.get("judge_calls", 0) or 0
        return attempts * _llm_cost_per_call() + judge_calls * _judge_cost_per_call()
    if path == "/detect":
        return _judge_cost_per_call()
    if path == "/sample":
        cands = body.get("candidates") or []
        scored = sum(1 for c in cands if c.get("p_ai") is not None)
        return len(cands) * _llm_cost_per_call() + scored * _judge_cost_per_call()
    return 0.0


class TelemetryMiddleware(BaseHTTPMiddleware):
    """JSONL request logger. Disabled when HUMANIZER_TELEMETRY_PATH is unset."""

    def __init__(self, app, log_path: Optional[str] = None):
        super().__init__(app)
        self.log_path = log_path or os.environ.get("HUMANIZER_TELEMETRY_PATH")
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    async def dispatch(self, request: Request, call_next):
        if not self.log_path or request.url.path not in _LOGGED_PATHS:
            return await call_next(request)

        t0 = time.time()
        response = await call_next(request)
        elapsed_ms = int((time.time() - t0) * 1000)

        # Buffer the response body so we can read it AND still serve it.
        # Starlette streams response bodies, so we need to capture and re-emit.
        body_chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            body_chunks.append(chunk)
        body = b"".join(body_chunks)

        record: dict = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "path": request.url.path,
            "status": response.status_code,
            "elapsed_ms": elapsed_ms,
            "client_ip": request.client.host if request.client else None,
        }

        if response.status_code == 200:
            try:
                parsed = json.loads(body.decode("utf-8"))
                record["attempts"] = parsed.get("attempts")
                record["judge_calls"] = parsed.get("judge_calls")
                record["passed"] = parsed.get("passed")
                record["score"] = parsed.get("score") or parsed.get("p_ai")
                record["cost_estimate_usd"] = round(
                    _estimate_cost_usd(request.url.path, parsed), 6,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                record["error"] = "non-json-response"
        else:
            record["error"] = f"http_{response.status_code}"

        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            # Don't fail requests because telemetry can't write. Operators
            # can monitor disk separately.
            pass

        # Return a fresh response with the same body — call_next consumed
        # body_iterator so we can't return `response` directly.
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
