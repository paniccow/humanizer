"""Originality.ai API detector.

Wraps the Originality AI Detection v1 endpoint
  POST https://api.originality.ai/api/v1/scan/ai
with header X-OAI-API-KEY. Reads the key from ORIGINALITY_API_KEY by
default. Pricing as of Apr 2026 is roughly $14.95/mo for 3M words —
substantially cheaper than GPTZero, comparable real-world accuracy.

Response schema (per their docs):
  {
    "success": true,
    "score": { "ai": 0.0..1.0, "original": 0.0..1.0 },
    "credits_used": int,
    "credits": int,
    ...
  }

We return p_ai = score.ai. If a "mixed"/"borderline" field appears in
future versions, treat it half-weight like GPTZero.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .base import Detector


@dataclass
class OriginalityConfig:
    api_key_env: str = "ORIGINALITY_API_KEY"
    endpoint: str = "https://api.originality.ai/api/v1/scan/ai"
    title: str = "humanizer-rejection"   # optional metadata Originality logs
    ai_model_version: str = "1"          # "1" = standard, "lite" = cheaper / weaker
    timeout_s: float = 30.0
    retries: int = 2
    retry_backoff_s: float = 1.5


class OriginalityDetector(Detector):
    """Detector backed by the Originality.ai public API."""

    name = "originality"

    def __init__(self, config: OriginalityConfig | None = None, *, api_key: str | None = None):
        self.config = config or OriginalityConfig()
        key = api_key if api_key is not None else os.environ.get(self.config.api_key_env)
        if not key:
            raise RuntimeError(
                f"OriginalityDetector needs an API key. Set {self.config.api_key_env} "
                f"or pass api_key=... — get one at https://originality.ai/api-access"
            )
        self._api_key = key

    def _post(self, content: str) -> dict:
        body = json.dumps({
            "content": content,
            "title": self.config.title,
            "aiModelVersion": self.config.ai_model_version,
            "storeScan": "false",
        }).encode("utf-8")
        req = urllib.request.Request(
            self.config.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-OAI-API-KEY": self._api_key,
            },
            method="POST",
        )
        last_err: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                if attempt < self.config.retries:
                    time.sleep(self.config.retry_backoff_s * (attempt + 1))
        raise RuntimeError(
            f"Originality API failed after {self.config.retries + 1} attempts: {last_err}"
        )

    @staticmethod
    def _extract_p_ai(payload: dict) -> float:
        if payload.get("success") is False:
            raise RuntimeError(f"Originality API returned failure: {payload}")
        score = payload.get("score") or {}
        if "ai" in score:
            return float(score["ai"])
        # Some plans return a flat `ai_score` field; tolerate both.
        if "ai_score" in payload:
            return float(payload["ai_score"])
        raise RuntimeError(f"Originality response missing 'score.ai': {payload}")

    def score(self, text: str) -> float:
        return self._extract_p_ai(self._post(text))
