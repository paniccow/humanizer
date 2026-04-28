"""Pangram (pangramlabs) API detector.

Wraps Pangram's text-classification v3 endpoint. Pangram is a 2024-
entrant with strong reported accuracy on long-form academic-style
text. Self-serve credit-based pricing (no enterprise contract needed
for low volumes): roughly $0.05 per 1,000 words = $0.00005/word, so
a 200-word humanization costs ~$0.01.

Get a key: https://pangram.com → sign up → dashboard → API section.

Endpoint (verified Apr 2026 against https://docs.pangram.com/api-reference/ai-detection):
  POST https://text.api.pangramlabs.com/v3
  header: x-api-key
  body:   { "text": "..." }
  resp:   {
            "text": "...",
            "version": "...",
            "headline": "...",
            "prediction": "...",
            "prediction_short": "AI" | "AI-Assisted" | "Human" | "Mixed",
            "fraction_ai": 0.0..1.0,            <-- canonical p_ai source
            "fraction_ai_assisted": 0.0..1.0,
            "fraction_human": 0.0..1.0,
            "num_ai_segments": int,
            ...
            "windows": [ ... per-paragraph breakdown ... ]
          }

We compute p_ai = fraction_ai + 0.5 * fraction_ai_assisted (treating
"AI-assisted" as half-AI, matching the conservative aggregation we use
elsewhere). Override mixed_weight in the config to adjust.
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
class PangramConfig:
    api_key_env: str = "PANGRAM_API_KEY"
    endpoint: str = "https://text.api.pangramlabs.com/v3"
    timeout_s: float = 30.0
    retries: int = 2
    retry_backoff_s: float = 1.5
    # AI-assisted weight in the aggregate: how much of fraction_ai_assisted
    # to count toward p_ai. 0.5 = treat as half-AI (default). 1.0 = treat
    # as fully AI (strict). 0.0 = ignore (loose).
    ai_assisted_weight: float = 0.5


class PangramDetector(Detector):
    """Detector backed by the Pangram public API."""

    name = "pangram"

    def __init__(self, config: PangramConfig | None = None, *, api_key: str | None = None):
        self.config = config or PangramConfig()
        key = api_key if api_key is not None else os.environ.get(self.config.api_key_env)
        if not key:
            raise RuntimeError(
                f"PangramDetector needs an API key. Set {self.config.api_key_env} "
                f"or pass api_key=... — request access at https://www.pangram.com/api"
            )
        self._api_key = key

    def _post(self, text: str) -> dict:
        body = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            self.config.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": self._api_key,
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
            f"Pangram API failed after {self.config.retries + 1} attempts: {last_err}"
        )

    @staticmethod
    def _extract_p_ai(payload: dict, ai_assisted_weight: float = 0.5) -> float:
        # Canonical v3 shape: fraction_ai + fraction_ai_assisted.
        if "fraction_ai" in payload:
            ai = float(payload["fraction_ai"])
            assisted = float(payload.get("fraction_ai_assisted", 0.0))
            return min(1.0, max(0.0, ai + ai_assisted_weight * assisted))
        # Legacy / SDK shapes — tolerate them in case of older endpoint.
        for k in ("ai_likelihood", "ai_probability", "ai_score"):
            if k in payload:
                return float(payload[k])
        probs = payload.get("class_probabilities") or {}
        if probs:
            return min(1.0, max(0.0,
                float(probs.get("ai", 0.0)) + ai_assisted_weight * float(probs.get("mixed", 0.0))
            ))
        if "predicted_class" in payload and "confidence" in payload:
            cls, conf = payload["predicted_class"], float(payload["confidence"])
            if cls == "ai": return conf
            if cls == "human": return 1.0 - conf
            if cls == "mixed": return 0.5
        raise RuntimeError(f"Pangram response missing fraction_ai field: {payload}")

    def score(self, text: str) -> float:
        return self._extract_p_ai(self._post(text), self.config.ai_assisted_weight)
