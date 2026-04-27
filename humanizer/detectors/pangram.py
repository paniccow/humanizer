"""Pangram (pangramlabs) API detector.

Wraps Pangram's text-classification endpoint. Pangram is a relatively
new entrant in 2024-2025 with strong reported accuracy on long-form
academic-style text. API access is paid; pricing on request via
https://www.pangram.com/api .

Endpoint (per public docs as of Apr 2026):
  POST https://api.pangram.com/v1/classify/text
  header: x-api-key
  body:   { "text": "..." }
  resp:   {
            "predicted_class": "ai" | "human" | "mixed",
            "ai_likelihood": 0.0..1.0,
            ...
          }

The exact response shape may evolve — _extract_p_ai tolerates both
`ai_likelihood` and `ai_probability` and `confidence` for class=ai.
Adjust _extract_p_ai if your subscription returns a different shape.
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
    endpoint: str = "https://api.pangram.com/v1/classify/text"
    timeout_s: float = 30.0
    retries: int = 2
    retry_backoff_s: float = 1.5


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
    def _extract_p_ai(payload: dict) -> float:
        # Common shapes — tolerate all of them.
        for k in ("ai_likelihood", "ai_probability", "ai_score"):
            if k in payload:
                return float(payload[k])
        # Class-probability style response.
        probs = payload.get("class_probabilities") or {}
        if probs:
            ai = float(probs.get("ai", 0.0))
            mixed = float(probs.get("mixed", 0.0))
            return min(1.0, max(0.0, ai + 0.5 * mixed))
        # Predicted-class + confidence style.
        if "predicted_class" in payload and "confidence" in payload:
            cls = payload["predicted_class"]
            conf = float(payload["confidence"])
            if cls == "ai":
                return conf
            if cls == "human":
                return 1.0 - conf
            if cls == "mixed":
                return 0.5
        raise RuntimeError(f"Pangram response missing AI-likelihood field: {payload}")

    def score(self, text: str) -> float:
        return self._extract_p_ai(self._post(text))
