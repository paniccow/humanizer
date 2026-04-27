"""GPTZero API detector — the real-world judge.

Wraps https://api.gptzero.me/v2/predict/text. Reads the API key from
GPTZERO_API_KEY (or whatever env var the config names). Returns
p_ai = ai_prob + 0.5 * mixed_prob — conservative aggregation that
treats "mixed" outputs as half-AI.

Used as the judge for RejectionSamplingHumanizer at inference time and
optionally as a held-out eval signal during training. NOT used in the
GRPO reward loop unless you want to burn API credits per step.
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
class GPTZeroConfig:
    api_key_env: str = "GPTZERO_API_KEY"
    endpoint: str = "https://api.gptzero.me/v2/predict/text"
    api_version: str = "2024-04-04"
    multilingual: bool = False
    timeout_s: float = 30.0
    retries: int = 2
    retry_backoff_s: float = 1.5
    mixed_weight: float = 0.5


class GPTZeroDetector(Detector):
    """Detector backed by the GPTZero public API."""

    name = "gptzero"

    def __init__(self, config: GPTZeroConfig | None = None, *, api_key: str | None = None):
        self.config = config or GPTZeroConfig()
        key = api_key if api_key is not None else os.environ.get(self.config.api_key_env)
        if not key:
            raise RuntimeError(
                f"GPTZeroDetector needs an API key. Set {self.config.api_key_env} "
                f"or pass api_key=... — get one at https://gptzero.me/"
            )
        self._api_key = key

    def _post(self, document: str) -> dict:
        body = json.dumps({
            "document": document,
            "version": self.config.api_version,
            "multilingual": self.config.multilingual,
        }).encode("utf-8")
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
        raise RuntimeError(f"GPTZero API failed after {self.config.retries + 1} attempts: {last_err}")

    @staticmethod
    def _extract_p_ai(payload: dict, mixed_weight: float) -> float:
        # GPTZero returns one document per request. Schema (as of 2024-04):
        #   { "documents": [ { "class_probabilities": {"ai": .., "mixed": .., "human": ..},
        #                      "predicted_class": "ai|mixed|human", ... } ] }
        # We accept either "completely_generated_prob" (older) or class_probabilities (newer).
        docs = payload.get("documents") or []
        if not docs:
            raise RuntimeError(f"GPTZero returned no documents: {payload}")
        doc = docs[0]
        probs = doc.get("class_probabilities")
        if probs is not None:
            ai = float(probs.get("ai", 0.0))
            mixed = float(probs.get("mixed", 0.0))
            return min(1.0, max(0.0, ai + mixed_weight * mixed))
        # Older schema fallback.
        if "completely_generated_prob" in doc:
            return float(doc["completely_generated_prob"])
        if "average_generated_prob" in doc:
            return float(doc["average_generated_prob"])
        raise RuntimeError(f"GPTZero response missing probability fields: {doc}")

    def score(self, text: str) -> float:
        payload = self._post(text)
        return self._extract_p_ai(payload, self.config.mixed_weight)
