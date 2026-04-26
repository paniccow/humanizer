"""Prompted-LLM humanizer — the simplest baseline.

Wraps any chat-style LLM (local Transformers model OR an OpenAI-compatible API)
behind a single `humanize(text)` call. The prompt is engineered against the
known weaknesses detectors look for: low perplexity, low burstiness, AI-typical
phrasings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .base import HumanizeResult, Humanizer

_SYSTEM_PROMPT = """You rewrite AI-generated text so it reads as if a real \
person wrote it, while preserving meaning, factual content, and approximate \
length.

Rules:
1. Preserve every fact, number, named entity, and claim in the source. Do not \
add new facts or remove existing ones.
2. Vary sentence length aggressively. Mix very short sentences (3-7 words) with \
longer, more complex ones (20-35 words). Do NOT settle into a uniform rhythm.
3. Replace stiff transitional phrases (Furthermore, Moreover, Additionally, \
In conclusion) with conversational ones, or drop them entirely.
4. Use contractions (it's, don't, that's) where they read naturally.
5. Avoid AI-favorite words: delve, leverage, navigate, tapestry, intricate, \
crucial, paramount, multifaceted. Use plain alternatives.
6. Allow the occasional sentence fragment, or a comma splice, where a human \
writer naturally would. Don't force it.
7. Keep the original tone (formal/casual). If the source is academic, stay \
academic but loosen the rhythm.

Output ONLY the rewritten text. No preamble, no explanation, no quotes.
"""

_USER_TEMPLATE = "Rewrite the following text:\n\n---\n{text}\n---"


@dataclass
class PromptHumanizerConfig:
    model: str = "gpt-4o-mini"          # or any OpenAI-compatible model
    temperature: float = 0.85
    top_p: float = 0.95
    max_output_tokens: int = 2048
    base_url: str | None = None         # for OpenAI-compatible servers (Ollama, vLLM)
    api_key_env: str = "OPENAI_API_KEY"


class PromptHumanizer(Humanizer):
    """Single-shot prompted humanization through an OpenAI-compatible API."""

    name = "prompt"

    def __init__(self, config: PromptHumanizerConfig | None = None):
        self.config = config or PromptHumanizerConfig()
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "PromptHumanizer requires `openai`. Install with: "
                "pip install 'humanizer[openai]'"
            ) from e
        api_key = os.environ.get(self.config.api_key_env, "EMPTY")
        self._client = OpenAI(api_key=api_key, base_url=self.config.base_url)

    def _generate(self, text: str, *, n: int = 1) -> list[str]:
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
            ],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_output_tokens,
            n=n,
        )
        return [c.message.content.strip() for c in resp.choices]

    def humanize(self, text: str, **_) -> HumanizeResult:
        out = self._generate(text)[0]
        return HumanizeResult(original=text, text=out, attempts=1)

    def sample(self, text: str, n: int) -> list[str]:
        """Generate `n` candidate humanizations. Used by AdversarialHumanizer."""
        return self._generate(text, n=n)
