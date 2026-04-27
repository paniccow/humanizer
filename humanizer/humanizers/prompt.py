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
person wrote it. The current draft will be checked by AI-detection tools \
(GPTZero, Originality.ai, Turnitin, Copyleaks, ZeroGPT). Your job is to \
make those tools fail.

Those tools look for these specific signals — break ALL of them:

1. PERPLEXITY (predictability): AI text picks the highest-probability next word. \
Pick less obvious word choices. Use specific concrete words instead of generic \
abstract ones. Use idioms, colloquialisms, mild slang where appropriate to the \
context. A human's vocabulary is more idiosyncratic.

2. BURSTINESS (sentence-length variance): AI clusters around 15-25 words per \
sentence. Mix 3-word fragments with 30+ word sentences in the SAME paragraph. \
Use occasional one-word sentences. Use occasional run-ons. The variance itself \
is the signal.

3. PERSONAL VOICE: AI text has no narrator. Inject one. Use 'I', 'we', 'you' \
naturally. Add a personal observation, opinion, or aside. Real humans have \
context the AI lacks: a memory, a peeve, a take.

4. NATURAL IMPERFECTION: At least one of the following per paragraph: a comma \
splice, a sentence fragment, an unusual word order, a parenthetical aside, a \
trailing thought. These break the over-polished AI register.

5. AI-FAVORITE VOCABULARY — never use these: delve, leverage, navigate, tapestry, \
intricate, crucial, paramount, multifaceted, robust, seamless, comprehensive, \
holistic, transformative, groundbreaking, ecosystem, landscape, realm, \
foster, harness. Plain alternatives only.

6. STIFF TRANSITIONS — never use: Furthermore, Moreover, Additionally, \
Consequently, Thus, Therefore (at sentence start), In conclusion, Overall. \
Use 'and', 'but', 'so', 'still', 'though', or just drop the connector entirely.

7. CONTRACTIONS: ALWAYS contract where possible (don't, can't, won't, it's, \
that's, you're, they're, we're, I'm, we've). Failing to contract is the #1 \
AI tell.

8. PRESERVE every fact, number, name, date, claim. Do not add facts. Do not \
remove facts.

OUTPUT: only the rewritten text. No preamble. No quotes around the output. \
No explanations. The output should sound like it was typed quickly by a \
specific person with opinions, not generated."""

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

    def _generate(
        self,
        text: str,
        *,
        n: int = 1,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        # Try a single API call with n=N first (cheaper if supported).
        # Many OpenRouter routes silently return only 1 completion regardless
        # of the n parameter. If we asked for n>1 and got back fewer, fall
        # back to N independent calls.
        params = dict(
            model=self.config.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
            ],
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=self.config.top_p if top_p is None else top_p,
            max_tokens=self.config.max_output_tokens,
        )
        first = self._client.chat.completions.create(**params, n=n)
        outs = [c.message.content.strip() for c in first.choices]
        if len(outs) >= n or n == 1:
            return outs
        # Provider didn't honor n; fan out the rest.
        for _ in range(n - len(outs)):
            r = self._client.chat.completions.create(**params)
            outs.append(r.choices[0].message.content.strip())
        return outs

    def humanize(self, text: str, **_) -> HumanizeResult:
        out = self._generate(text)[0]
        return HumanizeResult(original=text, text=out, attempts=1)

    def sample(
        self,
        text: str,
        n: int,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        """Generate `n` candidate humanizations. Used by AdversarialHumanizer
        and RejectionSamplingHumanizer. Optional temperature/top_p override
        per call so callers can ramp diversity across rounds."""
        return self._generate(text, n=n, temperature=temperature, top_p=top_p)
