"""Trained humanizer — loads a checkpoint produced by `humanizer.train.grpo`.

The trained policy is a (Q)LoRA adapter on top of a small instruction-tuned
base model (Qwen2.5-3B-Instruct by default). At inference we still use the
adversarial best-of-N wrapper — empirically it adds 5-15 ASR points on top of
the trained policy with negligible cost.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..detectors.base import pick_device
from .base import HumanizeResult, Humanizer
from .prompt import _SYSTEM_PROMPT, _USER_TEMPLATE


@dataclass
class TrainedHumanizerConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path: str | None = None
    device: str | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.85
    top_p: float = 0.95
    dtype: torch.dtype = torch.bfloat16


class TrainedHumanizer(Humanizer):
    name = "trained"

    def __init__(self, config: TrainedHumanizerConfig | None = None):
        self.config = config or TrainedHumanizerConfig()
        self.device = pick_device(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, torch_dtype=self.config.dtype
        ).to(self.device)
        if self.config.adapter_path:
            from peft import PeftModel  # imported lazily so non-train installs work
            self.model = PeftModel.from_pretrained(self.model, self.config.adapter_path)
        self.model.eval()

    def _build_prompt(self, text: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def _generate(self, text: str, n: int) -> list[str]:
        prompt = self._build_prompt(text)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            num_return_sequences=n,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = enc.input_ids.shape[1]
        completions = out[:, prompt_len:]
        return [
            self.tokenizer.decode(c, skip_special_tokens=True).strip() for c in completions
        ]

    def humanize(self, text: str, **_) -> HumanizeResult:
        return HumanizeResult(original=text, text=self._generate(text, n=1)[0], attempts=1)

    def sample(self, text: str, n: int) -> list[str]:
        return self._generate(text, n=n)
