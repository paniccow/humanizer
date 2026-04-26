"""Binoculars zero-shot AI-text detector (ICML 2024).

Reference: Hans et al. "Spotting LLMs With Binoculars: Zero-Shot Detection of
Machine-Generated Text". https://github.com/ahans30/Binoculars

Idea: AI text has unusually low *cross-perplexity* relative to its perplexity
under a closely related model. Binoculars score = log_perplexity(text | observer)
divided by cross-perplexity(text | observer || performer). Low score => AI.

We compress that to p_ai in [0, 1] via a logistic on the negated score so the
detector module has a uniform interface.
"""
from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Detector, pick_device

# Threshold from the Binoculars paper at FPR=0.01% on Falcon7B/Falcon7B-Instruct.
# Score below this => predict AI. We turn the gap into a probability via logistic.
_BINO_DEFAULT_THRESHOLD = 0.9015
_BINO_LOGISTIC_SCALE = 25.0  # tuned so that a 0.1 gap saturates near 0/1


class Binoculars(Detector):
    """Two-LLM zero-shot detector. Heavyweight — only enable on a GPU machine."""

    name = "binoculars"

    def __init__(
        self,
        observer_id: str = "tiiuae/falcon-7b",
        performer_id: str = "tiiuae/falcon-7b-instruct",
        device: str | None = None,
        max_length: int = 512,
        threshold: float = _BINO_DEFAULT_THRESHOLD,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = pick_device(device)
        if self.device == "cpu":
            raise RuntimeError(
                "Binoculars requires a GPU. Use RoBERTaDetector on CPU machines."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(observer_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_id, torch_dtype=dtype
        ).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_id, torch_dtype=dtype
        ).to(self.device).eval()
        self.max_length = max_length
        self.threshold = threshold

    @torch.no_grad()
    def _ce(self, model, input_ids, attention_mask, labels=None):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        target = (labels if labels is not None else input_ids)[:, 1:]
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss = loss.view(target.size())
        mask = attention_mask[:, 1:].to(loss.dtype)
        return (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)

    @torch.no_grad()
    def _bino_score(self, text: str) -> float:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        ppl_obs = self._ce(self.observer, enc.input_ids, enc.attention_mask)
        # Cross-perplexity: observer logits, performer "labels" (token IDs are shared).
        # Following the paper: feed text to performer, use its predicted distribution
        # against the observer's tokens.
        perf_logits = self.performer(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask
        ).logits[:, :-1, :]
        target = enc.input_ids[:, 1:]
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss = ce(perf_logits.reshape(-1, perf_logits.size(-1)), target.reshape(-1))
        loss = loss.view(target.size())
        mask = enc.attention_mask[:, 1:].to(loss.dtype)
        x_ppl = (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        score = (ppl_obs / x_ppl).item()
        return score

    def score(self, text: str) -> float:
        s = self._bino_score(text)
        # Below threshold => AI. Convert gap to probability with a logistic.
        gap = self.threshold - s
        return 1.0 / (1.0 + math.exp(-_BINO_LOGISTIC_SCALE * gap))
