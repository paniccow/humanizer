"""Quality metrics — fluency (perplexity) and surface stats."""
from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..detectors.base import pick_device


@lru_cache(maxsize=2)
def _load_lm(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    return tok, model


@torch.no_grad()
def perplexity(
    text: str,
    model_id: str = "gpt2",
    device: str | None = None,
    stride: int = 512,
) -> float:
    """Causal-LM perplexity. Lower = more fluent / more typical for the LM.

    AI text typically has *low* perplexity under GPT-2-family LMs; humanization
    should push it up moderately without making it gibberish.
    """
    device = pick_device(device)
    tok, model = _load_lm(model_id, device)
    enc = tok(text, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    max_len = model.config.max_position_embeddings
    seq_len = input_ids.size(1)

    nlls = []
    n_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        target = ids.clone()
        target[:, :-trg_len] = -100
        out = model(ids, labels=target)
        # out.loss is mean over the trg_len tokens — multiply back to NLL sum.
        nlls.append(out.loss.float() * trg_len)
        n_tokens += trg_len
        prev_end = end
        if end == seq_len:
            break
    if n_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.stack(nlls).sum() / n_tokens))


def length_ratio(candidate: str, reference: str) -> float:
    """Word-count ratio. ~1.0 means humanization preserved length."""
    c = max(len(candidate.split()), 1)
    r = max(len(reference.split()), 1)
    return c / r
