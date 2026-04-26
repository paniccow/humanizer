"""Semantic similarity metrics — guardrail to ensure humanization doesn't change meaning."""
from __future__ import annotations

from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer, util


@lru_cache(maxsize=2)
def _load(model_id: str, device: str | None) -> SentenceTransformer:
    return SentenceTransformer(model_id, device=device)


def embedding_similarity(
    a: str | list[str],
    b: str | list[str],
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
) -> torch.Tensor:
    """Cosine similarity between sentence embeddings of `a` and `b`. Element-wise.

    Returns a tensor of shape (len(a),) when both are lists, else a 1-element tensor.
    """
    model = _load(model_id, device)
    a_list = [a] if isinstance(a, str) else a
    b_list = [b] if isinstance(b, str) else b
    if len(a_list) != len(b_list):
        raise ValueError("a and b must have the same length")
    ea = model.encode(a_list, convert_to_tensor=True, normalize_embeddings=True)
    eb = model.encode(b_list, convert_to_tensor=True, normalize_embeddings=True)
    return util.pairwise_cos_sim(ea, eb)


def bertscore_f1(
    candidates: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str | None = None,
) -> list[float]:
    """BERTScore F1 — token-level semantic similarity. Heavier than embedding sim.

    Use `microsoft/deberta-xlarge-mnli` for best correlation with human judgment;
    fall back to `roberta-large` on memory-constrained setups.
    """
    from bert_score import score as bs

    if len(candidates) != len(references):
        raise ValueError("candidates and references must have the same length")
    _, _, f1 = bs(
        candidates, references, model_type=model_type, device=device, verbose=False
    )
    return f1.cpu().tolist()
