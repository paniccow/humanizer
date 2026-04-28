"""Lazy re-exports — torch-heavy modules only load on demand."""
from __future__ import annotations

# Pure-python — safe to import eagerly.
from .burstiness import BurstinessStats, sentence_length_stats, split_sentences
from .facts import FactSet, entity_overlap, extract_facts

_LAZY = {
    "embedding_similarity": (".semantic", "embedding_similarity"),
    "bertscore_f1": (".semantic", "bertscore_f1"),
    "perplexity": (".quality", "perplexity"),
    "length_ratio": (".quality", "length_ratio"),
}

__all__ = [
    "embedding_similarity", "bertscore_f1", "perplexity", "length_ratio",
    "BurstinessStats", "sentence_length_stats", "split_sentences",
    "FactSet", "entity_overlap", "extract_facts",
]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
