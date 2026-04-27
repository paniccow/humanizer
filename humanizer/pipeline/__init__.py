"""Multi-stage humanization pipeline.

Pure-python `scrub` is eager (no deps). Full `Pipeline` (which needs torch
through humanizers/metrics) is lazy.
"""
from __future__ import annotations

from .scrub import ScrubConfig, ScrubResult, scrub

_LAZY = {
    "Pipeline": (".pipeline", "Pipeline"),
    "PipelineConfig": (".pipeline", "PipelineConfig"),
    "PipelineResult": (".pipeline", "PipelineResult"),
}

__all__ = [
    "scrub", "ScrubConfig", "ScrubResult",
    "Pipeline", "PipelineConfig", "PipelineResult",
]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
