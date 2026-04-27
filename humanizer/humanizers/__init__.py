"""Humanizers — pure-python base/result eager; model-loading classes lazy."""
from __future__ import annotations

from .base import HumanizeResult, Humanizer

_LAZY = {
    "PromptHumanizer": (".prompt", "PromptHumanizer"),
    "PromptHumanizerConfig": (".prompt", "PromptHumanizerConfig"),
    "AdversarialHumanizer": (".adversarial", "AdversarialHumanizer"),
    "AdversarialConfig": (".adversarial", "AdversarialConfig"),
    "TrainedHumanizer": (".trained", "TrainedHumanizer"),
    "TrainedHumanizerConfig": (".trained", "TrainedHumanizerConfig"),
}

__all__ = [
    "Humanizer", "HumanizeResult",
    "PromptHumanizer", "PromptHumanizerConfig",
    "AdversarialHumanizer", "AdversarialConfig",
    "TrainedHumanizer", "TrainedHumanizerConfig",
]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
