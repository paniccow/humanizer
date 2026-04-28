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
    "RejectionSamplingHumanizer": (".rejection", "RejectionSamplingHumanizer"),
    "RejectionConfig": (".rejection", "RejectionConfig"),
    "Strategy": (".strategies", "Strategy"),
    "ALL_STRATEGIES": (".strategies", "ALL_STRATEGIES"),
    "DEFAULT_REWRITE": (".strategies", "DEFAULT_REWRITE"),
    "TWEET_THREAD": (".strategies", "TWEET_THREAD"),
    "REDDIT_POST": (".strategies", "REDDIT_POST"),
    "SLACK_MESSAGE": (".strategies", "SLACK_MESSAGE"),
    "DIARY_ENTRY": (".strategies", "DIARY_ENTRY"),
    "INTERVIEW": (".strategies", "INTERVIEW"),
}

__all__ = [
    "Humanizer", "HumanizeResult",
    "PromptHumanizer", "PromptHumanizerConfig",
    "AdversarialHumanizer", "AdversarialConfig",
    "TrainedHumanizer", "TrainedHumanizerConfig",
    "RejectionSamplingHumanizer", "RejectionConfig",
    "Strategy", "ALL_STRATEGIES",
    "DEFAULT_REWRITE", "TWEET_THREAD", "REDDIT_POST",
    "SLACK_MESSAGE", "DIARY_ENTRY", "INTERVIEW",
]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
