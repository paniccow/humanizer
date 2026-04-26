"""humanizer — train an LLM to humanize AI-generated text against AI detectors.

Public API:
    from humanizer import (
        # detectors
        default_ensemble, DetectorEnsemble, RoBERTaDetector,
        # humanizers
        PromptHumanizer, AdversarialHumanizer, TrainedHumanizer,
        # eval
        evaluate,
    )

Symbols are imported lazily — `from humanizer.postprocess import X` does NOT
pull in torch/transformers/etc. Pay only for what you use.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Map exported name -> (submodule path relative to this package, attribute name)
_LAZY: dict[str, tuple[str, str]] = {
    # detectors
    "Detector": (".detectors", "Detector"),
    "DetectorEnsemble": (".detectors", "DetectorEnsemble"),
    "DetectorScore": (".detectors", "DetectorScore"),
    "EnsembleResult": (".detectors", "EnsembleResult"),
    "RoBERTaDetector": (".detectors", "RoBERTaDetector"),
    "default_ensemble": (".detectors", "default_ensemble"),
    # humanizers
    "Humanizer": (".humanizers", "Humanizer"),
    "HumanizeResult": (".humanizers", "HumanizeResult"),
    "PromptHumanizer": (".humanizers", "PromptHumanizer"),
    "PromptHumanizerConfig": (".humanizers", "PromptHumanizerConfig"),
    "AdversarialHumanizer": (".humanizers", "AdversarialHumanizer"),
    "AdversarialConfig": (".humanizers", "AdversarialConfig"),
    "TrainedHumanizer": (".humanizers", "TrainedHumanizer"),
    "TrainedHumanizerConfig": (".humanizers", "TrainedHumanizerConfig"),
    # postprocess
    "apply_burstiness": (".postprocess", "apply_burstiness"),
    "BurstinessConfig": (".postprocess", "BurstinessConfig"),
    # patterns
    "Fingerprint": (".patterns", "Fingerprint"),
    "analyze_patterns": (".patterns", "analyze"),
    # eval
    "evaluate": (".eval", "evaluate"),
    "EvalReport": (".eval", "EvalReport"),
}

__all__ = list(_LAZY) + ["__version__"]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        mod_path, attr = _LAZY[name]
        mod = import_module(mod_path, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
