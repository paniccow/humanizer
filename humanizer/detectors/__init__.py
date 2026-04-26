"""Detector subpackage. Pure-python interfaces eager; model-loading detectors lazy."""
from __future__ import annotations

from .base import Detector, DetectorScore, pick_device
from .ensemble import DetectorEnsemble, EnsembleResult

_LAZY = {
    "RoBERTaDetector": (".roberta", "RoBERTaDetector"),
    "Binoculars": (".binoculars", "Binoculars"),
    "default_ensemble": (".factory", "default_ensemble"),
}

__all__ = [
    "Detector", "DetectorScore", "DetectorEnsemble", "EnsembleResult",
    "RoBERTaDetector", "Binoculars", "pick_device", "default_ensemble",
]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
