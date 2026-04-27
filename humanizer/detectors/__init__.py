"""Detector subpackage. Pure-python interfaces eager; model-loading detectors lazy."""
from __future__ import annotations

from .base import Detector, DetectorScore, pick_device
from .ensemble import DetectorEnsemble, EnsembleResult

_LAZY = {
    "RoBERTaDetector": (".roberta", "RoBERTaDetector"),
    "Binoculars": (".binoculars", "Binoculars"),
    "default_ensemble": (".factory", "default_ensemble"),
    "holdout_ensemble": (".holdout", "holdout_ensemble"),
    "HOLDOUT_DETECTOR_IDS": (".holdout", "HOLDOUT_DETECTOR_IDS"),
    "GPTZeroDetector": (".gptzero", "GPTZeroDetector"),
    "GPTZeroConfig": (".gptzero", "GPTZeroConfig"),
    "OriginalityDetector": (".originality", "OriginalityDetector"),
    "OriginalityConfig": (".originality", "OriginalityConfig"),
    "PangramDetector": (".pangram", "PangramDetector"),
    "PangramConfig": (".pangram", "PangramConfig"),
    "EnsembleJudge": (".judge", "EnsembleJudge"),
    "judge_from_env": (".judge", "judge_from_env"),
    "available_paid_detectors": (".judge", "available_paid_detectors"),
}

__all__ = [
    "Detector", "DetectorScore", "DetectorEnsemble", "EnsembleResult",
    "RoBERTaDetector", "Binoculars", "pick_device",
    "default_ensemble", "holdout_ensemble", "HOLDOUT_DETECTOR_IDS",
    "GPTZeroDetector", "GPTZeroConfig",
    "OriginalityDetector", "OriginalityConfig",
    "PangramDetector", "PangramConfig",
    "EnsembleJudge", "judge_from_env", "available_paid_detectors",
]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
