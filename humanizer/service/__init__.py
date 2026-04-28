"""HTTP service layer — wraps the rejection-sampling humanizer behind a
single POST /humanize endpoint. Run via `humanizer serve` or:

    uvicorn humanizer.service.app:app --host 0.0.0.0 --port 8000

Optional deps: `pip install -e '.[serve,openai]'` (FastAPI + uvicorn +
the OpenAI-compatible client).
"""
from __future__ import annotations

# Lazy import — the FastAPI dep is optional. Importing this package
# without `serve` extras shouldn't crash; only attempting to use the
# app object should.

_LAZY = {
    "app": (".app", "app"),
    "build_app": (".app", "build_app"),
    "ServiceConfig": (".app", "ServiceConfig"),
}

__all__ = ["app", "build_app", "ServiceConfig"]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
