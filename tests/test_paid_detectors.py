"""Paid-detector API clients (GPTZero, Originality, Pangram).

These tests stub urllib at the module level — no network calls. Verifies
the response-parsing logic on the documented response shapes plus the
fallback paths (mixed weighting, schema variants, error surfaces).
"""
from __future__ import annotations

import io
import json
from contextlib import contextmanager

import pytest

from humanizer.detectors.gptzero import GPTZeroConfig, GPTZeroDetector
from humanizer.detectors.originality import OriginalityConfig, OriginalityDetector
from humanizer.detectors.pangram import PangramConfig, PangramDetector


@contextmanager
def _stub_urlopen(monkeypatch, module, payload: dict):
    captured: dict = {}

    class _FakeResp:
        def __init__(self, body: bytes):
            self._buf = io.BytesIO(body)

        def read(self, *a, **kw):
            return self._buf.read(*a, **kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8")) if req.data else None
        return _FakeResp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    yield captured


# ---------- GPTZero ----------

def test_gptzero_class_probabilities(monkeypatch):
    from humanizer.detectors import gptzero as gz_mod
    payload = {
        "documents": [{
            "class_probabilities": {"ai": 0.8, "mixed": 0.1, "human": 0.1},
            "predicted_class": "ai",
        }]
    }
    with _stub_urlopen(monkeypatch, gz_mod, payload) as cap:
        d = GPTZeroDetector(api_key="test-key")
        score = d.score("hello world")
    # ai 0.8 + 0.5 * mixed 0.1 = 0.85
    assert score == pytest.approx(0.85)
    # urllib title-cases custom header names; check case-insensitively.
    headers_ci = {k.lower(): v for k, v in cap["headers"].items()}
    assert headers_ci.get("x-api-key") == "test-key"


def test_gptzero_legacy_completely_generated_prob(monkeypatch):
    from humanizer.detectors import gptzero as gz_mod
    payload = {"documents": [{"completely_generated_prob": 0.42}]}
    with _stub_urlopen(monkeypatch, gz_mod, payload):
        d = GPTZeroDetector(api_key="k")
        assert d.score("x") == pytest.approx(0.42)


def test_gptzero_no_api_key_raises(monkeypatch):
    monkeypatch.delenv("GPTZERO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        GPTZeroDetector()


def test_gptzero_empty_documents_raises(monkeypatch):
    from humanizer.detectors import gptzero as gz_mod
    with _stub_urlopen(monkeypatch, gz_mod, {"documents": []}):
        d = GPTZeroDetector(api_key="k")
        with pytest.raises(RuntimeError, match="no documents"):
            d.score("x")


def test_gptzero_mixed_weight_configurable():
    payload = {"documents": [{
        "class_probabilities": {"ai": 0.4, "mixed": 0.6, "human": 0.0}
    }]}
    # Default weight 0.5 -> 0.4 + 0.3 = 0.7
    s_default = GPTZeroDetector._extract_p_ai(payload, mixed_weight=0.5)
    assert s_default == pytest.approx(0.7)
    # Strict weight 1.0 -> mixed counted as full AI -> 1.0
    s_strict = GPTZeroDetector._extract_p_ai(payload, mixed_weight=1.0)
    assert s_strict == pytest.approx(1.0)


# ---------- Originality.ai ----------

def test_originality_score_ai(monkeypatch):
    from humanizer.detectors import originality as o_mod
    payload = {"success": True, "score": {"ai": 0.97, "original": 0.03}, "credits_used": 1}
    with _stub_urlopen(monkeypatch, o_mod, payload) as cap:
        d = OriginalityDetector(api_key="oai-test")
        assert d.score("foo") == pytest.approx(0.97)
    body = cap["body"]
    assert body["content"] == "foo"
    assert body["aiModelVersion"] == "1"


def test_originality_failure_response_raises(monkeypatch):
    from humanizer.detectors import originality as o_mod
    payload = {"success": False, "error": "rate limited"}
    with _stub_urlopen(monkeypatch, o_mod, payload):
        d = OriginalityDetector(api_key="k")
        with pytest.raises(RuntimeError, match="failure"):
            d.score("foo")


def test_originality_flat_ai_score_fallback(monkeypatch):
    from humanizer.detectors import originality as o_mod
    payload = {"success": True, "ai_score": 0.62}
    with _stub_urlopen(monkeypatch, o_mod, payload):
        d = OriginalityDetector(api_key="k")
        assert d.score("x") == pytest.approx(0.62)


def test_originality_no_key_raises(monkeypatch):
    monkeypatch.delenv("ORIGINALITY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        OriginalityDetector()


# ---------- Pangram ----------

def test_pangram_ai_likelihood(monkeypatch):
    from humanizer.detectors import pangram as p_mod
    payload = {"predicted_class": "ai", "ai_likelihood": 0.88}
    with _stub_urlopen(monkeypatch, p_mod, payload) as cap:
        d = PangramDetector(api_key="pgr")
        assert d.score("hi") == pytest.approx(0.88)
    assert cap["body"] == {"text": "hi"}


def test_pangram_class_probabilities(monkeypatch):
    from humanizer.detectors import pangram as p_mod
    payload = {"class_probabilities": {"ai": 0.5, "mixed": 0.4, "human": 0.1}}
    with _stub_urlopen(monkeypatch, p_mod, payload):
        d = PangramDetector(api_key="k")
        # 0.5 + 0.5*0.4 = 0.7
        assert d.score("x") == pytest.approx(0.7)


def test_pangram_predicted_class_human_inverts_confidence(monkeypatch):
    from humanizer.detectors import pangram as p_mod
    payload = {"predicted_class": "human", "confidence": 0.95}
    with _stub_urlopen(monkeypatch, p_mod, payload):
        d = PangramDetector(api_key="k")
        # human with conf 0.95 -> p_ai = 1 - 0.95 = 0.05
        assert d.score("x") == pytest.approx(0.05)


def test_pangram_predicted_class_mixed_yields_half(monkeypatch):
    from humanizer.detectors import pangram as p_mod
    payload = {"predicted_class": "mixed", "confidence": 0.7}
    with _stub_urlopen(monkeypatch, p_mod, payload):
        d = PangramDetector(api_key="k")
        assert d.score("x") == pytest.approx(0.5)


def test_pangram_unknown_schema_raises(monkeypatch):
    from humanizer.detectors import pangram as p_mod
    payload = {"unexpected": "shape"}
    with _stub_urlopen(monkeypatch, p_mod, payload):
        d = PangramDetector(api_key="k")
        with pytest.raises(RuntimeError, match="missing AI-likelihood"):
            d.score("x")


def test_pangram_no_key_raises(monkeypatch):
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        PangramDetector()
