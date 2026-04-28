"""HTTP service tests with FastAPI TestClient + a stubbed rejection
sampler. No real LLM, no real detector — verifies routing, validation,
auth, and the response-shape contract.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from humanizer.humanizers.base import HumanizeResult


class _StubRejection:
    """Drop-in for RejectionSamplingHumanizer.humanize()."""

    def __init__(self, text="OUT", passed=True, score=0.01):
        self._text = text
        self._passed = passed
        self._score = score
        self.config = type("Cfg", (), {
            "max_rounds": 4, "candidates_per_round": 8,
            "p_ai_threshold": 0.05, "similarity_threshold": 0.78,
        })()
        self.last_overrides: dict = {}

    def humanize(self, text, **_):
        # Capture knob overrides applied by the endpoint for test assertions
        self.last_overrides = {
            "max_rounds": self.config.max_rounds,
            "candidates_per_round": self.config.candidates_per_round,
            "p_ai_threshold": self.config.p_ai_threshold,
        }
        return HumanizeResult(
            original=text,
            text=self._text,
            score=self._score,
            attempts=8,
            metadata={
                "passed": self._passed,
                "judge": "stub",
                "judge_calls": 8,
                "rounds_used": 1 if self._passed else 4,
                "best_p_ai": self._score,
            },
        )


def _build_app_with_stub(stub: _StubRejection, monkeypatch=None, env=None):
    if monkeypatch and env:
        for k, v in env.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
    from humanizer.service.app import ServiceConfig, build_app
    api = build_app(ServiceConfig())
    api.state.svc.get_humanizer = lambda: stub
    api.state.svc._judge_resolved_name = "stub"
    return api


def test_humanize_pass(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    stub = _StubRejection(text="rewritten", passed=True, score=0.02)
    api = _build_app_with_stub(stub, monkeypatch, {"HUMANIZER_API_KEY": None})
    client = TestClient(api)
    r = client.post("/humanize", json={"text": "Furthermore, leverage AI."})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "rewritten"
    assert body["passed"] is True
    assert body["score"] == 0.02
    assert body["judge"] == "stub"
    assert body["attempts"] == 8
    assert body["elapsed_ms"] >= 0


def test_humanize_exhausted(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    stub = _StubRejection(text="best_we_got", passed=False, score=0.4)
    api = _build_app_with_stub(stub)
    r = TestClient(api).post("/humanize", json={"text": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is False
    assert body["text"] == "best_we_got"
    assert body["rounds_used"] == 4


def test_humanize_overrides_apply_to_config(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    stub = _StubRejection()
    api = _build_app_with_stub(stub)
    r = TestClient(api).post("/humanize", json={
        "text": "hi", "max_rounds": 2, "candidates": 4, "threshold": 0.1,
    })
    assert r.status_code == 200
    assert stub.last_overrides == {
        "max_rounds": 2, "candidates_per_round": 4, "p_ai_threshold": 0.1,
    }


def test_humanize_rejects_oversized_input(monkeypatch):
    # Env vars are read at build_app() time, so set BEFORE building.
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    monkeypatch.setenv("HUMANIZER_MAX_CHARS", "100")
    from humanizer.service.app import ServiceConfig, build_app
    api = build_app(ServiceConfig())
    api.state.svc.get_humanizer = lambda: _StubRejection()
    r = TestClient(api).post("/humanize", json={"text": "x" * 200})
    assert r.status_code == 413
    assert "max_chars" in r.json()["detail"]


def test_humanize_validates_empty_text(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    stub = _StubRejection()
    api = _build_app_with_stub(stub)
    r = TestClient(api).post("/humanize", json={"text": ""})
    assert r.status_code == 422  # pydantic min_length=1 violation


def test_auth_required_when_env_set(monkeypatch):
    monkeypatch.setenv("HUMANIZER_API_KEY", "secret-token")
    stub = _StubRejection()
    api = _build_app_with_stub(stub)
    client = TestClient(api)

    # No auth header -> 401
    r = client.post("/humanize", json={"text": "hi"})
    assert r.status_code == 401

    # Wrong token -> 401
    r = client.post(
        "/humanize", json={"text": "hi"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401

    # Right token -> 200
    r = client.post(
        "/humanize", json={"text": "hi"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert r.status_code == 200


def test_health_does_not_require_auth(monkeypatch):
    monkeypatch.setenv("HUMANIZER_API_KEY", "secret-token")
    stub = _StubRejection()
    api = _build_app_with_stub(stub)
    r = TestClient(api).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ready", "cold")
    assert "judge" in body
    assert "paid_keys_set" in body


def test_version_endpoint(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    stub = _StubRejection()
    api = _build_app_with_stub(stub)
    r = TestClient(api).get("/version")
    assert r.status_code == 200
    assert r.json()["name"] == "humanizer"


class _StubJudgeWithBreakdown:
    """Has .score() and .last_breakdown (mimics EnsembleJudge contract)."""
    name = "ensemble(orig+pgr)"

    def __init__(self, p_ai: float, breakdown: dict):
        self._p = p_ai
        self.last_breakdown = breakdown

    def score(self, text):
        return self._p


def test_detect_endpoint_returns_per_detector(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    judge = _StubJudgeWithBreakdown(0.42, {"originality": 0.5, "pangram": 0.34})
    stub = _StubRejection()
    stub.judge = judge  # detect endpoint reads humanizer.judge
    api = _build_app_with_stub(stub)
    r = TestClient(api).post("/detect", json={"text": "some text to score"})
    assert r.status_code == 200
    body = r.json()
    assert body["p_ai"] == pytest.approx(0.42)
    assert body["judge"] == "ensemble(orig+pgr)"
    assert body["per_detector"] == {"originality": 0.5, "pangram": 0.34}
    assert body["elapsed_ms"] >= 0


def test_detect_works_with_single_detector_no_breakdown(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    # Single Detector — no last_breakdown attribute
    class _SingleDet:
        name = "single"
        def score(self, text):
            return 0.7
    stub = _StubRejection()
    stub.judge = _SingleDet()
    api = _build_app_with_stub(stub)
    r = TestClient(api).post("/detect", json={"text": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["p_ai"] == pytest.approx(0.7)
    assert body["judge"] == "single"
    assert body["per_detector"] is None


def test_detect_requires_auth(monkeypatch):
    monkeypatch.setenv("HUMANIZER_API_KEY", "tok")
    judge = _StubJudgeWithBreakdown(0.1, {})
    stub = _StubRejection()
    stub.judge = judge
    api = _build_app_with_stub(stub)
    r = TestClient(api).post("/detect", json={"text": "hi"})
    assert r.status_code == 401
    r = TestClient(api).post(
        "/detect", json={"text": "hi"},
        headers={"Authorization": "Bearer tok"},
    )
    assert r.status_code == 200


class _StubBaseHumanizer:
    """Drop-in for PromptHumanizer with a controllable sample()."""
    name = "stub-base"
    def __init__(self, candidates):
        self.candidates = candidates
        self.sample_calls = []
    def humanize(self, text, **_):
        from humanizer.humanizers.base import HumanizeResult
        return HumanizeResult(original=text, text=self.candidates[0])
    def sample(self, text, n, *, temperature=None, top_p=None):
        self.sample_calls.append({"n": n, "temperature": temperature})
        return list(self.candidates[:n])


def test_sample_with_scoring(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    base = _StubBaseHumanizer(["c1", "c2", "c3"])
    judge = _StubJudgeWithBreakdown(0.0, {})

    class _SeqJudge:
        name = "seq-judge"
        last_breakdown = {}
        _scores = iter([0.1, 0.5, 0.9])
        def score(self, text):
            return next(self._scores)

    seq = _SeqJudge()
    stub = _StubRejection()
    stub.base = base
    stub.judge = seq
    api = _build_app_with_stub(stub)
    r = TestClient(api).post("/sample", json={"text": "hi", "n": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["candidates"]) == 3
    assert [c["text"] for c in body["candidates"]] == ["c1", "c2", "c3"]
    assert [c["p_ai"] for c in body["candidates"]] == [0.1, 0.5, 0.9]
    assert body["judge"] == "seq-judge"


def test_sample_no_scoring_skips_judge(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    base = _StubBaseHumanizer(["c1", "c2"])

    class _CountingJudge:
        name = "counting"
        def __init__(self): self.calls = 0
        def score(self, text):
            self.calls += 1
            return 0.0

    j = _CountingJudge()
    stub = _StubRejection()
    stub.base = base
    stub.judge = j
    api = _build_app_with_stub(stub)
    r = TestClient(api).post("/sample", json={"text": "hi", "n": 2, "score": False})
    assert r.status_code == 200
    body = r.json()
    assert all(c["p_ai"] is None for c in body["candidates"])
    assert body["judge"] is None
    assert j.calls == 0  # judge never called


def test_sample_temperature_passed_through(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    base = _StubBaseHumanizer(["a", "b"])

    class _NopJudge:
        name = "nop"
        def score(self, text): return 0.0

    stub = _StubRejection()
    stub.base = base
    stub.judge = _NopJudge()
    api = _build_app_with_stub(stub)
    r = TestClient(api).post(
        "/sample", json={"text": "hi", "n": 2, "temperature": 1.2, "score": False},
    )
    assert r.status_code == 200
    assert base.sample_calls == [{"n": 2, "temperature": 1.2}]


def test_rate_limit_blocks_after_threshold(monkeypatch):
    """3-per-second rate limit returns 429 on the 4th hit, with
    Retry-After header. /health stays open."""
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    monkeypatch.setenv("HUMANIZER_RATE_LIMIT", "3/second")
    monkeypatch.delenv("HUMANIZER_TELEMETRY_PATH", raising=False)

    from humanizer.service.app import ServiceConfig, build_app
    api = build_app(ServiceConfig())
    api.state.svc.get_humanizer = lambda: _StubRejection()
    client = TestClient(api)

    # First 3 succeed
    for _ in range(3):
        r = client.post("/humanize", json={"text": "x"})
        assert r.status_code == 200, r.text
        assert r.headers.get("X-RateLimit-Limit") == "3"

    # 4th is rate-limited
    r = client.post("/humanize", json={"text": "x"})
    assert r.status_code == 429
    assert "retry" in r.json()["detail"].lower()
    assert int(r.headers["Retry-After"]) >= 1
    assert r.headers["X-RateLimit-Remaining"] == "0"

    # /health bypass — not rate limited
    r = client.get("/health")
    assert r.status_code == 200


def test_rate_limit_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    monkeypatch.delenv("HUMANIZER_RATE_LIMIT", raising=False)
    stub = _StubRejection()
    api = _build_app_with_stub(stub)
    client = TestClient(api)
    # 20 calls in a row, all succeed
    for _ in range(20):
        assert client.post("/humanize", json={"text": "x"}).status_code == 200


def test_parse_rate_spec():
    from humanizer.service.ratelimit import parse_rate
    assert parse_rate("60/minute") == (60, 60)
    assert parse_rate("1000/hour") == (1000, 3600)
    assert parse_rate("10/s") == (10, 1)
    with pytest.raises(ValueError):
        parse_rate("bad spec")
    with pytest.raises(ValueError):
        parse_rate("60/decade")


def test_estimate_endpoint(monkeypatch):
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    monkeypatch.setenv("HUMANIZER_LLM_COST", "0.001")
    monkeypatch.setenv("HUMANIZER_JUDGE_COST", "0.003")
    monkeypatch.setenv("HUMANIZER_REJECT_N", "8")
    monkeypatch.setenv("HUMANIZER_REJECT_ROUNDS", "4")

    from humanizer.service.app import ServiceConfig, build_app
    api = build_app(ServiceConfig())
    api.state.svc.get_humanizer = lambda: _StubRejection()
    api.state.svc._judge_resolved_name = "fake-judge"

    # Stub out the judge name lookup (rejection.judge.name)
    class _NamedStub(_StubRejection):
        def __init__(self):
            super().__init__()
            class _J: name = "fake-judge"
            self.judge = _J()

    api.state.svc.get_humanizer = lambda: _NamedStub()
    r = TestClient(api).post(
        "/estimate",
        json={"text": "Sales hit $5,000 in 2024 with 25% growth."},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 8 candidates × ($0.001 + $0.003) = $0.032
    assert body["expected_cost_usd"] == pytest.approx(0.032)
    # 8 × 4 × ($0.001 + $0.003) = $0.128
    assert body["worst_case_cost_usd"] == pytest.approx(0.128)
    assert body["candidates_per_round"] == 8
    assert body["max_rounds"] == 4
    assert body["judge"] == "fake-judge"
    # Detected: $5,000 (currency), 2024 (year), 25% (pct) = 3 facts
    assert body["facts_detected"] >= 3


def test_telemetry_writes_jsonl(monkeypatch, tmp_path):
    """When HUMANIZER_TELEMETRY_PATH is set, every /humanize request
    appends a JSONL record with timing + judge_calls + cost estimate."""
    log_file = tmp_path / "telemetry.jsonl"
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    monkeypatch.setenv("HUMANIZER_TELEMETRY_PATH", str(log_file))
    monkeypatch.setenv("HUMANIZER_LLM_COST", "0.001")
    monkeypatch.setenv("HUMANIZER_JUDGE_COST", "0.003")

    stub = _StubRejection(text="OUT", passed=True, score=0.02)
    api = _build_app_with_stub(stub)
    client = TestClient(api)
    r = client.post("/humanize", json={"text": "test"})
    assert r.status_code == 200

    assert log_file.exists()
    lines = [json.loads(l) for l in log_file.read_text().splitlines() if l]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["path"] == "/humanize"
    assert rec["status"] == 200
    assert rec["attempts"] == 8
    assert rec["judge_calls"] == 8
    assert rec["passed"] is True
    assert rec["score"] == 0.02
    # 8 attempts × $0.001 + 8 judge_calls × $0.003 = $0.032
    assert rec["cost_estimate_usd"] == pytest.approx(0.032, rel=0.01)
    assert rec["elapsed_ms"] >= 0


def test_telemetry_disabled_by_default(monkeypatch, tmp_path):
    """No HUMANIZER_TELEMETRY_PATH => no log file, no overhead."""
    monkeypatch.delenv("HUMANIZER_TELEMETRY_PATH", raising=False)
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    stub = _StubRejection()
    api = _build_app_with_stub(stub)
    r = TestClient(api).post("/humanize", json={"text": "x"})
    assert r.status_code == 200
    log_file = tmp_path / "telemetry.jsonl"
    assert not log_file.exists()


def test_telemetry_logs_errors_too(monkeypatch, tmp_path):
    """4xx responses are logged with error field set."""
    log_file = tmp_path / "telemetry.jsonl"
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)
    monkeypatch.setenv("HUMANIZER_TELEMETRY_PATH", str(log_file))
    monkeypatch.setenv("HUMANIZER_MAX_CHARS", "10")

    from humanizer.service.app import ServiceConfig, build_app
    api = build_app(ServiceConfig())
    api.state.svc.get_humanizer = lambda: _StubRejection()
    r = TestClient(api).post("/humanize", json={"text": "way too long input"})
    assert r.status_code == 413

    rec = json.loads(log_file.read_text().splitlines()[0])
    assert rec["status"] == 413
    assert rec["error"] == "http_413"


def test_humanize_response_includes_per_detector_when_set(monkeypatch):
    """When the rejection sampler's metadata includes per_detector
    (because the judge is EnsembleJudge), the /humanize response
    surfaces it."""
    monkeypatch.delenv("HUMANIZER_API_KEY", raising=False)

    # Override stub to include per_detector in metadata
    class _Stub2(_StubRejection):
        def humanize(self, text, **_):
            from humanizer.humanizers.base import HumanizeResult
            return HumanizeResult(
                original=text, text="OUT", score=0.02, attempts=8,
                metadata={
                    "passed": True, "judge": "ens", "judge_calls": 8,
                    "rounds_used": 1, "best_p_ai": 0.02,
                    "per_detector": {"originality": 0.01, "pangram": 0.03},
                },
            )

    api = _build_app_with_stub(_Stub2())
    r = TestClient(api).post("/humanize", json={"text": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["per_detector"] == {"originality": 0.01, "pangram": 0.03}
