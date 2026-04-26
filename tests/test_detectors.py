"""Smoke tests for the detector ensemble shape — no model downloads."""
from humanizer.detectors.base import Detector, DetectorScore
from humanizer.detectors.ensemble import DetectorEnsemble


class _StubDetector(Detector):
    def __init__(self, name: str, p: float):
        self.name = name
        self.p = p

    def score(self, text: str) -> float:  # noqa: ARG002
        return self.p


def test_ensemble_aggregates_mean():
    ens = DetectorEnsemble([_StubDetector("a", 0.2), _StubDetector("b", 0.8)])
    res = ens.score("hello")
    assert abs(res.aggregate - 0.5) < 1e-9
    assert res.by_name == {"a": 0.2, "b": 0.8}


def test_ensemble_reward_is_one_minus_mean_pai():
    ens = DetectorEnsemble([_StubDetector("a", 0.1), _StubDetector("b", 0.3)])
    assert abs(ens.reward("anything") - 0.8) < 1e-9


def test_ensemble_asr_threshold():
    ens = DetectorEnsemble([_StubDetector("a", 0.4), _StubDetector("b", 0.6)])
    asr = ens.asr(["t1", "t2", "t3"])
    assert asr["a"] == 1.0  # 0.4 < 0.5
    assert asr["b"] == 0.0  # 0.6 not < 0.5


def test_weighted_ensemble():
    ens = DetectorEnsemble(
        [_StubDetector("a", 0.0), _StubDetector("b", 1.0)], weights=[3.0, 1.0]
    )
    # weighted mean = (0*3 + 1*1) / 4 = 0.25
    assert abs(ens.score("x").aggregate - 0.25) < 1e-9
