import json

import numpy as np
import pytest
from spark import model_loader
from spark.model_loader import apply_threshold, load_threshold


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, p_fraud):
        self.p = p_fraud

    def predict_proba(self, _):
        return np.array([[1 - self.p, self.p]])


def test_below_threshold_is_legit():
    assert apply_threshold(FakeModel(0.3), None, 0.4) == (0, pytest.approx(0.7))


def test_at_or_above_threshold_is_fraud_with_fraud_probability():
    assert apply_threshold(FakeModel(0.4), None, 0.4) == (1, pytest.approx(0.4))


def test_load_threshold_reads_file(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"threshold": 0.33}))
    assert load_threshold(p) == 0.33


def test_load_threshold_defaults_when_missing_or_corrupt(tmp_path):
    assert load_threshold(tmp_path / "nope.json") == 0.5
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    assert load_threshold(bad) == 0.5


def test_predict_job_uses_threshold(monkeypatch):
    monkeypatch.setattr(model_loader, "load_model_and_vectorizer", lambda: (FakeModel(0.3), None))
    monkeypatch.setattr(model_loader, "build_feature_vector", lambda t, m, v: None)
    monkeypatch.setattr(model_loader, "load_threshold", lambda *a, **k: 0.25)
    assert model_loader.predict_job("some job text")[0] == 1
