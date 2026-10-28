import json

import numpy as np
from ml.train_model import build_random_forest, find_best_threshold, save_threshold


def test_random_forest_uses_balanced_class_weight():
    assert build_random_forest().class_weight == "balanced_subsample"


def test_best_threshold_maximises_f1():
    y = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.35, 0.45, 0.4, 0.9])
    threshold, f1 = find_best_threshold(y, proba)
    preds = (proba >= threshold).astype(int)
    from sklearn.metrics import f1_score

    assert f1 == f1_score(y, preds)
    assert all(f1 >= f1_score(y, (proba >= t).astype(int)) for t in np.linspace(0.05, 0.95, 19))


def test_best_threshold_is_not_hardcoded_half():
    y = np.array([0] * 8 + [1] * 2)
    proba = np.array([0.01] * 8 + [0.3, 0.35])
    threshold, f1 = find_best_threshold(y, proba)
    assert threshold < 0.5
    assert f1 == 1.0


def test_best_threshold_handles_no_positives():
    threshold, f1 = find_best_threshold(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]))
    assert threshold == 0.5
    assert f1 == 0.0


def test_save_threshold_writes_json(tmp_path):
    path = tmp_path / "threshold.json"
    save_threshold(0.37, path)
    assert json.loads(path.read_text())["threshold"] == 0.37
