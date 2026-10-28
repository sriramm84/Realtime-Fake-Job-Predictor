"""
Reproducible evaluation of the saved model on the same stratified 80/20 split used in training.
Reports classification metrics at the tuned threshold and single-process inference speed,
and writes them to docs/metrics.json.
"""

import json
import time
from pathlib import Path

import joblib
from ml.preprocess import run_preprocessing
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from spark.model_loader import apply_threshold, build_feature_vector, load_threshold

ROOT = Path(__file__).resolve().parents[1]
NUM_FEATURES = ["has_company_profile", "has_salary_range", "text_length"]


def evaluate():
    df = run_preprocessing(str(ROOT / "data/raw/fake_job_postings.csv"))
    X, y = df[["combined_text"] + NUM_FEATURES], df["fraudulent"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = joblib.load(ROOT / "ml/saved_model/fraud_model.pkl")
    tfidf = joblib.load(ROOT / "ml/saved_model/tfidf.pkl")
    threshold = load_threshold(ROOT / "ml/saved_model/threshold.json")

    features = hstack([tfidf.transform(X_test["combined_text"]), X_test[NUM_FEATURES].values])
    preds = (model.predict_proba(features)[:, 1] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # Single-process, single-record inference (the path each Spark UDF call takes).
    sample = X_test["combined_text"].head(500).tolist()
    start = time.perf_counter()
    for text in sample:
        apply_threshold(model, build_feature_vector(text, model, tfidf), threshold)
    per_sec = len(sample) / (time.perf_counter() - start)

    return {
        "dataset_rows": int(len(df)),
        "fraud_rate": round(float(y.mean()), 4),
        "test_rows": int(len(y_test)),
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "single_process_inference_per_sec": round(per_sec, 1),
    }


if __name__ == "__main__":
    metrics = evaluate()
    out = ROOT.parent / "docs" / "metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
