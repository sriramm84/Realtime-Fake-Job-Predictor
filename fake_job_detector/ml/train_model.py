"""
Trains the Machine Learning models (Logistic Regression & Random Forest) on the preprocessed data.
Evaluates the models and persists the best performing model alongside the TF-IDF vectorizer.
"""

import json
import os
import warnings

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Use relative or absolute import depending on how script is called
try:
    from ml.preprocess import run_preprocessing
except ModuleNotFoundError:
    from preprocess import run_preprocessing

# Ignore sklearn convergence warnings for cleaner output
warnings.filterwarnings("ignore")


def evaluate_model(model_name, y_true, y_pred):
    """
    Computes and prints model evaluation metrics.
    Returns the F1-score to assist with best model selection.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(f"{cm}\n")

    return f1


def build_random_forest():
    """Random Forest with per-tree class weighting to counter the ~5% fraud base rate."""
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def find_best_threshold(y_true, proba):
    """
    Sweeps decision thresholds and returns (threshold, f1) maximising F1.
    Falls back to (0.5, 0.0) when there are no positive labels to tune against.
    """
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return 0.5, 0.0
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (np.asarray(proba) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(round(t, 2)), f1
    return best_t, best_f1


def save_threshold(threshold, path):
    """Persists the tuned decision threshold next to the model artifacts."""
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"threshold": float(threshold)}, fh)


def save_artifacts(best_model, tfidf_vectorizer, model_path, tfidf_path):
    """
    Saves the trained model and the TF-IDF vectorizer to the provided paths.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(best_model, model_path)
    print(f"[Saved] Best Model -> {model_path}")

    joblib.dump(tfidf_vectorizer, tfidf_path)
    print(f"[Saved] TF-IDF vectorizer -> {tfidf_path}")


def main():
    # Define paths relative to project root
    data_path = "data/raw/fake_job_postings.csv"
    model_save_path = "ml/saved_model/fraud_model.pkl"
    tfidf_save_path = "ml/saved_model/tfidf.pkl"

    print("Step 1: Loading and Preprocessing Data...")
    df = run_preprocessing(data_path)

    if df is None:
        print("Pipeline terminated: Unable to load data. Ensure the dataset exists.")
        return

    if "fraudulent" not in df.columns:
        print("Pipeline terminated: Target column 'fraudulent' not found in dataset.")
        return

    # Extract feature inputs (X) and target outputs (y)
    X = df[["combined_text", "has_company_profile", "has_salary_range", "text_length"]]
    y = df["fraudulent"].astype(int)

    print("\nStep 2: Splitting dataset into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nStep 3: Vectorizing text features using TF-IDF (max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=5000)

    X_train_text_tfidf = tfidf.fit_transform(X_train["combined_text"])
    X_test_text_tfidf = tfidf.transform(X_test["combined_text"])

    print("Combining TF-IDF textual features with engineered numerical features...")
    num_features = ["has_company_profile", "has_salary_range", "text_length"]
    X_train_num = X_train[num_features].values
    X_test_num = X_test[num_features].values

    # Use hstack to combine sparse TF-IDF matrix with dense numerical features
    X_train_final = hstack([X_train_text_tfidf, X_train_num])
    X_test_final = hstack([X_test_text_tfidf, X_test_num])

    print("\nStep 4: Training Models...")
    # Train Logistic Regression
    print(" > Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_model.fit(X_train_final, y_train)
    lr_preds = lr_model.predict(X_test_final)

    # Train Random Forest (class-weighted); tune the decision threshold on a held-out
    # validation slice of the training data so the test set is never used for tuning.
    print(" > Training Random Forest...")
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    rf_tuner = build_random_forest().fit(X_fit, y_fit)
    threshold, val_f1 = find_best_threshold(y_val, rf_tuner.predict_proba(X_val)[:, 1])
    print(f" > Tuned threshold={threshold:.2f} (validation F1={val_f1:.4f})")
    rf_model = build_random_forest().fit(X_train_final, y_train)
    rf_preds = (rf_model.predict_proba(X_test_final)[:, 1] >= threshold).astype(int)

    print("\nStep 5: Model Evaluation (primary metric: F1-score)")
    lr_f1 = evaluate_model("Logistic Regression", y_test, lr_preds)
    rf_f1 = evaluate_model("Random Forest (tuned threshold)", y_test, rf_preds)

    # Determine the Champion Model
    if lr_f1 >= rf_f1:
        best_model, best_name, threshold = lr_model, "Logistic Regression", 0.5
    else:
        best_model, best_name = rf_model, "Random Forest"

    print(f"** Best model selected based on F1-score: {best_name} **\n")

    print("Step 6: Saving Artifacts")
    save_artifacts(best_model, tfidf, model_save_path, tfidf_save_path)
    save_threshold(threshold, "ml/saved_model/threshold.json")
    print("\nTraining pipeline successfully completed.")


if __name__ == "__main__":
    main()
