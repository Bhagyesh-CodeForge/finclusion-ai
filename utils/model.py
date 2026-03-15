# utils/model.py
# Full ML training pipeline for the FinInclusion AI model.
# Run standalone: python utils/model.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH  = os.path.join(BASE_DIR, "data", "processed", "processed_workers.csv")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "financial_model.pkl")

TARGET_COL      = "financial_stability"
LABEL_NAMES     = ["Unstable", "Moderate", "Stable"]   # maps 0 → 1 → 2


# ── Step 1: Load processed dataset ───────────────────────────────────────────
def load_processed(path: str = PROCESSED_PATH):
    """
    Load processed CSV and split into features (X) and target (y).
    Drops the target column from features automatically.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at: {path}\n"
            "Run  python utils/preprocess.py  first."
        )
    df = pd.read_csv(path)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    print(f"✅  Loaded processed data  →  {X.shape[0]:,} rows | {X.shape[1]} features")
    return X, y


# ── Step 2: Train / Evaluate helpers ─────────────────────────────────────────
def _train_and_evaluate(name, model, X_train, X_test, y_train, y_test) -> dict:
    """Fit a model and return a results dict with accuracy, matrix, report."""
    print(f"\n   Training {name} …")
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    matrix   = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=LABEL_NAMES)

    print(f"   Accuracy : {accuracy * 100:.2f}%")
    print(f"   Confusion Matrix:\n{matrix}")
    print(f"   Classification Report:\n{report}")

    return {
        "name":     name,
        "model":    model,
        "accuracy": accuracy,
        "matrix":   matrix,
        "report":   report,
    }


# ── Step 3: Train both models & auto-select best ──────────────────────────────
def train_and_select(X, y):
    """
    Train RandomForest and GradientBoosting.
    Auto-selects and returns the best model by accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n── Training split: {len(X_train):,} train | {len(X_test):,} test ──")

    candidates = [
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            ),
        ),
    ]

    results = []
    for name, model in candidates:
        result = _train_and_evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    # Auto-select best by accuracy
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n🏆  Best model → {best['name']}  (accuracy: {best['accuracy'] * 100:.2f}%)")
    return best["model"], best["name"], best["accuracy"]


# ── Step 4: Save / Load ───────────────────────────────────────────────────────
def save_model(model, path: str = MODEL_PATH):
    """Persist trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅  Model saved  →  {path}")


def load_model(path: str = MODEL_PATH):
    """Load a previously saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)


# ── Step 5: Credit score & risk category ─────────────────────────────────────
def predict_stability(model, features: np.ndarray) -> dict:
    """
    Run inference for one worker profile and return a human-readable result.

    Parameters:
        model    : trained sklearn model
        features : 2-D numpy array of shape (1, n_features)

    Returns dict with:
        label          – Unstable / Moderate / Stable
        credit_score   – integer between 300 and 850
        risk_category  – High / Medium / Low
        confidence     – model's max class probability (%)
        probabilities  – per-class probabilities dict
    """
    proba     = model.predict_proba(features)[0]   # [p_unstable, p_moderate, p_stable]
    label_idx = int(model.predict(features)[0])
    label     = LABEL_NAMES[label_idx]

    # Credit score: weighted sum of class probabilities mapped to 300–850
    # Higher weight to "stable" probability, penalise "unstable"
    stability_score = (proba[2] * 1.0) + (proba[1] * 0.5) + (proba[0] * 0.0)
    credit_score    = int(300 + stability_score * 550)
    credit_score    = max(300, min(850, credit_score))   # hard clamp

    # Risk category based on credit score bands
    if credit_score >= 700:
        risk_category = "Low"
    elif credit_score >= 500:
        risk_category = "Medium"
    else:
        risk_category = "High"

    return {
        "label":         label,
        "credit_score":  credit_score,
        "risk_category": risk_category,
        "confidence":    round(float(max(proba)) * 100, 1),
        "probabilities": {
            "Unstable": round(float(proba[0]), 4),
            "Moderate": round(float(proba[1]), 4),
            "Stable":   round(float(proba[2]), 4),
        },
    }


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline():
    """End-to-end: load → train → evaluate → save best model."""
    print("\n── FinInclusion ML Training Pipeline ──────────────────────────")

    X, y          = load_processed()
    best_model, best_name, best_acc = train_and_select(X, y)
    save_model(best_model)

    # Quick sanity-check prediction on first row
    sample        = X.iloc[[0]].values
    result        = predict_stability(best_model, sample)
    print(f"\n── Sample prediction (row 0) ───────────────────────────────────")
    for k, v in result.items():
        print(f"   {k:<20}: {v}")
    print("────────────────────────────────────────────────────────────────\n")

    return best_model


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()