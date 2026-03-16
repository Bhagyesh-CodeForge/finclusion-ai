# utils/preprocess.py
# Full preprocessing pipeline for the FinInclusion AI model.
# Run standalone: python utils/preprocess.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH       = os.path.join(BASE_DIR, "data", "raw",       "informal_workers.csv")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "processed_workers.csv")


# ── Step 1: Load ──────────────────────────────────────────────────────────────
def load_data(filepath: str = RAW_PATH) -> pd.DataFrame:
    """Load raw CSV dataset from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}\n"
                                "Run  python data/generate_dataset.py  first.")
    df = pd.read_csv(filepath)
    print(f"Loaded  {len(df):,} rows  |  {df.shape[1]} columns")
    return df


# ── Step 2: Clean ─────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and fill missing numeric values with column medians."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"   Dropped {before - after} duplicate rows")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing = df[numeric_cols].isnull().sum().sum()
    if missing:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print(f"   Filled {missing} missing numeric values with column medians")

    print(f"Cleaned  →  {len(df):,} rows remaining")
    return df


# ── Step 3: Feature Engineering ───────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create two new derived features:
      - income_consistency  : income_min / income_max  (0–1, higher = more consistent)
      - monthly_income      : daily_income * work_days_per_month
    """
    df = df.copy()

    # Income consistency — how stable the worker's income range is
    df["income_consistency"] = df["income_min"] / (df["income_max"] + 1e-9)

    # Monthly income — total estimated monthly earning
    df["monthly_income"] = df["daily_income"] * df["work_days_per_month"]

    print("Engineered features: income_consistency, monthly_income")
    return df


# ── Step 4: Encode Categorical Columns ────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 'occupation' column using LabelEncoder.
    Mapping is printed for transparency.
    """
    df = df.copy()
    le = LabelEncoder()
    df["occupation"] = le.fit_transform(df["occupation"])

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Encoded 'occupation'  →  {mapping}")
    return df


# ── Step 5: Encode Target Label ───────────────────────────────────────────────
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map financial_stability text labels to integers:
      unstable → 0 | moderate → 1 | stable → 2
    """
    label_map = {"unstable": 0, "moderate": 1, "stable": 2}
    df["financial_stability"] = df["financial_stability"].map(label_map)
    print(f"Encoded target  →  {label_map}")
    return df


# ── Step 6: Normalize Numerical Features ─────────────────────────────────────
FEATURE_COLS = [
    "daily_income",
    "work_days_per_month",
    "monthly_transactions",
    "savings_ratio",
    "income_min",
    "income_max",
    "digital_payment_ratio",
    "income_consistency",
    "monthly_income",
]

def normalize_features(df: pd.DataFrame, feature_cols: list = FEATURE_COLS):
    """Apply Min-Max scaling (0–1) to all numerical feature columns."""
    scaler = MinMaxScaler()
    df = df.copy()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"Normalized {len(feature_cols)} numerical features with MinMaxScaler")
    return df, scaler


# ── Step 7: Save ──────────────────────────────────────────────────────────────
def save_processed(df: pd.DataFrame, path: str = PROCESSED_PATH):
    """Save the processed DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed dataset  →  {path}")


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    raw_path:       str = RAW_PATH,
    processed_path: str = PROCESSED_PATH,
):
    """
    Execute all preprocessing steps end-to-end.
    Returns processed DataFrame and fitted scaler.
    """
    print("\n── FinInclusion Preprocessing Pipeline ────────────────────────")

    df = load_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df = encode_target(df)
    df, scaler = normalize_features(df, FEATURE_COLS)
    save_processed(df, processed_path)

    print("\n── Final dataset preview ───────────────────────────────────────")
    print(df.head(3).to_string())
    print(f"\n── Shape: {df.shape}  |  Columns: {list(df.columns)}")
    print("────────────────────────────────────────────────────────────────\n")

    return df, scaler


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
