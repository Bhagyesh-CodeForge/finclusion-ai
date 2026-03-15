# data/generate_dataset.py
# Generates a synthetic dataset of informal workers for the FinInclusion AI model.
# Run: python data/generate_dataset.py

import numpy as np
import pandas as pd
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
N = 5000

# ── Occupation profiles ───────────────────────────────────────────────────────
# Each occupation has realistic income ranges and work-day tendencies
# Note - The generated data is not true and is generated to test the working of the model
OCCUPATION_PROFILES = {
    "street_vendor": {
        "income_range":   (200, 900),
        "work_days_mean": 24,
        "work_days_std":  4,
        "tx_mean":        18,
        "tx_std":         5,
        "digital_ratio":  (0.05, 0.35),
        "savings_mean":   0.08,
        "savings_std":    0.05,
        "loan_prob":      0.30,
    },
    "driver": {
        "income_range":   (500, 1600),
        "work_days_mean": 22,
        "work_days_std":  3,
        "tx_mean":        30,
        "tx_std":         8,
        "digital_ratio":  (0.30, 0.80),
        "savings_mean":   0.15,
        "savings_std":    0.07,
        "loan_prob":      0.45,
    },
    "laborer": {
        "income_range":   (200, 700),
        "work_days_mean": 20,
        "work_days_std":  5,
        "tx_mean":        10,
        "tx_std":         4,
        "digital_ratio":  (0.02, 0.20),
        "savings_mean":   0.05,
        "savings_std":    0.04,
        "loan_prob":      0.25,
    },
    "delivery_worker": {
        "income_range":   (600, 1800),
        "work_days_mean": 25,
        "work_days_std":  3,
        "tx_mean":        40,
        "tx_std":         10,
        "digital_ratio":  (0.50, 0.95),
        "savings_mean":   0.18,
        "savings_std":    0.07,
        "loan_prob":      0.50,
    },
    "shopkeeper": {
        "income_range":   (700, 2000),
        "work_days_mean": 26,
        "work_days_std":  2,
        "tx_mean":        50,
        "tx_std":         12,
        "digital_ratio":  (0.25, 0.75),
        "savings_mean":   0.22,
        "savings_std":    0.08,
        "loan_prob":      0.55,
    },
}

OCCUPATIONS = list(OCCUPATION_PROFILES.keys())

# ── Helper: clamp values ──────────────────────────────────────────────────────
def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)

# ── Generate rows ─────────────────────────────────────────────────────────────
rows = []

# Distribute 5000 workers roughly evenly across occupations
occupation_list = np.random.choice(OCCUPATIONS, size=N, p=[0.22, 0.20, 0.20, 0.20, 0.18])

for occ in occupation_list:
    p = OCCUPATION_PROFILES[occ]

    # Income range for this occupation
    lo, hi = p["income_range"]

    # Daily income: normally distributed within the range
    daily_income = np.random.uniform(lo, hi)

    # income_min / income_max: daily_income ± some variance
    variance = daily_income * np.random.uniform(0.10, 0.30)
    income_min = round(max(lo, daily_income - variance), 2)
    income_max = round(min(hi, daily_income + variance), 2)

    # Work days per month (clamped 10–30)
    work_days = int(clamp(
        np.random.normal(p["work_days_mean"], p["work_days_std"]), 10, 30
    ))

    # Monthly transactions
    monthly_tx = int(clamp(
        np.random.normal(p["tx_mean"], p["tx_std"]), 1, 120
    ))

    # Savings ratio (0–0.4)
    savings = clamp(
        np.random.normal(p["savings_mean"], p["savings_std"]), 0.0, 0.40
    )

    # Digital payment ratio
    dlo, dhi = p["digital_ratio"]
    digital = round(np.random.uniform(dlo, dhi), 3)

    # Loan history: binary
    loan = 1 if np.random.random() < p["loan_prob"] else 0

    rows.append({
        "occupation":            occ,
        "daily_income":          round(daily_income, 2),
        "work_days_per_month":   work_days,
        "monthly_transactions":  monthly_tx,
        "savings_ratio":         round(savings, 4),
        "income_min":            income_min,
        "income_max":            income_max,
        "digital_payment_ratio": digital,
        "loan_history":          loan,
    })

df = pd.DataFrame(rows)

# ── Compute financial_stability label ────────────────────────────────────────
# Score combines: income level, work consistency, savings, transactions, digital adoption
# Each component is normalised to [0, 1] then weighted.

df["_income_score"]    = (df["daily_income"] - 200) / (2000 - 200)
df["_work_score"]      = (df["work_days_per_month"] - 10) / (30 - 10)
df["_savings_score"]   = df["savings_ratio"] / 0.40
df["_tx_score"]        = (df["monthly_transactions"] - 1) / 119
df["_digital_score"]   = df["digital_payment_ratio"]
df["_income_consistency"] = 1 - (df["income_max"] - df["income_min"]) / (df["income_max"] + 1e-5)

# Weighted composite stability score
df["_stability_score"] = (
    df["_income_score"]       * 0.30 +
    df["_work_score"]         * 0.20 +
    df["_savings_score"]      * 0.20 +
    df["_tx_score"]           * 0.10 +
    df["_digital_score"]      * 0.10 +
    df["_income_consistency"] * 0.10
)

# Add mild noise so boundaries aren't perfectly sharp
df["_stability_score"] += np.random.normal(0, 0.03, N)
df["_stability_score"]  = clamp(df["_stability_score"], 0, 1)

# Label thresholds
def assign_label(score):
    if score >= 0.60:
        return "stable"
    elif score >= 0.35:
        return "moderate"
    else:
        return "unstable"

df["financial_stability"] = df["_stability_score"].apply(assign_label)

# ── Drop internal scoring columns ─────────────────────────────────────────────
internal_cols = [c for c in df.columns if c.startswith("_")]
df.drop(columns=internal_cols, inplace=True)

# ── Save ──────────────────────────────────────────────────────────────────────
output_dir  = os.path.join(os.path.dirname(__file__), "raw")
output_path = os.path.join(output_dir, "informal_workers.csv")
os.makedirs(output_dir, exist_ok=True)
df.to_csv(output_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n✅  Dataset saved → {output_path}")
print(f"    Rows   : {len(df):,}")
print(f"    Columns: {list(df.columns)}\n")
print("── Label distribution ─────────────────────────────────")
print(df["financial_stability"].value_counts().to_string())
print("\n── Occupation distribution ────────────────────────────")
print(df["occupation"].value_counts().to_string())
print("\n── Numeric summary ────────────────────────────────────")
print(df.describe().round(2).to_string())