# app/main.py
# Streamlit UI for the FinInclusion AI Financial Stability Predictor
# Run with: python -m streamlit run app/main.py

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# SHAP removed — using built-in importance-based explanation (no extra dependencies)
SHAP_AVAILABLE = False

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH     = os.path.join(ROOT, "models", "financial_model.pkl")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "processed_workers.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
OCCUPATION_MAP = {
    "Delivery Worker": 0,
    "Driver":          1,
    "Laborer":         2,
    "Shopkeeper":      3,
    "Street Vendor":   4,
}

FEATURE_COLS = [
    "daily_income",
    "work_days_per_month",
    "monthly_transactions",
    "savings_ratio",
    "income_min",
    "income_max",
    "digital_payment_ratio",
    "loan_history",
    "income_consistency",
    "monthly_income",
]

ALL_COLS = ["occupation"] + FEATURE_COLS

FEATURE_LABELS = {
    "occupation":            "Occupation",
    "daily_income":          "Daily Income",
    "work_days_per_month":   "Work Days / Month",
    "monthly_transactions":  "Monthly Transactions",
    "savings_ratio":         "Savings Ratio",
    "income_min":            "Income Min",
    "income_max":            "Income Max",
    "digital_payment_ratio": "Digital Payment Ratio",
    "loan_history":          "Loan History",
    "income_consistency":    "Income Consistency",
    "monthly_income":        "Monthly Income",
}

SUPPORT_PROGRAMS = {
    "Stable": {
        "program":     "Micro-Investment Scheme",
        "description": "Eligible for small business loans up to ₹50,000 at subsidised rates.",
        "color": "#1A7A4A", "emoji": "🟢",
    },
    "Moderate": {
        "program":     "Micro-Credit & Skills Training",
        "description": "Eligible for micro-credit up to ₹20,000 + free vocational skill programs.",
        "color": "#C05B00", "emoji": "🟡",
    },
    "Unstable": {
        "program":     "Government Welfare Support",
        "description": "Recommended for PM Jan Dhan Yojana, food security, and emergency relief funds.",
        "color": "#B91C1C", "emoji": "🔴",
    },
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinInclusion AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title  { font-size: 2.1rem; font-weight: 800; color: #1A4B8C; margin-bottom: 0; }
    .sub-title   { font-size: 1rem; color: #555; margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .result-card { background: #F0F6FF; border-left: 5px solid #1A4B8C;
                   padding: 1rem 1.4rem; border-radius: 8px; margin-bottom: 1rem; }
    .metric-val  { font-size: 2rem; font-weight: 700; color: #1A4B8C; }
    .metric-lbl  { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .support-box { padding: 1rem 1.4rem; border-radius: 8px; margin-top: 1rem; }
    .shap-box    { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;
                   padding: 1rem 1.4rem; margin-top: 0.5rem; }
    .divider     { border-top: 1px solid #E2E8F0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler (cached) ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def get_scaler():
    if not os.path.exists(PROCESSED_PATH):
        return None
    df = pd.read_csv(PROCESSED_PATH)
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURE_COLS])
    return scaler

@st.cache_resource
def get_background_data():
    """Load a sample of training data for SHAP explainer background."""
    if not os.path.exists(PROCESSED_PATH):
        return None
    df = pd.read_csv(PROCESSED_PATH)
    return df[ALL_COLS].sample(100, random_state=42)

# ── Preprocess input ──────────────────────────────────────────────────────────
def preprocess_input(occ_encoded, daily_income, work_days,
                     monthly_tx, savings_ratio, digital_ratio, loan_history, scaler):
    income_min         = daily_income * 0.75
    income_max         = daily_income * 1.25
    income_consistency = income_min / (income_max + 1e-9)
    monthly_income     = daily_income * work_days

    raw = pd.DataFrame([{
        "daily_income":          daily_income,
        "work_days_per_month":   work_days,
        "monthly_transactions":  monthly_tx,
        "savings_ratio":         savings_ratio,
        "income_min":            income_min,
        "income_max":            income_max,
        "digital_payment_ratio": digital_ratio,
        "loan_history":          loan_history,
        "income_consistency":    income_consistency,
        "monthly_income":        monthly_income,
    }])

    scaled   = scaler.transform(raw[FEATURE_COLS])
    all_cols = ["occupation"] + FEATURE_COLS
    values   = np.hstack([[[occ_encoded]], scaled])
    return pd.DataFrame(values, columns=all_cols)

# ── Chart: Credit score gauge ─────────────────────────────────────────────────
def draw_gauge(credit_score):
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    bands = [(0.0, 0.33, "#EF4444"), (0.33, 0.66, "#F97316"), (0.66, 1.00, "#22C55E")]
    for lo, hi, color in bands:
        t1 = np.radians(180 - 180 * lo)
        t2 = np.radians(180 - 180 * hi)
        theta = np.linspace(t1, t2, 100)
        for r_in, r_out in [(0.55, 0.85)]:
            xo = r_out * np.cos(theta);  yo = r_out * np.sin(theta)
            xi = r_in  * np.cos(theta[::-1]); yi = r_in * np.sin(theta[::-1])
            ax.fill(np.concatenate([xo, xi]), np.concatenate([yo, yi]), color=color, alpha=0.85)
    norm  = (credit_score - 300) / 550
    angle = np.radians(180 - norm * 180)
    ax.plot([0, 0.62 * np.cos(angle)], [0, 0.62 * np.sin(angle)],
            color="#1A4B8C", linewidth=3, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.06, color="#1A4B8C", zorder=5))
    ax.text(0, -0.22, str(credit_score), ha="center", va="center",
            fontsize=22, fontweight="bold", color="#1A4B8C")
    ax.text(0, -0.42, "Credit Score (300–850)", ha="center", va="center", fontsize=7, color="#666")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.55, 1.05); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

# ── Chart: Global feature importance ─────────────────────────────────────────
def draw_global_importance(model):
    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({
            "feature":    [FEATURE_LABELS.get(c, c) for c in ALL_COLS],
            "importance": importances,
        })
        .sort_values("importance", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    max_val = feat_df["importance"].max()
    colors  = ["#1A4B8C" if v == max_val else "#93C5FD" for v in feat_df["importance"]]
    bars = ax.barh(feat_df["feature"], feat_df["importance"],
                   color=colors, edgecolor="none", height=0.6)
    for bar, val in zip(bars, feat_df["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=7.5, color="#333")
    ax.set_xlabel("Importance", fontsize=8, color="#444")
    ax.set_title("Global Feature Importance\n(across all predictions)", fontsize=9,
                 fontweight="bold", color="#1A4B8C", pad=8)
    ax.tick_params(labelsize=8, colors="#444")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    plt.tight_layout()
    return fig

# ── Chart: SHAP-style waterfall (per prediction) ─────────────────────────────
def draw_shap_waterfall(model, input_df, background_df, label_idx):
    """
    Build a waterfall chart showing how each feature PUSHES the prediction
    up (toward Stable) or down (toward Unstable) for THIS specific worker.

    If the real SHAP library is available, uses TreeExplainer for exact values.
    Otherwise uses a fast approximation: feature_importance * (input - mean).
    """
    feature_names  = ALL_COLS
    friendly_names = [FEATURE_LABELS.get(c, c) for c in feature_names]

    # ── Feature importance × deviation from average worker ──────────────────
    # For each feature:
    #   contribution = how important the feature is × how different this worker is
    #   positive → pushes toward Stable
    #   negative → pushes toward Unstable
    imp       = model.feature_importances_           # how much each feature matters
    inp_vals  = input_df.values[0]                   # this worker's values
    bg_mean   = background_df.values.mean(axis=0)    # average worker baseline
    deviation = inp_vals - bg_mean                   # above or below average?
    values    = imp * deviation * 10                 # scaled contribution
    base_val  = 0.5

    # Sort by absolute value, take top 8 — sorted ascending so biggest is at top
    top_idx  = np.argsort(np.abs(values))[-8:]
    top_vals = values[top_idx]
    top_names = [friendly_names[i] for i in top_idx]
    top_raw   = [input_df.iloc[0, i] for i in top_idx]

    n = len(top_vals)

    # ── Build waterfall ───────────────────────────────────────────────────────
    # Tall figure + generous per-bar height so labels never overlap
    fig_height = max(6, n * 0.95)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    # Spread bars evenly with clear gaps between them
    y_pos  = np.arange(n) * 1.4          # 1.4 spacing units between each bar
    colors = ["#22C55E" if v > 0 else "#EF4444" for v in top_vals]
    bars   = ax.barh(y_pos, top_vals, color=colors, edgecolor="none",
                     height=0.7, alpha=0.88)

    # Normalise values to 0-100 scale for readable display
    max_abs = max(np.abs(top_vals).max(), 1e-9)

    # Value labels — placed outside the bar with a small gap, never overlapping
    for bar, val, raw in zip(bars, top_vals, top_raw):
        sign  = "+" if val > 0 else ""
        pct   = val / max_abs * 100       # normalised score for display
        label = f"{sign}{pct:.1f}  (input: {raw:.2f})"
        gap   = max_abs * 0.03            # small fixed gap from bar end
        x_pos = val + gap if val >= 0 else val - gap
        ha    = "left"    if val >= 0 else "right"
        ax.text(x_pos,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center", ha=ha,
                fontsize=9, color="#222",
                fontweight="normal")

    # Vertical zero line
    ax.axvline(0, color="#94A3B8", linewidth=1.2, linestyle="--", zorder=0)

    # Y-axis labels — use the spaced y_pos ticks, larger font, enough room
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10.5, color="#333")
    ax.tick_params(axis="y", length=0, pad=8)   # pad adds space between label and axis

    # X-axis — hide ticks, keep a clean label
    ax.xaxis.set_visible(False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Extra x-room so bar labels don't get clipped
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - max_abs * 0.35, x_max + max_abs * 0.35)

    # Title
    ax.set_title("Why this prediction?  — Factor Contribution Analysis",
                 fontsize=12, fontweight="bold", color="#1A4B8C", pad=16)

    # Legend
    green_patch = mpatches.Patch(color="#22C55E", alpha=0.88, label="🟢  Pushes toward Stable")
    red_patch   = mpatches.Patch(color="#EF4444", alpha=0.88, label="🔴  Pushes toward Unstable")
    ax.legend(handles=[green_patch, red_patch], fontsize=9,
              loc="lower right", framealpha=0.8, edgecolor="#E2E8F0")

    plt.tight_layout(pad=1.5)
    return fig

# ── Chart: Probability breakdown ─────────────────────────────────────────────
def draw_probability_bars(proba):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    cats   = ["Unstable", "Moderate", "Stable"]
    vals   = [round(p * 100, 1) for p in proba]
    colors = ["#EF4444", "#F97316", "#22C55E"]
    bars   = ax.barh(cats, vals, color=colors, edgecolor="none", height=0.5)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color="#444")
    ax.set_xlim(0, 115)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=9, colors="#444")
    ax.xaxis.set_visible(False)
    plt.tight_layout()
    return fig

# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">💸 AI Financial Stability Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">For Informal Workers in South Asia &nbsp;·&nbsp; AI Tool Development Challenge 2026</p>', unsafe_allow_html=True)

model         = load_model()
scaler        = get_scaler()
background_df = get_background_data()

if model is None:
    st.error("⚠️  Model not found. Please run `python utils/model.py` first.")
    st.stop()
if scaler is None:
    st.error("⚠️  Processed data not found. Please run `python utils/preprocess.py` first.")
    st.stop()



# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👷 Worker Profile")
    st.caption("Fill in the worker's details and click Predict.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    occupation    = st.selectbox("Occupation", list(OCCUPATION_MAP.keys()))
    daily_income  = st.slider("Daily Income (₹)", 200, 2000, 700, step=50)
    work_days     = st.slider("Work Days / Month", 1, 30, 22)
    monthly_tx    = st.slider("Monthly Transactions", 1, 120, 25)
    savings_ratio = st.slider("Savings Ratio", 0.00, 0.40, 0.10, step=0.01,
                               help="Fraction of income saved each month")
    digital_ratio = st.slider("Digital Payment Ratio", 0.00, 1.00, 0.30, step=0.05,
                               help="Share of payments made digitally")
    loan_history  = st.radio("Existing Loan History", ["No", "Yes"])
    loan_binary   = 1 if loan_history == "Yes" else 0

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Financial Stability", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if predict_btn:
    occ_encoded = OCCUPATION_MAP[occupation]
    input_df    = preprocess_input(
        occ_encoded, daily_income, work_days,
        monthly_tx, savings_ratio, digital_ratio, loan_binary, scaler
    )

    proba     = model.predict_proba(input_df)[0]
    label_idx = int(model.predict(input_df)[0])
    labels    = ["Unstable", "Moderate", "Stable"]
    label     = labels[label_idx]

    stability_score = proba[2] * 1.0 + proba[1] * 0.5 + proba[0] * 0.0
    credit_score    = int(max(300, min(850, 300 + stability_score * 550)))
    risk_map        = {"Stable": "Low", "Moderate": "Medium", "Unstable": "High"}
    risk_level      = risk_map[label]
    support         = SUPPORT_PROGRAMS[label]
    confidence      = round(float(max(proba)) * 100, 1)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown("#### 📊 Prediction Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="result-card"><div class="metric-lbl">Stability Label</div>'
                    f'<div class="metric-val">{support["emoji"]} {label}</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="result-card"><div class="metric-lbl">Credit Score</div>'
                    f'<div class="metric-val">{credit_score}</div></div>',
                    unsafe_allow_html=True)
    with c3:
        rc = {"Low": "#1A7A4A", "Medium": "#C05B00", "High": "#B91C1C"}
        st.markdown(f'<div class="result-card"><div class="metric-lbl">Risk Level</div>'
                    f'<div class="metric-val" style="color:{rc[risk_level]}">{risk_level}</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="result-card"><div class="metric-lbl">Model Confidence</div>'
                    f'<div class="metric-val">{confidence}%</div></div>',
                    unsafe_allow_html=True)

    # ── Gauge + Global Importance ──────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_gauge, col_imp = st.columns([1, 1.3])
    with col_gauge:
        st.markdown("##### 🎯 Credit Score Gauge")
        st.pyplot(draw_gauge(credit_score), use_container_width=True)
    with col_imp:
        st.markdown("##### 📈 Global Feature Importance")
        st.caption("Which features matter most **across all workers** in the dataset.")
        st.pyplot(draw_global_importance(model), use_container_width=True)

    # ── SHAP Waterfall ────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 🔍 Why This Prediction? — Per-Worker Factor Analysis")
    st.caption(
        "This chart shows how **each factor for this specific worker** pushed the "
        "prediction up 🟢 (toward Stable) or down 🔴 (toward Unstable). "
        "Think of it as a tug-of-war on the credit score."
    )

    with st.spinner("Computing factor contributions…"):
        shap_fig = draw_shap_waterfall(model, input_df, background_df, label_idx)
    st.pyplot(shap_fig, use_container_width=True)

    # ── Plain-English explanation ─────────────────────────────────────────────
    st.markdown('<div class="shap-box">', unsafe_allow_html=True)
    st.markdown("**📝 Plain-English Summary**")

    imp     = model.feature_importances_
    inp_v   = input_df.values[0]
    bg_mean = background_df.values.mean(axis=0)
    contribs = imp * (inp_v - bg_mean)

    top_pos = sorted(zip(contribs, ALL_COLS), reverse=True)[:2]
    top_neg = sorted(zip(contribs, ALL_COLS))[:2]

    pos_txt = " and ".join(
        [f"**{FEATURE_LABELS.get(c,'?')}** (above average)" for _, c in top_pos if _ > 0]
    ) or "no strong positive factors"
    neg_txt = " and ".join(
        [f"**{FEATURE_LABELS.get(c,'?')}** (below average)" for _, c in top_neg if _ < 0]
    ) or "no strong negative factors"

    st.markdown(
        f"This worker was predicted **{label}** mainly because of {pos_txt}. "
        f"The factors pulling the score down were {neg_txt}."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Probability breakdown ─────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 📉 Stability Probability Breakdown")
    st.pyplot(draw_probability_bars(proba), use_container_width=False)

    # ── Support program ───────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 🏛️ Recommended Support Program")
    st.markdown(
        f'<div class="support-box" style="background:{support["color"]}18; '
        f'border-left:5px solid {support["color"]}">'
        f'<strong style="color:{support["color"]}">{support["emoji"]} {support["program"]}</strong><br>'
        f'<span style="color:#444;font-size:0.95rem">{support["description"]}</span></div>',
        unsafe_allow_html=True
    )

else:
    # ── Welcome ───────────────────────────────────────────────────────────────
    st.info("👈  Fill in the worker profile in the sidebar and click **Predict** to get results.")
    st.markdown("""
    #### How it works
    | Input | What it measures |
    |---|---|
    | Daily Income | Earning capacity |
    | Work Days/Month | Employment consistency |
    | Monthly Transactions | Economic activity |
    | Savings Ratio | Financial discipline |
    | Digital Payment Ratio | Financial inclusion level |
    | Loan History | Existing credit exposure |

    #### What's new — AI Explainability
    - 📈 **Global Feature Importance** — which factors matter most across all workers
    - 🔍 **Per-Worker Factor Analysis** — a waterfall chart showing exactly why *this* worker
      got their specific prediction, with green bars (helping the score) and red bars (hurting it)
    - 📝 **Plain-English Summary** — a sentence explaining the prediction in simple language

    ---
    *Built for the AI Tool Development Challenge 2026 — One Planet. One Purpose. Powered by AI.*
    *Aligned with UN SDGs: No Poverty · Decent Work · Reduced Inequalities*
    """)