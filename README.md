# 💸 FinInclusion AI
### AI-Powered Financial Stability Predictor for Informal Workers in South Asia

> Built for the **AI Tool Development Challenge 2026 — One Planet. One Purpose. Powered by AI.**

---

## 🌍 The Problem

Across South Asia, an estimated **300–400 million workers** operate in the informal economy — street vendors, daily wage labourers, gig delivery workers, auto-rickshaw drivers, and small shopkeepers. Despite contributing significantly to national GDP, they are **systematically excluded from formal financial systems** because they have no traditional financial records — no payslips, no bank statements, no credit history.

Without these, they cannot access:
- Formal bank loans or micro-credit
- Government welfare and subsidy programs
- Insurance or emergency financial safety nets

**FinInclusion AI solves this** by using alternative economic indicators — income patterns, transaction frequency, digital payment behaviour — to assess financial stability and connect workers to the right support.

---

## 🎯 SDG Alignment

| Priority | SDG | Relevance |
|---|---|---|
| 🥇 Primary | **SDG 16** — Governance, Transparency & Justice | Explainable AI decisions for fair, accountable credit assessment |
| 🥈 Core | **SDG 1** — No Poverty | Connecting excluded workers to credit and welfare |
| 🥈 Core | **SDG 10** — Reduced Inequalities | Alternative scoring levels the playing field |
| 🥉 Supporting | **SDG 8** — Decent Work & Economic Growth | Enabling micro-credit for entrepreneurship |
| 🥉 Supporting | **SDG 11** — Sustainable Cities & Communities | Targeting urban informal workers |
| 🥉 Supporting | **SDG 9** — Smart Industry & Infrastructure | Scalable AI financial infrastructure |
| ➕ Additional | **SDG 17** — Partnerships for the Goals | Designed for MFI, NGO, and government partnerships |

---

## 🤖 What the AI Does

Instead of asking *"What is your bank balance?"*, the system asks:

> *"How consistently do you earn? How often do you transact? How digitally active are you?"*

It builds a **financial profile from behaviour** rather than formal records, and outputs:

| Output | Description |
|---|---|
| 🏷️ **Stability Label** | Stable / Moderate / Unstable |
| 📊 **Credit Score** | Alternative score from 300 to 850 |
| ⚠️ **Risk Category** | Low / Medium / High |
| 🏛️ **Support Program** | Micro-Investment / Micro-Credit / Government Welfare |
| 🔍 **Factor Analysis** | Which inputs helped or hurt the score — and by how much |

---

## 📁 Project Structure

```
finclusion/
│
├── data/
│   ├── generate_dataset.py         ← Generates 5,000 synthetic worker profiles
│   ├── raw/
│   │   └── informal_workers.csv    ← Raw synthetic dataset (auto-created)
│   └── processed/
│       └── processed_workers.csv   ← Cleaned, encoded, normalised dataset
│
├── models/
│   └── financial_model.pkl         ← Trained GradientBoosting model (auto-saved)
│
├── utils/
│   ├── preprocess.py               ← Full preprocessing pipeline
│   └── model.py                    ← ML training, evaluation & prediction
│
├── app/
│   └── main.py                     ← Streamlit web app (entry point)
│
├----Project-Overview-document          <-----------(Iilustrated Document)
│-----Project-Overview-document-Google-docs <------------------(Simplified Document)   
│
├── requirements.txt                ← All Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone or download the project
```bash
cd finclusion
```

### Step 2 — Create a virtual environment (recommended)
```bash
# Create
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Verify installation
```bash
python -c "import pandas, numpy, sklearn, joblib, streamlit, matplotlib; print('All libraries OK ✅')"
```

---

## 🚀 Running the Full Pipeline

Run these commands **in order** from inside the `finclusion/` folder:

### 1. Generate the synthetic dataset
```bash
python data/generate_dataset.py
```
Creates `data/raw/informal_workers.csv` with 5,000 worker profiles.

### 2. Preprocess the data
```bash
python utils/preprocess.py
```
Cleans, engineers features, encodes, and normalises. Saves to `data/processed/processed_workers.csv`.

### 3. Train the ML model
```bash
python utils/model.py
```
Trains Random Forest and Gradient Boosting, auto-selects the best, saves to `models/financial_model.pkl`.

### 4. Launch the Streamlit app
```bash
# Recommended on Windows
python -m streamlit run app/main.py

# If port 8501 is blocked, use a different port
python -m streamlit run app/main.py --server.port 8502
```

Opens in your browser at `http://localhost:8501`

---

## 🖥️ App Features

### Sidebar — Worker Profile Input
| Field | Type | Range |
|---|---|---|
| Occupation | Dropdown | Street Vendor, Driver, Laborer, Delivery Worker, Shopkeeper |
| Daily Income (₹) | Slider | ₹200 — ₹2,000 |
| Work Days / Month | Slider | 1 — 30 |
| Monthly Transactions | Slider | 1 — 120 |
| Savings Ratio | Slider | 0.00 — 0.40 |
| Digital Payment Ratio | Slider | 0.00 — 1.00 |
| Loan History | Radio | Yes / No |

### Results Panel
- **4 Metric Cards** — Stability Label, Credit Score, Risk Level, Model Confidence
- **Credit Score Gauge** — Semicircle needle chart (red → orange → green)
- **Global Feature Importance** — Which features matter most across all workers
- **Factor Contribution Analysis** — Waterfall chart showing exactly why this specific worker got their prediction (🟢 green = helps score, 🔴 red = hurts score)
- **Plain-English Summary** — One sentence explaining the prediction in simple language
- **Probability Breakdown** — Unstable / Moderate / Stable class probabilities
- **Support Program Recommendation** — Tailored welfare or credit program

---

## 🧠 ML Model Details

### Model Comparison
| Model | Test Accuracy | Selected |
|---|---|---|
| RandomForestClassifier | 89.9% | |
| GradientBoostingClassifier | **90.5%** | ✅ Auto-selected |

### Input Features (Alternative Indicators)
| Feature | What It Measures |
|---|---|
| `daily_income` | Average earnings per working day |
| `work_days_per_month` | Employment consistency |
| `monthly_transactions` | Economic activity level |
| `savings_ratio` | Financial discipline |
| `income_min` / `income_max` | Income range |
| `digital_payment_ratio` | Financial inclusion level |
| `loan_history` | Existing credit exposure (binary) |
| `income_consistency` | Engineered — min/max income ratio |
| `monthly_income` | Engineered — daily income × work days |

### Target Labels
| Label | Meaning | Credit Score Range |
|---|---|---|
| Stable | Financially resilient | 700 — 850 |
| Moderate | Some vulnerability | 500 — 699 |
| Unstable | High financial risk | 300 — 499 |

### Credit Score Formula
```
stability_score = P(Stable) × 1.0  +  P(Moderate) × 0.5  +  P(Unstable) × 0.0
credit_score    = 300 + (stability_score × 550)
```

### Classification Report (GradientBoosting on test set)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Unstable | 0.93 | 0.92 | 0.92 |
| Moderate | 0.90 | 0.91 | 0.90 |
| Stable | 0.89 | 0.87 | 0.88 |
| **Overall** | **0.91** | **0.91** | **0.90** |

---

## 📊 Dataset

The synthetic dataset contains **5,000 rows** generated with realistic occupation-based profiles:

| Occupation | Income Range | Work Days | Digital Ratio |
|---|---|---|---|
| Street Vendor | ₹200 – ₹900 | ~24/month | Low (5–35%) |
| Driver | ₹500 – ₹1,600 | ~22/month | Medium (30–80%) |
| Laborer | ₹200 – ₹700 | ~20/month | Very Low (2–20%) |
| Delivery Worker | ₹600 – ₹1,800 | ~25/month | High (50–95%) |
| Shopkeeper | ₹700 – ₹2,000 | ~26/month | Medium (25–75%) |

**Label distribution:** Moderate (49%) · Unstable (31%) · Stable (20%)

---

## 🔬 Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.2.2 | Data loading, cleaning, manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `scikit-learn` | 1.5.0 | ML models, preprocessing, evaluation |
| `joblib` | 1.4.2 | Save and load trained models |
| `streamlit` | 1.35.0 | Interactive web prototype |
| `matplotlib` | 3.9.0 | Charts and visualisations |
| `seaborn` | 0.13.2 | Statistical plots |

---

## 🗺️ Roadmap

| Phase | Status | Description |
|---|---|---|
| POC | ✅ Complete | Synthetic data, trained model, Streamlit prototype, explainability |
| Pilot | 🔜 Next | Partner with MFI/NGO, collect real worker data, retrain on 50k+ records |
| Regional Scale | 📋 Planned | Multilingual UI (Hindi, Urdu, Bengali), REST API, mobile PWA |
| National Scale | 📋 Planned | Federated learning, Aadhaar integration, cloud deployment |

---

## 🛡️ Responsible AI Principles

- **Transparency** — Every prediction includes a Factor Contribution Analysis explaining the decision
- **Fairness** — Model audited for bias across occupation groups and income levels
- **Privacy** — No raw personal data stored after prediction
- **Accountability** — Workers can request re-assessment after improving financial behaviour
- **Consent-first** — Workers explicitly agree to data use before assessment

---

## 📈 Key Performance Indicators

| KPI | Current (POC) | Target (Production) |
|---|---|---|
| Model Accuracy | 90.5% | ≥ 88% on real data |
| Recall — Unstable class | 92% | ≥ 90% |
| F1 Score (Macro Average) | 0.90 | ≥ 0.87 |
| Workers Assessed / year | — | 10,000 |
| Micro-credit Approvals Enabled | — | 2,000 |
| Welfare Enrollments | — | 1,500 |
| False Denial Rate | — | < 8% |

---

## 🏛️ Support Programs Mapped

| Prediction | Recommended Program | Max Support |
|---|---|---|
| 🟢 Stable | Micro-Investment Scheme | ₹50,000 loan at subsidised rate |
| 🟡 Moderate | Micro-Credit + Skills Training | ₹20,000 + free vocational training |
| 🔴 Unstable | Government Welfare Support | PM Jan Dhan Yojana + emergency relief |

---

## 👥 Who This Is For

- **Informal Workers** — Get a credit score and support recommendation without formal records
- **Microfinance Institutions** — Automate informal applicant screening
- **Government Welfare Agencies** — AI-assisted welfare eligibility identification
- **NGOs & Field Workers** — Rapid on-the-spot assessment tool
- **Policymakers & Researchers** — Aggregate insights on financial vulnerability patterns

---

## 📄 Project Documentation

Full submission documentation is in `FinInclusion_Project_Overview.docx` covering:
- Problem statement and SDG alignment in depth
- System architecture and end-to-end workflow
- Expected impact and real-world use cases
- Scalability and deployment roadmap
- Full KPI framework

---

## 📜 License

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

🔗 Full license text: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)

---

*Built with ❤️ for financial inclusion*
*AI Tool Development Challenge 2026 — One Planet. One Purpose. Powered by AI.*
*UN SDG Alignment: 16 · 1 · 10 · 8 · 11 · 9 · 17*
