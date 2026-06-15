# 🏦 Credit Risk Scorecard & Default Prediction

> **Production-grade credit risk model** built on 2.26M LendingClub loans.  
> Covers the full DS-I job description: SQL pipelines, statistical modelling,  
> ML, A/B testing, experiment tracking, REST API, and interactive dashboard.

---

## 🎯 Business Problem

Banks lose billions annually to loan defaults. This project builds an
**end-to-end credit risk scorecard** — from raw data ingestion to a live
scoring API — that predicts the probability of default and assigns a
credit score (300–850) to each loan applicant.

**Key result:** The XGBoost challenger model reduces expected credit loss
by **$5.6M (10.19%)** vs the incumbent scorecard, confirmed by a
statistically significant champion-challenger A/B test (p < 0.000001).

---

## 📊 Dataset

| Source | Rows | Columns | Notes |
|--------|------|---------|-------|
| LendingClub Accepted Loans | 2,260,668 | 150 | Primary modelling data |
| LendingClub Rejected Loans | 27,648,741 | 9 | Reject inference |
| FRED Economic Data | — | 2 | Macro enrichment (UNRATE, FEDFUNDS) |

---

## 🏗️ Project Architecture
raw CSV (2.2M rows)

↓

DuckDB 3-layer warehouse (raw → staging → features)

↓

WoE encoding + feature engineering (SQL)

↓

┌─────────────────────────────────────┐

│  Champion: LR Scorecard             │  ← interpretable, regulatory-grade

│  Challenger: XGBoost                │  ← higher performance

└─────────────────────────────────────┘

↓

Champion-Challenger A/B Test (OOT 2017-2018)

↓

FastAPI Scoring Endpoint  +  Streamlit Dashboard

---

## 📈 Model Results

| Model | Val AUC | Val KS | Val Gini | OOT AUC | OOT KS | OOT Gini |
|-------|---------|--------|----------|---------|--------|----------|
| LR Scorecard (Champion) | 0.6968 | 0.2851 | 0.3935 | 0.6883 | 0.2738 | 0.3767 |
| Random Forest | 0.7026 | 0.2944 | 0.4052 | 0.6889 | 0.2788 | 0.3779 |
| XGBoost (Challenger) | 0.7097 | 0.3044 | 0.4195 | 0.6966 | 0.2873 | 0.3932 |

**Industry benchmarks:** KS > 0.30 = Acceptable | AUC > 0.70 = Acceptable

---

## 🔬 Champion-Challenger A/B Test

| Metric | Value |
|--------|-------|
| Z-Score | 4.8759 |
| P-Value | 0.000001 |
| P(Challenger Wins) | 100.00% |
| Expected Loss — Champion | $55,135,618 |
| Expected Loss — Challenger | $49,514,967 |
| **EL Reduction** | **$5,620,651 (10.19%)** |
| Bad Loans Approved — Champion | 3,129 |
| Bad Loans Approved — Challenger | 3,016 |
| **Recommendation** | **DEPLOY CHALLENGER** |

---

## 🛠️ Technical Stack

| Layer | Tools |
|-------|-------|
| Data warehouse | DuckDB (3-layer: raw → staging → features) |
| Feature engineering | SQL window functions, CTEs, WoE encoding |
| Statistical analysis | Scipy, Statsmodels (KS test, Z-test, Bayesian A/B) |
| ML models | Scikit-learn, XGBoost |
| Experiment tracking | MLflow (SQLite backend) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| CI/CD | GitHub Actions |
| Data | LendingClub (Kaggle), FRED API |

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/credit-risk-scorecard.git
cd credit-risk-scorecard

# 2. Environment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Add data
# Download from Kaggle → data/raw/

# 4. Run full pipeline
python src/data/ingest.py
python src/data/clean.py
python src/data/sql_pipeline.py
python src/models/train.py
python src/evaluation/metrics.py
python src/ab_testing/champion_challenger.py

# 5. Launch API
uvicorn src.api.main:app --port 8000

# 6. Launch dashboard
streamlit run dashboard/app.py
```

---

## 🔌 API Usage

```bash
POST /score
{
  "loan_amnt": 10000,
  "grade": "A",
  "fico_avg": 740,
  "dti": 12.5,
  ...
}

→ {
  "credit_score": 782,
  "probability_of_default": 0.1124,
  "risk_tier": "Prime",
  "decision": "APPROVED",
  "scorecard_points": {...},
  "processing_time_ms": 2.5
}
```

**Swagger UI:** `http://localhost:8000/docs`

---

## 📁 Project Structure
credit-risk-scorecard/

├── data/raw/               ← Kaggle CSVs (gitignored)

├── data/processed/         ← DuckDB warehouse

├── src/

│   ├── data/               ← ingest, clean, SQL pipeline

│   ├── features/           ← WoE encoder

│   ├── models/             ← scorecard, train

│   ├── evaluation/         ← metrics, plots

│   ├── ab_testing/         ← champion-challenger, Bayesian

│   └── api/                ← FastAPI endpoint

├── dashboard/app.py        ← Streamlit dashboard

├── models/champion/        ← saved LR scorecard

├── models/challenger/      ← saved RF + XGBoost

├── reports/

│   ├── figures/            ← 14 evaluation charts

│   ├── model_comparison.csv

│   └── decision_memo.txt   ← A/B test decision memo

├── tests/                  ← pytest unit + integration tests

├── .github/workflows/      ← GitHub Actions CI

└── mlflow.db               ← MLflow experiment tracking

---

## 💡 Key Concepts Demonstrated

- **Reject inference** — using 27.6M rejected loans to correct survivorship bias
- **Weight of Evidence (WoE)** encoding — industry standard for credit scorecards
- **Out-of-time validation** — correct temporal split, no data leakage
- **Population Stability Index (PSI)** — production model monitoring
- **Champion-Challenger testing** — bank-grade model governance
- **Expected Loss formula** — PD × LGD × EAD business impact quantification
- **Regulatory interpretability** — LR scorecard with point contributions per factor

---

## 👤 Author

**Ayush Debnath**  
B.Tech CSE — Graphic Era Hill University (CGPA 8.47)  
[LinkedIn](https://linkedin.com/in/ayush) | [GitHub](https://github.com/Ayush)

*Data Analyst Intern — IIT Roorkee (Jan–Apr 2025)*