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