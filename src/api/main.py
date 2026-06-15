"""
main.py
-------
FastAPI credit risk scoring API.

Endpoints:
  GET  /health        — liveness check
  GET  /model/info    — model metadata
  POST /score         — score a single loan application
  POST /score/batch   — score multiple applications
  GET  /metrics       — model performance metrics
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Optional
import json
import time

ROOT     = Path(__file__).resolve().parents[2]
CHAMPION = ROOT / "models" / "champion"

# ── load model once at startup ─────────────────────────────────────────────────
try:
    MODEL = joblib.load(CHAMPION / "scorecard_model.pkl")
    SCORECARD_POINTS = pd.read_csv(CHAMPION / "scorecard_points.csv")
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Model load error: {e}")

# WoE lookup — loaded from DuckDB at startup
WOE_LOOKUP: dict = {}

def load_woe_lookup():
    """Load WoE table into memory for fast inference."""
    import duckdb
    DB_PATH = ROOT / "data" / "processed" / "credit_risk.duckdb"
    conn    = duckdb.connect(str(DB_PATH), read_only=True)
    woe_df  = conn.execute("SELECT * FROM features.woe_table").df()
    conn.close()

    for _, row in woe_df.iterrows():
        key = (row['feature'], str(row['bin']))
        WOE_LOOKUP[key] = float(row['woe'])

    print(f"WoE lookup loaded — {len(WOE_LOOKUP)} entries")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Credit Risk Scoring API",
    description = "Production-grade credit scorecard — LR Scorecard Champion Model",
    version     = "1.0.0",
)

@app.on_event("startup")
async def startup_event():
    load_woe_lookup()
    print("API ready.")


# ── request / response schemas ─────────────────────────────────────────────────
class LoanApplication(BaseModel):
    """
    Input schema — mirrors a real loan application form.
    All fields have validation rules matching business constraints.
    """
    loan_amnt          : float = Field(..., gt=0,   le=40000,
                                       description="Loan amount requested ($)")
    term_months        : int   = Field(...,
                                       description="Loan term (36 or 60 months)")
    int_rate           : float = Field(..., gt=0,   le=35,
                                       description="Interest rate (%)")
    grade              : str   = Field(...,
                                       description="LC loan grade (A-G)")
    emp_length_yrs     : float = Field(..., ge=0,   le=10,
                                       description="Employment length (years)")
    annual_inc         : float = Field(..., gt=0,
                                       description="Annual income ($)")
    home_ownership     : str   = Field(...,
                                       description="RENT / OWN / MORTGAGE")
    dti                : float = Field(..., ge=0,   le=60,
                                       description="Debt-to-income ratio")
    fico_avg           : float = Field(..., ge=580, le=850,
                                       description="FICO score")
    inq_last_6mths     : int   = Field(..., ge=0,
                                       description="Credit inquiries last 6 months")
    open_acc           : int   = Field(..., ge=0,
                                       description="Open credit accounts")
    pub_rec            : int   = Field(..., ge=0,
                                       description="Public derogatory records")
    revol_util         : float = Field(..., ge=0,   le=100,
                                       description="Revolving utilization (%)")
    total_acc          : int   = Field(..., ge=0,
                                       description="Total credit accounts")
    delinq_2yrs        : int   = Field(..., ge=0,
                                       description="Delinquencies last 2 years")
    verification_status: str   = Field(...,
                                       description="Verified / Source Verified / Not Verified")
    purpose            : str   = Field(...,
                                       description="Loan purpose")
    application_type   : str   = Field(default="Individual",
                                       description="Individual / Joint App")

    @field_validator('grade')
    @classmethod
    def grade_must_be_valid(cls, v):
        if v.upper() not in list('ABCDEFG'):
            raise ValueError("Grade must be A-G")
        return v.upper()

    @field_validator('term_months')
    @classmethod
    def term_must_be_valid(cls, v):
        if v not in [36, 60]:
            raise ValueError("Term must be 36 or 60 months")
        return v

    @field_validator('home_ownership')
    @classmethod
    def ownership_must_be_valid(cls, v):
        valid = ['RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE']
        if v.upper() not in valid:
            raise ValueError(f"home_ownership must be one of {valid}")
        return v.upper()


class ScoreResponse(BaseModel):
    credit_score        : int
    probability_of_default: float
    risk_tier           : str
    decision            : str
    decision_reason     : str
    scorecard_points    : dict
    processing_time_ms  : float


class BatchScoreResponse(BaseModel):
    results             : list
    total_applications  : int
    approved            : int
    declined            : int
    processing_time_ms  : float


# ── feature engineering (mirrors sql_pipeline.py logic) ───────────────────────
def engineer_features(app: LoanApplication) -> dict:
    """
    Replicates the SQL feature engineering pipeline for inference.
    Must exactly mirror what was done during training.
    """
    # bands
    if app.fico_avg < 620:   fico_band = '1_below_620'
    elif app.fico_avg < 660: fico_band = '2_620_659'
    elif app.fico_avg < 700: fico_band = '3_660_699'
    elif app.fico_avg < 740: fico_band = '4_700_739'
    elif app.fico_avg < 780: fico_band = '5_740_779'
    else:                    fico_band = '6_780_plus'

    if app.dti < 10:   dti_band = '1_low'
    elif app.dti < 20: dti_band = '2_medium'
    elif app.dti < 30: dti_band = '3_high'
    else:              dti_band = '4_very_high'

    if app.loan_amnt < 5000:   
        loan_amnt_band = '1_micro'
    elif app.loan_amnt < 10000: 
        loan_amnt_band = '2_small'
    elif app.loan_amnt < 20000: 
        loan_amnt_band = '3_medium'
    elif app.loan_amnt < 30000: 
        loan_amnt_band = '4_large'
    else:                       
        loan_amnt_band = '5_xlarge'

    if app.emp_length_yrs < 2:  
        emp_stability = '1_junior'
    elif app.emp_length_yrs < 5: 
        emp_stability = '2_mid'
    elif app.emp_length_yrs < 8: 
        emp_stability = '3_senior'
    else:
        emp_stability = '4_veteran'

    if app.int_rate < 8:    int_rate_band = '1_prime'
    elif app.int_rate < 12: int_rate_band = '2_near_prime'
    elif app.int_rate < 16: int_rate_band = '3_subprime'
    elif app.int_rate < 22: int_rate_band = '4_deep_subprime'
    else:                   int_rate_band = '5_distressed'

    # WoE lookup
    def woe(feature, bin_val):
        key = (feature, str(bin_val))
        return WOE_LOOKUP.get(key, 0.0)

    return {
        'grade_woe'              : woe('grade',               app.grade),
        'fico_band_woe'          : woe('fico_band',           fico_band),
        'dti_band_woe'           : woe('dti_band',            dti_band),
        'loan_amnt_band_woe'     : woe('loan_amnt_band',      loan_amnt_band),
        'emp_stability_woe'      : woe('emp_stability',        emp_stability),
        'int_rate_band_woe'      : woe('int_rate_band',        int_rate_band),
        'home_ownership_woe'     : woe('home_ownership',       app.home_ownership),
        'verification_status_woe': woe('verification_status',  app.verification_status),
        'purpose_woe'            : woe('purpose',              app.purpose),
        'application_type_woe'   : woe('application_type',     app.application_type),
    }


def prob_to_score(prob: float) -> int:
    """Convert PD probability to 300-850 credit score."""
    factor     = 20 / np.log(2)
    base_score = 600
    base_odds  = 50
    offset     = base_score - factor * np.log(base_odds)
    odds       = (1 - prob) / max(prob, 1e-10)
    score      = offset + factor * np.log(max(odds, 1e-10))
    return int(np.clip(round(score), 300, 850))


def get_risk_tier(score: int) -> tuple:
    """Map score to risk tier, decision, and reason."""
    if score >= 720:
        return ("Prime",         "APPROVED",
                "Strong credit profile — low default risk")
    elif score >= 680:
        return ("Near-Prime",    "APPROVED",
                "Acceptable credit profile — moderate risk")
    elif score >= 640:
        return ("Sub-Prime",     "APPROVED",
                "Elevated risk — approve with higher rate")
    elif score >= 600:
        return ("Deep Sub-Prime","REVIEW",
                "High default risk — manual underwriting required")
    else:
        return ("High Risk",     "DECLINED",
                "Credit profile does not meet minimum threshold")


# ── endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status"      : "healthy",
        "model_loaded": MODEL_LOADED,
        "woe_entries" : len(WOE_LOOKUP),
        "version"     : "1.0.0"
    }


@app.get("/model/info")
async def model_info():
    return {
        "model_type"     : "Logistic Regression Scorecard",
        "champion"       : True,
        "features"       : "WoE-encoded (10 features)",
        "score_range"    : "300-850",
        "training_data"  : "LendingClub 2007-2015",
        "validation_auc" : 0.6968,
        "validation_ks"  : 0.2851,
        "validation_gini": 0.3935,
        "oot_auc"        : 0.6883,
        "oot_ks"         : 0.2738,
    }


@app.get("/metrics")
async def metrics():
    return {
        "champion_model": {
            "validation": {"auc": 0.6968, "ks": 0.2851, "gini": 0.3935},
            "oot"       : {"auc": 0.6883, "ks": 0.2738, "gini": 0.3767},
        },
        "challenger_model": {
            "validation": {"auc": 0.7097, "ks": 0.3044, "gini": 0.4195},
            "oot"       : {"auc": 0.6966, "ks": 0.2873, "gini": 0.3932},
        },
        "ab_test": {
            "z_score"              : 4.8759,
            "p_value"              : 0.000001,
            "significant"          : True,
            "prob_challenger_wins" : 1.0,
            "el_reduction_usd"     : 5620651,
            "el_reduction_pct"     : 10.19,
            "recommendation"       : "DEPLOY CHALLENGER"
        }
    }


@app.post("/score", response_model=ScoreResponse)
async def score_application(application: LoanApplication):
    """
    Score a single loan application.
    Returns credit score (300-850), PD probability, risk tier, and decision.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503,
                            detail="Model not loaded")

    start = time.time()

    try:
        # engineer features
        features = engineer_features(application)
        X        = pd.DataFrame([features])

        # predict
        prob  = float(MODEL.predict_proba(X)[0, 1])
        score = prob_to_score(prob)
        tier, decision, reason = get_risk_tier(score)

        # scorecard contribution per feature
        points_contrib = {}
        for feat, woe_val in features.items():
            coef_row = SCORECARD_POINTS[
                SCORECARD_POINTS['feature'] == feat]
            if not coef_row.empty:
                pts = float(coef_row['points_per_unit_woe'].values[0])
                points_contrib[feat] = round(woe_val * pts, 2)

        elapsed = (time.time() - start) * 1000

        return ScoreResponse(
            credit_score             = score,
            probability_of_default   = round(prob, 6),
            risk_tier                = tier,
            decision                 = decision,
            decision_reason          = reason,
            scorecard_points         = points_contrib,
            processing_time_ms       = round(elapsed, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(applications: list[LoanApplication]):
    """Score multiple applications in one call."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503,
                            detail="Model not loaded")
    if len(applications) > 1000:
        raise HTTPException(status_code=400,
                            detail="Batch size limit is 1000")

    start   = time.time()
    results = []

    for app_data in applications:
        features = engineer_features(app_data)
        X        = pd.DataFrame([features])
        prob     = float(MODEL.predict_proba(X)[0, 1])
        score    = prob_to_score(prob)
        tier, decision, reason = get_risk_tier(score)
        results.append({
            "credit_score"            : score,
            "probability_of_default"  : round(prob, 6),
            "risk_tier"               : tier,
            "decision"                : decision,
        })

    elapsed  = (time.time() - start) * 1000
    approved = sum(1 for r in results if r['decision'] == 'APPROVED')
    declined = sum(1 for r in results if r['decision'] == 'DECLINED')

    return BatchScoreResponse(
        results              = results,
        total_applications   = len(results),
        approved             = approved,
        declined             = len(results) - approved,
        processing_time_ms   = round(elapsed, 2)
    )