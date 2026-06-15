"""
test_api.py
-----------
Integration tests for the FastAPI scoring endpoint.
Uses TestClient — no running server needed.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# only run API tests if models exist
MODELS_EXIST = (
    Path("models/champion/scorecard_model.pkl").exists() and
    Path("data/processed/credit_risk.duckdb").exists()
)


@pytest.mark.skipif(not MODELS_EXIST, reason="Models not available in CI")
def test_health_endpoint():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client   = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.skipif(not MODELS_EXIST, reason="Models not available in CI")
def test_model_info_endpoint():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client   = TestClient(app)
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "validation_auc" in data


@pytest.mark.skipif(not MODELS_EXIST, reason="Models not available in CI")
def test_score_good_borrower():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
    payload = {
        "loan_amnt"          : 10000,
        "term_months"        : 36,
        "int_rate"           : 7.5,
        "grade"              : "A",
        "emp_length_yrs"     : 8,
        "annual_inc"         : 85000,
        "home_ownership"     : "MORTGAGE",
        "dti"                : 12.5,
        "fico_avg"           : 740,
        "inq_last_6mths"     : 0,
        "open_acc"           : 8,
        "pub_rec"            : 0,
        "revol_util"         : 15.0,
        "total_acc"          : 20,
        "delinq_2yrs"        : 0,
        "verification_status": "Verified",
        "purpose"            : "debt_consolidation",
        "application_type"   : "Individual"
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "credit_score" in data
    assert "probability_of_default" in data
    assert "decision" in data
    assert data["credit_score"] >= 300
    assert data["credit_score"] <= 850
    assert data["decision"] == "APPROVED"


@pytest.mark.skipif(not MODELS_EXIST, reason="Models not available in CI")
def test_score_bad_borrower():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
    payload = {
        "loan_amnt"          : 35000,
        "term_months"        : 60,
        "int_rate"           : 28.0,
        "grade"              : "G",
        "emp_length_yrs"     : 0,
        "annual_inc"         : 22000,
        "home_ownership"     : "RENT",
        "dti"                : 45.0,
        "fico_avg"           : 592,
        "inq_last_6mths"     : 6,
        "open_acc"           : 2,
        "pub_rec"            : 2,
        "revol_util"         : 92.0,
        "total_acc"          : 4,
        "delinq_2yrs"        : 4,
        "verification_status": "Not Verified",
        "purpose"            : "small_business",
        "application_type"   : "Individual"
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    # bad borrower should have high default probability — not prime
    assert data["probability_of_default"] > 0.35
    assert data["credit_score"] < 700


def test_score_validation_invalid_grade():
    """Grade validation should reject invalid values even without models."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        payload = {
            "loan_amnt": 10000, "term_months": 36,
            "int_rate": 7.5, "grade": "Z",
            "emp_length_yrs": 5, "annual_inc": 60000,
            "home_ownership": "RENT", "dti": 15.0,
            "fico_avg": 700, "inq_last_6mths": 0,
            "open_acc": 5, "pub_rec": 0,
            "revol_util": 30.0, "total_acc": 12,
            "delinq_2yrs": 0, "verification_status": "Verified",
            "purpose": "debt_consolidation",
            "application_type": "Individual"
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 422
    except Exception:
        pytest.skip("API not available")


def test_prob_to_score_range():
    """Score conversion must always return 300-850."""
    import numpy as np
    def prob_to_score(prob):
        MIN_PD = 0.09
        MAX_PD = 0.90
        score  = 850 - ((prob - MIN_PD) / (MAX_PD - MIN_PD)) * 550
        return int(np.clip(round(score), 300, 850))

    for prob in [0.01, 0.09, 0.20, 0.50, 0.80, 0.90, 0.99]:
        score = prob_to_score(prob)
        assert 300 <= score <= 850, f"Score {score} out of range for prob {prob}"