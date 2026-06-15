"""
test_features.py
----------------
Unit tests for feature engineering logic.
Tests that WoE band assignments are correct.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def get_fico_band(fico):
    if fico < 620:   return '1_below_620'
    elif fico < 660: return '2_620_659'
    elif fico < 700: return '3_660_699'
    elif fico < 740: return '4_700_739'
    elif fico < 780: return '5_740_779'
    else:            return '6_780_plus'


def get_dti_band(dti):
    if dti < 10:   return '1_low'
    elif dti < 20: return '2_medium'
    elif dti < 30: return '3_high'
    else:          return '4_very_high'


def get_emp_stability(yrs):
    if yrs < 2:  return '1_junior'
    elif yrs < 5: return '2_mid'
    elif yrs < 8: return '3_senior'
    else:         return '4_veteran'


def get_int_rate_band(rate):
    if rate < 8:    return '1_prime'
    elif rate < 12: return '2_near_prime'
    elif rate < 16: return '3_subprime'
    elif rate < 22: return '4_deep_subprime'
    else:           return '5_distressed'


def get_loan_amnt_band(amnt):
    if amnt < 5000:    return '1_micro'
    elif amnt < 10000: return '2_small'
    elif amnt < 20000: return '3_medium'
    elif amnt < 30000: return '4_large'
    else:              return '5_xlarge'


def prob_to_score(prob):
    import numpy as np
    MIN_PD = 0.09
    MAX_PD = 0.90
    score  = 850 - ((prob - MIN_PD) / (MAX_PD - MIN_PD)) * 550
    return int(np.clip(round(score), 300, 850))


# ── FICO band tests ────────────────────────────────────────────────────────────
def test_fico_band_below_620():
    assert get_fico_band(580) == '1_below_620'
    assert get_fico_band(619) == '1_below_620'

def test_fico_band_620_659():
    assert get_fico_band(620) == '2_620_659'
    assert get_fico_band(659) == '2_620_659'

def test_fico_band_780_plus():
    assert get_fico_band(780) == '6_780_plus'
    assert get_fico_band(850) == '6_780_plus'

def test_fico_band_boundaries():
    assert get_fico_band(699) == '3_660_699'
    assert get_fico_band(700) == '4_700_739'
    assert get_fico_band(739) == '4_700_739'
    assert get_fico_band(740) == '5_740_779'


# ── DTI band tests ─────────────────────────────────────────────────────────────
def test_dti_low():
    assert get_dti_band(0)  == '1_low'
    assert get_dti_band(9)  == '1_low'

def test_dti_very_high():
    assert get_dti_band(30) == '4_very_high'
    assert get_dti_band(60) == '4_very_high'

def test_dti_boundaries():
    assert get_dti_band(10) == '2_medium'
    assert get_dti_band(20) == '3_high'
    assert get_dti_band(29) == '3_high'


# ── Employment stability tests ─────────────────────────────────────────────────
def test_emp_junior():
    assert get_emp_stability(0) == '1_junior'
    assert get_emp_stability(1) == '1_junior'

def test_emp_veteran():
    assert get_emp_stability(8)  == '4_veteran'
    assert get_emp_stability(10) == '4_veteran'

def test_emp_boundaries():
    assert get_emp_stability(2) == '2_mid'
    assert get_emp_stability(5) == '3_senior'
    assert get_emp_stability(7) == '3_senior'


# ── Interest rate band tests ───────────────────────────────────────────────────
def test_int_rate_prime():
    assert get_int_rate_band(5.0) == '1_prime'
    assert get_int_rate_band(7.9) == '1_prime'

def test_int_rate_distressed():
    assert get_int_rate_band(22.0) == '5_distressed'
    assert get_int_rate_band(31.0) == '5_distressed'

def test_int_rate_boundaries():
    assert get_int_rate_band(8.0)  == '2_near_prime'
    assert get_int_rate_band(12.0) == '3_subprime'
    assert get_int_rate_band(16.0) == '4_deep_subprime'


# ── Loan amount band tests ─────────────────────────────────────────────────────
def test_loan_amnt_micro():
    assert get_loan_amnt_band(1000) == '1_micro'
    assert get_loan_amnt_band(4999) == '1_micro'

def test_loan_amnt_xlarge():
    assert get_loan_amnt_band(30000) == '5_xlarge'
    assert get_loan_amnt_band(40000) == '5_xlarge'


# ── Score conversion tests ─────────────────────────────────────────────────────
def test_score_best_borrower():
    score = prob_to_score(0.09)
    assert score == 850

def test_score_worst_borrower():
    score = prob_to_score(0.90)
    assert score == 300

def test_score_clipping():
    assert prob_to_score(0.01) == 850
    assert prob_to_score(0.99) == 300

def test_score_midpoint():
    score = prob_to_score(0.495)
    assert 560 <= score <= 620