import joblib
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path

ROOT     = Path().resolve()
CHAMPION = ROOT / "models" / "champion"

model  = joblib.load(CHAMPION / "scorecard_model.pkl")
sc_pts = pd.read_csv(CHAMPION / "scorecard_points.csv")

print("Scorecard Points Table:")
print(sc_pts.to_string())

intercept = model.named_steps['lr'].intercept_[0]
coefs     = model.named_steps['lr'].coef_[0]
print(f"\nIntercept: {intercept}")
print(f"Coefficients: {coefs}")

conn = duckdb.connect(
    str(ROOT / "data" / "processed" / "credit_risk.duckdb"),
    read_only=True
)
woe = conn.execute("""
    SELECT feature, bin, woe
    FROM features.woe_table
    ORDER BY feature, woe DESC
""").df()
conn.close()

print("\nBest WoE value per feature:")
print(woe.groupby("feature").first().reset_index()[
    ["feature", "bin", "woe"]].to_string())

# simulate best possible applicant
WOE_FEATURES = [
    'grade_woe', 'fico_band_woe', 'dti_band_woe',
    'loan_amnt_band_woe', 'emp_stability_woe', 'int_rate_band_woe',
    'home_ownership_woe', 'verification_status_woe',
    'purpose_woe', 'application_type_woe'
]

# simulate BEST applicant — lowest WoE features (Grade A, high FICO, low DTI)
best_woe = woe.groupby("feature")["woe"].min()  # MIN not MAX
X = pd.DataFrame([[
    best_woe.get("grade", 0),
    best_woe.get("fico_band", 0),
    best_woe.get("dti_band", 0),
    best_woe.get("loan_amnt_band", 0),
    best_woe.get("emp_stability", 0),
    best_woe.get("int_rate_band", 0),
    best_woe.get("home_ownership", 0),
    best_woe.get("verification_status", 0),
    best_woe.get("purpose", 0),
    best_woe.get("application_type", 0),
]], columns=WOE_FEATURES)

prob = float(model.predict_proba(X)[0, 1])
print(f"\nBest possible applicant (min WoE = Grade A, high FICO):")
print(f"  Raw PD probability: {prob:.6f}")

# also check worst applicant
worst_woe = woe.groupby("feature")["woe"].max()
X2 = pd.DataFrame([[
    worst_woe.get("grade", 0),
    worst_woe.get("fico_band", 0),
    worst_woe.get("dti_band", 0),
    worst_woe.get("loan_amnt_band", 0),
    worst_woe.get("emp_stability", 0),
    worst_woe.get("int_rate_band", 0),
    worst_woe.get("home_ownership", 0),
    worst_woe.get("verification_status", 0),
    worst_woe.get("purpose", 0),
    worst_woe.get("application_type", 0),
]], columns=WOE_FEATURES)

prob2 = float(model.predict_proba(X2)[0, 1])
print(f"Worst possible applicant (max WoE = Grade G, low FICO):")
print(f"  Raw PD probability: {prob2:.6f}")

# what score range does our function produce?
for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    factor = 20 / np.log(2)
    offset = 600 - factor * np.log(50)
    odds   = (1 - p) / p
    score  = offset + factor * np.log(odds)
    score  = int(np.clip(round(score), 300, 850))
    print(f"  PD={p:.0%}  →  score={score}")

# correct approach: min-max rescale to 300-850
min_pd = 0.09
max_pd = 0.90

for p in [0.09, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.89]:
    score = 850 - ((p - min_pd) / (max_pd - min_pd)) * (850 - 300)
    score = int(np.clip(round(score), 300, 850))
    print(f"  PD={p:.0%}  →  score={score}")