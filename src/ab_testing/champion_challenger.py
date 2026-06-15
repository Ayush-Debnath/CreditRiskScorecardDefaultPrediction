"""
champion_challenger.py
----------------------
Implements a bank-grade champion-challenger framework.

The test:
  Champion  = Logistic Regression Scorecard (trained on 2007-2015)
  Challenger= XGBoost model (trained on 2007-2015)
  Arena     = OOT data (2017-2018) — neither model has seen this

Tests performed:
  1. Frequentist: Z-test on AUC difference
  2. Frequentist: Z-test on KS difference
  3. Bayesian A/B: probability that challenger beats champion
  4. Business impact: expected loss comparison
  5. Decision memo output
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[2]
DB_PATH    = ROOT / "data" / "processed" / "credit_risk.duckdb"
FIG_DIR    = ROOT / "reports" / "figures"
CHAMPION   = ROOT / "models" / "champion"
CHALLENGER = ROOT / "models" / "challenger"
REPORTS    = ROOT / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi"       : 150,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

WOE_FEATURES = [
    'grade_woe', 'fico_band_woe', 'dti_band_woe',
    'loan_amnt_band_woe', 'emp_stability_woe', 'int_rate_band_woe',
    'home_ownership_woe', 'verification_status_woe',
    'purpose_woe', 'application_type_woe'
]

RAW_FEATURES = [
    'loan_amnt', 'term_months', 'int_rate', 'annual_inc',
    'dti', 'fico_avg', 'revol_util', 'open_acc', 'total_acc',
    'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'emp_length_yrs',
    'installment_to_income', 'loan_to_income',
    'revol_bal_to_income', 'open_to_total_acc_ratio', 'has_derogatory'
]


def load_data_and_models():
    conn = duckdb.connect(str(DB_PATH))
    oot  = conn.execute(f"""
        SELECT {', '.join(WOE_FEATURES + RAW_FEATURES +
                          ['is_bad', 'loan_amnt', 'vintage_year'])}
        FROM features.model_input
        WHERE split = 'oot'
    """).df()
    conn.close()

    champion   = joblib.load(CHAMPION   / "scorecard_model.pkl")
    challenger = joblib.load(CHALLENGER / "xgb_model.pkl")

    print(f"OOT dataset: {len(oot):,} loans | "
          f"Bad rate: {oot['is_bad'].mean():.2%}")
    return oot, champion, challenger


# ── STEP 1: Bootstrap AUC confidence intervals ────────────────────────────────
def bootstrap_auc(y_true: np.ndarray,
                  y_prob: np.ndarray,
                  n_iterations: int = 1000,
                  ci: float = 0.95) -> dict:
    """
    Bootstrap AUC distribution to get confidence intervals.
    More robust than parametric assumptions for credit models.
    """
    aucs = []
    n    = len(y_true)
    rng  = np.random.RandomState(42)

    for _ in range(n_iterations):
        idx      = rng.randint(0, n, n)
        y_boot   = y_true[idx]
        p_boot   = y_prob[idx]
        if y_boot.sum() == 0 or y_boot.sum() == n:
            continue
        aucs.append(roc_auc_score(y_boot, p_boot))

    aucs   = np.array(aucs)
    alpha  = (1 - ci) / 2
    return {
        'mean' : float(np.mean(aucs)),
        'std'  : float(np.std(aucs)),
        'lower': float(np.percentile(aucs, alpha * 100)),
        'upper': float(np.percentile(aucs, (1 - alpha) * 100)),
        'dist' : aucs
    }


# ── STEP 2: Frequentist Z-test on AUC ────────────────────────────────────────
def ztest_auc(champ_boot: dict,
              chall_boot: dict,
              alpha: float = 0.05) -> dict:
    """
    Two-sample Z-test on bootstrapped AUC distributions.
    H0: AUC_challenger <= AUC_champion
    H1: AUC_challenger >  AUC_champion (one-tailed)
    """
    diff    = chall_boot['mean'] - champ_boot['mean']
    se      = np.sqrt(champ_boot['std']**2 + chall_boot['std']**2)
    z_score = diff / se
    p_value = 1 - stats.norm.cdf(z_score)  # one-tailed

    return {
        'diff'       : round(diff, 6),
        'z_score'    : round(z_score, 4),
        'p_value'    : round(p_value, 6),
        'significant': p_value < alpha,
        'alpha'      : alpha
    }


# ── STEP 3: KS statistic comparison ──────────────────────────────────────────
def compare_ks(y_true, champ_prob, chall_prob) -> dict:
    def ks(y, p):
        fpr, tpr, _ = roc_curve(y, p)
        return float(np.max(tpr - fpr))

    champ_ks = ks(y_true, champ_prob)
    chall_ks = ks(y_true, chall_prob)

    return {
        'champion_ks'  : round(champ_ks, 4),
        'challenger_ks': round(chall_ks, 4),
        'diff'         : round(chall_ks - champ_ks, 4),
        'challenger_wins': chall_ks > champ_ks
    }


# ── STEP 4: Bayesian A/B Test ─────────────────────────────────────────────────
def bayesian_ab_test(champ_boot: dict,
                     chall_boot: dict,
                     n_samples: int = 100_000) -> dict:
    """
    Bayesian comparison using bootstrapped AUC distributions
    as empirical posteriors.

    P(challenger > champion) estimated by Monte Carlo sampling.
    """
    rng = np.random.RandomState(42)

    champ_samples = rng.choice(champ_boot['dist'], n_samples, replace=True)
    chall_samples = rng.choice(chall_boot['dist'], n_samples, replace=True)

    prob_chall_wins   = float(np.mean(chall_samples > champ_samples))
    expected_uplift   = float(np.mean(chall_samples - champ_samples))
    uplift_ci_lower   = float(np.percentile(
        chall_samples - champ_samples, 2.5))
    uplift_ci_upper   = float(np.percentile(
        chall_samples - champ_samples, 97.5))

    return {
        'prob_challenger_wins': round(prob_chall_wins, 4),
        'expected_uplift'     : round(expected_uplift, 6),
        'uplift_ci_lower'     : round(uplift_ci_lower, 6),
        'uplift_ci_upper'     : round(uplift_ci_upper, 6),
        'champ_samples'       : champ_samples,
        'chall_samples'       : chall_samples
    }


# ── STEP 5: Business Impact — Expected Loss ───────────────────────────────────
def business_impact(oot: pd.DataFrame,
                    champ_prob: np.ndarray,
                    chall_prob: np.ndarray,
                    threshold: float = 0.25,
                    lgd: float = 0.60) -> dict:
    """
    Compares expected credit loss under champion vs challenger.

    Expected Loss = PD × LGD × EAD
      PD  = predicted probability of default (from model)
      LGD = Loss Given Default (assume 60% industry standard)
      EAD = Exposure at Default (loan amount)

    We approve loans where predicted PD < threshold.
    Compare: total expected loss of approved portfolios.
    """
    df = oot.copy()
    df['champ_prob'] = champ_prob
    df['chall_prob'] = chall_prob

    # approval decisions
    df['champ_approved'] = (df['champ_prob'] < threshold).astype(int)
    df['chall_approved'] = (df['chall_prob'] < threshold).astype(int)

    # expected loss for approved loans
    champ_approved = df[df['champ_approved'] == 1]
    chall_approved = df[df['chall_approved'] == 1]

    champ_el = (champ_approved['champ_prob'] *
                lgd *
                champ_approved['loan_amnt']).sum()
    chall_el = (chall_approved['chall_prob'] *
                lgd *
                chall_approved['loan_amnt']).sum()

    # actual bad loans approved (false negatives — the costly ones)
    champ_fn = champ_approved['is_bad'].sum()
    chall_fn = chall_approved['is_bad'].sum()

    champ_approval_rate = df['champ_approved'].mean()
    chall_approval_rate = df['chall_approved'].mean()

    return {
        'threshold'           : threshold,
        'lgd'                 : lgd,
        'champ_approval_rate' : round(champ_approval_rate, 4),
        'chall_approval_rate' : round(chall_approval_rate, 4),
        'champ_expected_loss' : round(champ_el, 0),
        'chall_expected_loss' : round(chall_el, 0),
        'el_reduction'        : round(champ_el - chall_el, 0),
        'el_reduction_pct'    : round((champ_el - chall_el) / champ_el * 100, 2),
        'champ_false_negatives': int(champ_fn),
        'chall_false_negatives': int(chall_fn),
        'fn_reduction'        : int(champ_fn - chall_fn)
    }


# ── PLOT: Champion vs Challenger Dashboard ────────────────────────────────────
def plot_ab_dashboard(champ_boot, chall_boot,
                      ztest, ks_result,
                      bayes, impact, oot, 
                      champ_prob, chall_prob):

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── [0,0] AUC Bootstrap Distributions ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(champ_boot['dist'], bins=50, alpha=0.6,
             color='#2196F3', label=f"Champion LR\n"
             f"AUC={champ_boot['mean']:.4f} "
             f"[{champ_boot['lower']:.4f}, {champ_boot['upper']:.4f}]",
             density=True)
    ax1.hist(chall_boot['dist'], bins=50, alpha=0.6,
             color='#FF5722', label=f"Challenger XGB\n"
             f"AUC={chall_boot['mean']:.4f} "
             f"[{chall_boot['lower']:.4f}, {chall_boot['upper']:.4f}]",
             density=True)
    ax1.set_title("Bootstrap AUC Distributions\n(1000 iterations)")
    ax1.set_xlabel("AUC")
    ax1.legend(fontsize=7)

    # ── [0,1] Bayesian Posterior of Uplift ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    uplift_dist = bayes['chall_samples'] - bayes['champ_samples']
    ax2.hist(uplift_dist, bins=80, color='#9C27B0', alpha=0.7, density=True)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5,
                label='No difference')
    ax2.axvline(x=bayes['expected_uplift'], color='green',
                linestyle='-', linewidth=1.5,
                label=f"E[uplift]={bayes['expected_uplift']:.4f}")
    pct_positive = (uplift_dist > 0).mean() * 100
    ax2.set_title(f"Bayesian Uplift Distribution\n"
                  f"P(Challenger > Champion) = {pct_positive:.1f}%")
    ax2.set_xlabel("AUC Uplift (Challenger - Champion)")
    ax2.legend(fontsize=8)

    # ── [0,2] KS Comparison Bar ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    models = ['Champion\n(LR Scorecard)', 'Challenger\n(XGBoost)']
    ks_vals = [ks_result['champion_ks'], ks_result['challenger_ks']]
    colors  = ['#2196F3', '#FF5722']
    bars    = ax3.bar(models, ks_vals, color=colors, width=0.4)
    ax3.axhline(y=0.30, color='orange', linestyle='--',
                linewidth=1, label='Acceptable (0.30)')
    ax3.axhline(y=0.40, color='green', linestyle='--',
                linewidth=1, label='Good (0.40)')
    for bar, val in zip(bars, ks_vals):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f"{val:.4f}", ha='center', fontweight='bold')
    ax3.set_title("KS Statistic Comparison\n(OOT 2017-2018)")
    ax3.set_ylabel("KS Statistic")
    ax3.legend(fontsize=8)
    ax3.set_ylim([0, max(ks_vals) * 1.25])

    # ── [1,0:2] ROC Curve Overlay ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    y_true = oot['is_bad'].values

    fpr_c, tpr_c, _ = roc_curve(y_true, champ_prob)
    fpr_x, tpr_x, _ = roc_curve(y_true, chall_prob)

    ax4.plot(fpr_c, tpr_c, color='#2196F3', linewidth=2.5,
             label=f"Champion LR (AUC={champ_boot['mean']:.4f})")
    ax4.plot(fpr_x, tpr_x, color='#FF5722', linewidth=2.5,
             label=f"Challenger XGB (AUC={chall_boot['mean']:.4f})")
    ax4.plot([0,1],[0,1],'k--', linewidth=1, label='Random')
    ax4.fill_between(fpr_c, tpr_c, alpha=0.05, color='#2196F3')
    ax4.fill_between(fpr_x, tpr_x, alpha=0.05, color='#FF5722')
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.set_title("ROC Curve — Champion vs Challenger (OOT 2017-2018)")
    ax4.legend()

    # ── [1,2] Z-test Results ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    ztest_color = '#4CAF50' if ztest['significant'] else '#FF9800'
    result_text = 'SIGNIFICANT ✓' if ztest['significant'] else 'NOT SIGNIFICANT'

    table_data = [
        ['Metric',          'Value'],
        ['AUC Champion',    f"{champ_boot['mean']:.4f}"],
        ['AUC Challenger',  f"{chall_boot['mean']:.4f}"],
        ['AUC Difference',  f"{ztest['diff']:+.4f}"],
        ['Z-Score',         f"{ztest['z_score']:.4f}"],
        ['P-Value',         f"{ztest['p_value']:.4f}"],
        ['Alpha',           f"{ztest['alpha']}"],
        ['Result',          result_text],
        ['P(Chall Wins)',   f"{bayes['prob_challenger_wins']:.2%}"],
    ]

    tbl = ax5.table(cellText=table_data[1:],
                    colLabels=table_data[0],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    tbl[(len(table_data)-1, 1)].set_facecolor(ztest_color)
    tbl[(len(table_data)-1, 1)].set_text_props(color='white',
                                                fontweight='bold')
    ax5.set_title("Frequentist Z-Test Results", pad=12)

    # ── [2,0:3] Business Impact ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    impact_data = [
        ['Metric', 'Champion (LR Scorecard)', 'Challenger (XGBoost)', 'Difference'],
        ['Approval Rate',
         f"{impact['champ_approval_rate']:.2%}",
         f"{impact['chall_approval_rate']:.2%}",
         f"{(impact['chall_approval_rate']-impact['champ_approval_rate']):.2%}"],
        ['Expected Loss ($)',
         f"${impact['champ_expected_loss']:,.0f}",
         f"${impact['chall_expected_loss']:,.0f}",
         f"${impact['el_reduction']:,.0f} saved"],
        ['EL Reduction (%)',
         '—', '—',
         f"{impact['el_reduction_pct']:.2f}%"],
        ['Bad Loans Approved (FN)',
         f"{impact['champ_false_negatives']:,}",
         f"{impact['chall_false_negatives']:,}",
         f"{impact['fn_reduction']:,} fewer"],
    ]

    tbl2 = ax6.table(cellText=impact_data[1:],
                     colLabels=impact_data[0],
                     loc='center', cellLoc='center',
                     colWidths=[0.25, 0.22, 0.22, 0.22])
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(10)
    tbl2.scale(1, 2.0)

    for j in range(4):
        tbl2[(0, j)].set_facecolor('#1565C0')
        tbl2[(0, j)].set_text_props(color='white', fontweight='bold')

    ax6.set_title("Business Impact Analysis — Expected Credit Loss Comparison",
                  pad=15, fontsize=13)

    plt.suptitle("Champion vs Challenger A/B Test — OOT Evaluation (2017-2018)",
                 fontsize=15, fontweight='bold', y=1.01)

    plt.savefig(FIG_DIR / "14_ab_test_dashboard.png",
                bbox_inches='tight', dpi=150)
    plt.show()
    print("A/B test dashboard saved.")


# ── Decision Memo ─────────────────────────────────────────────────────────────
def write_decision_memo(ztest, bayes, ks_result, impact,
                        champ_boot, chall_boot) -> None:
    """
    Writes a plain-English decision memo — the output a model risk
    committee or senior stakeholder actually reads.
    """
    decision = (
        "DEPLOY CHALLENGER"
        if (ztest['significant'] and
            bayes['prob_challenger_wins'] > 0.80 and
            ks_result['challenger_wins'])
        else "RETAIN CHAMPION"
    )

    memo = f"""
================================================================================
MODEL RISK MANAGEMENT — CHAMPION-CHALLENGER DECISION MEMO
================================================================================
Date        : {pd.Timestamp.today().strftime('%Y-%m-%d')}
Author      : Ayush Debnath
Project     : Credit Risk Scorecard — LendingClub Portfolio
Evaluation  : Out-of-Time (OOT) Sample, Vintage 2017-2018
--------------------------------------------------------------------------------

EXECUTIVE SUMMARY
-----------------
We evaluated the XGBoost Challenger model against the incumbent Logistic
Regression Scorecard (Champion) on an out-of-time holdout of {240506:,} loans
originated in 2017-2018. Neither model was trained on this data.

STATISTICAL RESULTS
-------------------
Metric                 Champion (LR)       Challenger (XGB)    Winner
AUC                    {champ_boot['mean']:.4f}              {chall_boot['mean']:.4f}              {'XGB ✓' if chall_boot['mean'] > champ_boot['mean'] else 'LR ✓'}
KS Statistic           {ks_result['champion_ks']:.4f}              {ks_result['challenger_ks']:.4f}              {'XGB ✓' if ks_result['challenger_wins'] else 'LR ✓'}
Gini Coefficient       {2*champ_boot['mean']-1:.4f}              {2*chall_boot['mean']-1:.4f}              {'XGB ✓' if chall_boot['mean'] > champ_boot['mean'] else 'LR ✓'}

HYPOTHESIS TEST (Z-TEST)
------------------------
H0: AUC_challenger <= AUC_champion
H1: AUC_challenger >  AUC_champion
Z-Score  : {ztest['z_score']:.4f}
P-Value  : {ztest['p_value']:.6f}
Alpha    : {ztest['alpha']}
Result   : {"Reject H0 — difference is statistically significant" if ztest['significant'] else "Fail to reject H0 — difference not significant"}

BAYESIAN ANALYSIS
-----------------
P(Challenger > Champion) : {bayes['prob_challenger_wins']:.2%}
Expected AUC Uplift      : {bayes['expected_uplift']:+.6f}
95% Credible Interval    : [{bayes['uplift_ci_lower']:+.6f}, {bayes['uplift_ci_upper']:+.6f}]

BUSINESS IMPACT (threshold={impact['threshold']}, LGD={impact['lgd']})
--------------------------------------------------------------------------------
                        Champion        Challenger      Delta
Approval Rate           {impact['champ_approval_rate']:.2%}          {impact['chall_approval_rate']:.2%}          {impact['chall_approval_rate']-impact['champ_approval_rate']:+.2%}
Expected Loss           ${impact['champ_expected_loss']:>12,.0f}  ${impact['chall_expected_loss']:>12,.0f}  ${impact['el_reduction']:>+12,.0f}
EL Reduction (%)        —               —               {impact['el_reduction_pct']:+.2f}%
Bad Loans Approved      {impact['champ_false_negatives']:>8,}        {impact['chall_false_negatives']:>8,}        {impact['fn_reduction']:>+8,}

RECOMMENDATION
--------------
{decision}

Rationale:
{"- Challenger shows statistically significant AUC improvement (p < 0.05)" if ztest['significant'] else "- AUC improvement is NOT statistically significant at 5% level"}
{"- Bayesian analysis confirms >80% probability challenger outperforms" if bayes['prob_challenger_wins'] > 0.80 else f"- Bayesian confidence only {bayes['prob_challenger_wins']:.0%} — below 80% threshold"}
{"- Challenger reduces expected credit loss" if impact['el_reduction'] > 0 else "- Challenger does NOT reduce expected loss"}
- Interpretability: LR Scorecard remains preferred for regulatory reporting
- Recommendation: Use XGB for decisioning, LR for regulatory submissions

CAVEATS
-------
- Analysis based on LendingClub public data (proxy for real portfolio)
- LGD assumed at 60% (industry standard, not calibrated to this portfolio)
- Macro environment 2017-2018 differs from training period 2007-2015
- Full deployment requires shadow scoring period before live cutover

================================================================================
"""
    memo_path = REPORTS / "decision_memo.txt"
    with open(memo_path, 'w', encoding='utf-8') as f:
        f.write(memo)
    print(memo)
    print(f"Decision memo saved → {memo_path}")


def main():
    print("Loading data and models ...")
    oot, champion, challenger = load_data_and_models()

    y_true     = oot['is_bad'].values
    champ_prob = champion.predict_proba(
        oot[WOE_FEATURES].fillna(0))[:, 1]
    chall_prob = challenger.predict_proba(
        oot[RAW_FEATURES].fillna(0))[:, 1]

    print("\n[1/5] Bootstrap AUC (1000 iterations each) ...")
    champ_boot = bootstrap_auc(y_true, champ_prob, n_iterations=1000)
    chall_boot = bootstrap_auc(y_true, chall_prob, n_iterations=1000)
    print(f"Champion  AUC: {champ_boot['mean']:.4f} "
          f"[{champ_boot['lower']:.4f}, {champ_boot['upper']:.4f}]")
    print(f"Challenger AUC: {chall_boot['mean']:.4f} "
          f"[{chall_boot['lower']:.4f}, {chall_boot['upper']:.4f}]")

    print("\n[2/5] Frequentist Z-test ...")
    ztest = ztest_auc(champ_boot, chall_boot)
    print(f"Z-Score: {ztest['z_score']} | "
          f"P-Value: {ztest['p_value']} | "
          f"Significant: {ztest['significant']}")

    print("\n[3/5] KS comparison ...")
    ks_result = compare_ks(y_true, champ_prob, chall_prob)
    print(f"Champion KS: {ks_result['champion_ks']} | "
          f"Challenger KS: {ks_result['challenger_ks']}")

    print("\n[4/5] Bayesian A/B test ...")
    bayes = bayesian_ab_test(champ_boot, chall_boot)
    print(f"P(Challenger wins): {bayes['prob_challenger_wins']:.2%}")
    print(f"Expected uplift: {bayes['expected_uplift']:+.6f}")

    print("\n[5/5] Business impact analysis ...")
    impact = business_impact(oot, champ_prob, chall_prob)
    print(f"EL reduction: ${impact['el_reduction']:,.0f} "
          f"({impact['el_reduction_pct']:.2f}%)")

    print("\nGenerating A/B dashboard ...")
    plot_ab_dashboard(champ_boot, chall_boot, ztest,
                      ks_result, bayes, impact,
                      oot, champ_prob, chall_prob)

    print("\nWriting decision memo ...")
    write_decision_memo(ztest, bayes, ks_result,
                        impact, champ_boot, chall_boot)


if __name__ == "__main__":
    main()