"""
metrics.py
----------
Credit risk specific evaluation metrics and plots:
  1. ROC Curve comparison (all 3 models)
  2. KS Plot (separation between good and bad)
  3. Scorecard bin analysis (bad rate by score band)
  4. Population Stability Index (PSI) — model monitoring
  5. Calibration plot (predicted vs actual bad rate)
  6. Lift & Gains chart
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[2]
DB_PATH    = ROOT / "data" / "processed" / "credit_risk.duckdb"
FIG_DIR    = ROOT / "reports" / "figures"
CHAMPION   = ROOT / "models" / "champion"
CHALLENGER = ROOT / "models" / "challenger"
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


def load_data():
    conn = duckdb.connect(str(DB_PATH))
    val = conn.execute(f"""
        SELECT {', '.join(WOE_FEATURES + RAW_FEATURES + ['is_bad', 'vintage_year'])}
        FROM features.model_input WHERE split = 'validation'
    """).df()
    oot = conn.execute(f"""
        SELECT {', '.join(WOE_FEATURES + RAW_FEATURES + ['is_bad', 'vintage_year'])}
        FROM features.model_input WHERE split = 'oot'
    """).df()
    train = conn.execute(f"""
        SELECT {', '.join(WOE_FEATURES + RAW_FEATURES + ['is_bad', 'vintage_year'])}
        FROM features.model_input WHERE split = 'train'
    """).df()
    conn.close()
    return train, val, oot


def load_models():
    lr  = joblib.load(CHAMPION   / "scorecard_model.pkl")
    rf  = joblib.load(CHALLENGER / "rf_model.pkl")
    xgb = joblib.load(CHALLENGER / "xgb_model.pkl")
    return lr, rf, xgb


# ── PLOT 1: ROC Curve Comparison ──────────────────────────────────────────────
def plot_roc_curves(val, lr, rf, xgb):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    datasets = [
        ('Validation (2016)', val),
    ]

    models = [
        ('LR Scorecard (Champion)', lr,  WOE_FEATURES, '#2196F3'),
        ('Random Forest',           rf,  RAW_FEATURES, '#4CAF50'),
        ('XGBoost (Challenger)',    xgb, RAW_FEATURES, '#FF5722'),
    ]

    for ax, (ds_name, ds) in zip(axes, datasets):
        for name, model, feats, color in models:
            X = ds[feats].fillna(0)
            y = ds['is_bad']
            prob = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, prob)
            auc = roc_auc_score(y, prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})",
                    color=color, linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {ds_name}")
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # OOT on second plot
    ax = axes[1]
    ax.cla()
    oot_data = val  # placeholder — will fix below
    conn = duckdb.connect(str(DB_PATH))
    oot_df = conn.execute(f"""
        SELECT {', '.join(WOE_FEATURES + RAW_FEATURES + ['is_bad'])}
        FROM features.model_input WHERE split = 'oot'
    """).df()
    conn.close()

    for name, model, feats, color in models:
        X = oot_df[feats].fillna(0)
        y = oot_df['is_bad']
        prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — OOT (2017–2018)")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_roc_curves.png", bbox_inches='tight')
    plt.show()
    print("ROC curves saved.")


# ── PLOT 2: KS Plot ───────────────────────────────────────────────────────────
def plot_ks(val, lr):
    """
    KS plot shows the maximum separation between the
    cumulative distribution of good and bad loans by score.
    The KS statistic is the maximum vertical distance.
    """
    X   = val[WOE_FEATURES].fillna(0)
    y   = val['is_bad']
    prob = lr.predict_proba(X)[:, 1]

    df = pd.DataFrame({'prob': prob, 'is_bad': y})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    df['cum_bad']  = (df['is_bad'] == 1).cumsum() / (df['is_bad'] == 1).sum()
    df['cum_good'] = (df['is_bad'] == 0).cumsum() / (df['is_bad'] == 0).sum()
    df['ks']       = df['cum_bad'] - df['cum_good']

    ks_val = df['ks'].max()
    ks_idx = df['ks'].idxmax()

    fig, ax = plt.subplots(figsize=(10, 6))
    pct = np.linspace(0, 100, len(df))

    ax.plot(pct, df['cum_bad']  * 100, color='#E8654C',
            linewidth=2.5, label='Cumulative Bad %')
    ax.plot(pct, df['cum_good'] * 100, color='#4C9BE8',
            linewidth=2.5, label='Cumulative Good %')

    # KS arrow
    ks_pct = pct[ks_idx]
    ax.annotate(
        f'KS = {ks_val:.4f}',
        xy=(ks_pct, df['cum_bad'].iloc[ks_idx] * 100),
        xytext=(ks_pct + 8, df['cum_bad'].iloc[ks_idx] * 100 - 15),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=11, fontweight='bold'
    )
    ax.axvline(x=ks_pct, color='grey', linestyle='--', linewidth=1)
    ax.fill_between(pct,
                    df['cum_bad'] * 100,
                    df['cum_good'] * 100,
                    alpha=0.1, color='purple')

    ax.set_xlabel("% of Population (sorted by score, high risk first)")
    ax.set_ylabel("Cumulative %")
    ax.set_title(f"KS Plot — LR Scorecard | KS = {ks_val:.4f}")
    ax.legend()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_ks_plot.png", bbox_inches='tight')
    plt.show()
    print(f"KS Statistic: {ks_val:.4f}")


# ── PLOT 3: Score Distribution & Bin Analysis ─────────────────────────────────
def plot_score_bins(val, lr):
    """
    Converts model probability to a 300-850 credit score scale.
    Shows bad rate by score band — the key scorecard output.
    """
    X    = val[WOE_FEATURES].fillna(0)
    y    = val['is_bad']
    prob = lr.predict_proba(X)[:, 1]

    # convert prob to score (300-850 range)
    # score = base_score - factor * log(odds)
    factor     = 20 / np.log(2)
    base_score = 600
    base_odds  = 50
    offset     = base_score - factor * np.log(base_odds)

    odds  = (1 - prob) / prob
    score = offset + factor * np.log(np.clip(odds, 1e-10, None))
    score = np.clip(score, 300, 850).round(0)

    df = pd.DataFrame({'score': score, 'is_bad': y})
    df['score_band'] = pd.cut(
        df['score'],
        bins=[299, 550, 600, 620, 650, 680, 720, 760, 851],
        labels=['<550', '550-600', '600-620', '620-650',
                '650-680', '680-720', '720-760', '760+']
    )

    bin_stats = (df.groupby('score_band', observed=True)
                   .agg(total=('is_bad', 'count'),
                        bad=('is_bad', 'sum'))
                   .reset_index())
    bin_stats['bad_rate'] = bin_stats['bad'] / bin_stats['total'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # bad rate by score band
    colors = plt.cm.RdYlGn(
        np.linspace(0.1, 0.9, len(bin_stats)))[::-1]
    bars = axes[0].bar(bin_stats['score_band'].astype(str),
                       bin_stats['bad_rate'], color=colors)
    axes[0].set_title("Bad Rate by Score Band")
    axes[0].set_xlabel("Score Band")
    axes[0].set_ylabel("Bad Rate (%)")
    axes[0].tick_params(axis='x', rotation=30)
    for bar, val_r in zip(bars, bin_stats['bad_rate']):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f"{val_r:.1f}%", ha='center', fontsize=9)

    # volume by score band
    axes[1].bar(bin_stats['score_band'].astype(str),
                bin_stats['total'],
                color='#4C9BE8', alpha=0.8)
    axes[1].set_title("Population Volume by Score Band")
    axes[1].set_xlabel("Score Band")
    axes[1].set_ylabel("Number of Loans")
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    plt.suptitle("Scorecard Bin Analysis — Validation Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_score_bin_analysis.png", bbox_inches='tight')
    plt.show()

    print("\nScore Bin Summary:")
    print(bin_stats.to_string(index=False))
    return score


# ── PLOT 4: PSI — Population Stability Index ──────────────────────────────────
def plot_psi(train, val, oot, lr):
    """
    PSI measures how much the score distribution has shifted
    between training and a new population.

    PSI < 0.10 : No significant shift — model stable
    PSI 0.10-0.25: Moderate shift — investigate
    PSI > 0.25 : Major shift — model needs retraining
    """
    def get_scores(df, feats):
        X    = df[feats].fillna(0)
        prob = lr.predict_proba(X)[:, 1]
        odds = (1 - prob) / prob
        factor = 20 / np.log(2)
        offset = 600 - factor * np.log(50)
        score  = offset + factor * np.log(np.clip(odds, 1e-10, None))
        return np.clip(score, 300, 850)

    bins   = [299, 550, 580, 610, 640, 670, 700, 730, 760, 851]
    labels = ['<550','550-580','580-610','610-640',
              '640-670','670-700','700-730','730-760','760+']

    train_scores = get_scores(train, WOE_FEATURES)
    val_scores   = get_scores(val,   WOE_FEATURES)
    oot_scores   = get_scores(oot,   WOE_FEATURES)

    def psi_calc(expected, actual, bins, labels):
        exp_cnt = pd.cut(expected, bins=bins, labels=labels).value_counts().sort_index()
        act_cnt = pd.cut(actual,   bins=bins, labels=labels).value_counts().sort_index()
        exp_pct = exp_cnt / exp_cnt.sum()
        act_pct = act_cnt / act_cnt.sum()
        psi_df  = pd.DataFrame({
            'band'       : labels,
            'exp_pct'    : exp_pct.values,
            'act_pct'    : act_pct.values,
        })
        psi_df['psi_component'] = (
            (psi_df['act_pct'] - psi_df['exp_pct']) *
            np.log((psi_df['act_pct'] + 1e-10) /
                   (psi_df['exp_pct'] + 1e-10))
        )
        return psi_df, psi_df['psi_component'].sum()

    val_psi_df, val_psi = psi_calc(train_scores, val_scores, bins, labels)
    oot_psi_df, oot_psi = psi_calc(train_scores, oot_scores, bins, labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (psi_df, psi_val, title) in zip(axes, [
        (val_psi_df, val_psi, f"Validation PSI = {val_psi:.4f}"),
        (oot_psi_df, oot_psi, f"OOT PSI = {oot_psi:.4f}")
    ]):
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, psi_df['exp_pct'] * 100,
               width=w, label='Train (Expected)', color='#4C9BE8', alpha=0.8)
        ax.bar(x + w/2, psi_df['act_pct'] * 100,
               width=w, label='Actual', color='#E8654C', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_ylabel("% of Population")
        ax.set_title(title)
        ax.legend()

        # PSI threshold lines
        color = ('#4CAF50' if psi_val < 0.10
                 else '#FF9800' if psi_val < 0.25
                 else '#F44336')
        status = ('Stable ✓' if psi_val < 0.10
                  else 'Monitor ⚠' if psi_val < 0.25
                  else 'Retrain ✗')
        ax.text(0.98, 0.95, status,
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=12, fontweight='bold', color=color)

    plt.suptitle("Population Stability Index (PSI) — Model Monitoring",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_psi_monitoring.png", bbox_inches='tight')
    plt.show()

    print(f"\nValidation PSI: {val_psi:.4f} — "
          f"{'Stable' if val_psi < 0.10 else 'Monitor' if val_psi < 0.25 else 'Retrain'}")
    print(f"OOT PSI:        {oot_psi:.4f} — "
          f"{'Stable' if oot_psi < 0.10 else 'Monitor' if oot_psi < 0.25 else 'Retrain'}")


# ── PLOT 5: Gains & Lift Chart ────────────────────────────────────────────────
def plot_gains_lift(val, lr):
    X    = val[WOE_FEATURES].fillna(0)
    y    = val['is_bad']
    prob = lr.predict_proba(X)[:, 1]

    df = pd.DataFrame({'prob': prob, 'is_bad': y})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)

    total_bad  = y.sum()
    total      = len(y)
    cum_bad    = df['is_bad'].cumsum()
    pct_pop    = (np.arange(1, total + 1) / total) * 100
    gains      = (cum_bad / total_bad) * 100
    lift       = gains / pct_pop

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gains chart
    axes[0].plot(pct_pop, gains, color='#2196F3', linewidth=2.5,
                 label='Model')
    axes[0].plot([0, 100], [0, 100], 'k--', linewidth=1,
                 label='Random baseline')
    axes[0].fill_between(pct_pop, gains, pct_pop,
                         alpha=0.1, color='#2196F3')
    axes[0].set_xlabel("% Population Contacted")
    axes[0].set_ylabel("% Bad Loans Captured")
    axes[0].set_title("Cumulative Gains Chart")
    axes[0].legend()

    # annotate 30% population
    idx_30 = np.searchsorted(pct_pop, 30)
    axes[0].annotate(
        f'Top 30% captures\n{gains.iloc[idx_30]:.1f}% of bads',
        xy=(30, gains.iloc[idx_30]),
        xytext=(40, gains.iloc[idx_30] - 15),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=9
    )

    # Lift chart
    axes[1].plot(pct_pop, lift, color='#E8654C', linewidth=2.5)
    axes[1].axhline(y=1, color='k', linestyle='--',
                    linewidth=1, label='No lift baseline')
    axes[1].set_xlabel("% Population")
    axes[1].set_ylabel("Lift")
    axes[1].set_title("Lift Chart")
    axes[1].legend()
    axes[1].set_xlim([0, 100])

    plt.suptitle("Gains & Lift — LR Scorecard (Validation)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_gains_lift.png", bbox_inches='tight')
    plt.show()

    print(f"\nTop 10% of population captures "
          f"{gains.iloc[int(total*0.10)]:.1f}% of all bad loans")
    print(f"Top 30% of population captures "
          f"{gains.iloc[int(total*0.30)]:.1f}% of all bad loans")


# ── PLOT 6: Calibration Plot ──────────────────────────────────────────────────
def plot_calibration(val, lr, xgb):
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model, feats, color in [
        ('LR Scorecard', lr,  WOE_FEATURES, '#2196F3'),
        ('XGBoost',      xgb, RAW_FEATURES, '#FF5722'),
    ]:
        X    = val[feats].fillna(0)
        y    = val['is_bad']
        prob = model.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker='o',
                linewidth=2, label=name, color=color)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Bad Rate)")
    ax.set_title("Calibration Plot — Predicted vs Actual Bad Rate")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_calibration.png", bbox_inches='tight')
    plt.show()
    print("Calibration plot saved.")


def main():
    print("Loading data and models ...")
    train, val, oot = load_data()
    lr, rf, xgb     = load_models()

    print("\n[1/6] ROC Curves ...")
    plot_roc_curves(val, lr, rf, xgb)

    print("\n[2/6] KS Plot ...")
    plot_ks(val, lr)

    print("\n[3/6] Score Bin Analysis ...")
    plot_score_bins(val, lr)

    print("\n[4/6] PSI Monitoring ...")
    plot_psi(train, val, oot, lr)

    print("\n[5/6] Gains & Lift ...")
    plot_gains_lift(val, lr)

    print("\n[6/6] Calibration Plot ...")
    plot_calibration(val, lr, xgb)

    print(f"\nAll 6 evaluation plots saved to {FIG_DIR}")


if __name__ == "__main__":
    main()