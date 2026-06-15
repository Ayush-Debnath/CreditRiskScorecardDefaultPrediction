"""
app.py
------
Streamlit dashboard for the Credit Risk Scorecard project.

Pages:
  1. Portfolio Overview    — dataset-level stats and EDA charts
  2. Loan Scorer          — interactive single application scorer
  3. Model Performance    — ROC, KS, scorecard bins, PSI
  4. A/B Test Results     — champion vs challenger dashboard
"""

import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT / "data"  / "processed" / "credit_risk.duckdb"
FIG_DIR  = ROOT / "reports" / "figures"
CHAMPION = ROOT / "models" / "champion"
CHALL    = ROOT / "models" / "challenger"

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

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Credit Risk Scorecard",
    page_icon  = "🏦",
    layout     = "wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4C9BE8;
    }
    .approved  { color: #4CAF50; font-size: 1.4rem; font-weight: 700; }
    .declined  { color: #F44336; font-size: 1.4rem; font-weight: 700; }
    .review    { color: #FF9800; font-size: 1.4rem; font-weight: 700; }
    .score-big { font-size: 3rem; font-weight: 800; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── cached data loaders ────────────────────────────────────────────────────────
@st.cache_data
def load_summary_stats():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    stats = conn.execute("""
        SELECT
            COUNT(*)                        AS total_loans,
            ROUND(AVG(is_bad)*100, 2)       AS bad_rate,
            ROUND(AVG(loan_amnt), 0)        AS avg_loan_amnt,
            ROUND(AVG(int_rate), 2)         AS avg_int_rate,
            ROUND(AVG(fico_avg), 0)         AS avg_fico,
            ROUND(AVG(annual_inc), 0)       AS avg_income,
            ROUND(AVG(dti), 2)              AS avg_dti
        FROM staging.loans
    """).df()
    conn.close()
    return stats.iloc[0]


@st.cache_data
def load_grade_stats():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("""
        SELECT grade,
               COUNT(*)                     AS total,
               ROUND(AVG(is_bad)*100, 2)    AS bad_rate,
               ROUND(AVG(int_rate), 2)      AS avg_rate
        FROM staging.loans
        GROUP BY grade ORDER BY grade
    """).df()
    conn.close()
    return df


@st.cache_data
def load_vintage_stats():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("""
        SELECT vintage_year,
               COUNT(*)                     AS total,
               ROUND(AVG(is_bad)*100, 2)    AS bad_rate,
               ROUND(AVG(loan_amnt), 0)     AS avg_loan
        FROM staging.loans
        GROUP BY vintage_year ORDER BY vintage_year
    """).df()
    conn.close()
    return df


@st.cache_data
def load_model_data():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    val = conn.execute(f"""
        SELECT {', '.join(WOE_FEATURES + RAW_FEATURES + ['is_bad'])}
        FROM features.model_input WHERE split = 'validation'
        USING SAMPLE 50000
    """).df()
    oot = conn.execute(f"""
        SELECT {', '.join(WOE_FEATURES + RAW_FEATURES + ['is_bad'])}
        FROM features.model_input WHERE split = 'oot'
        USING SAMPLE 50000
    """).df()
    conn.close()
    return val, oot


@st.cache_resource
def load_models():
    lr  = joblib.load(CHAMPION / "scorecard_model.pkl")
    xgb = joblib.load(CHALL    / "xgb_model.pkl")
    return lr, xgb


@st.cache_data
def load_woe_lookup():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df   = conn.execute("SELECT * FROM features.woe_table").df()
    conn.close()
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row['feature'], str(row['bin']))] = float(row['woe'])
    return lookup


# ── feature engineering for scorer ────────────────────────────────────────────
def engineer_woe(loan_amnt, term_months, int_rate, grade,
                 emp_length_yrs, annual_inc, home_ownership,
                 dti, fico_avg, inq_last_6mths, open_acc,
                 pub_rec, revol_util, total_acc, delinq_2yrs,
                 verification_status, purpose, application_type,
                 woe_lookup):

    if fico_avg < 620:   fico_band = '1_below_620'
    elif fico_avg < 660: fico_band = '2_620_659'
    elif fico_avg < 700: fico_band = '3_660_699'
    elif fico_avg < 740: fico_band = '4_700_739'
    elif fico_avg < 780: fico_band = '5_740_779'
    else:                fico_band = '6_780_plus'

    if dti < 10:   dti_band = '1_low'
    elif dti < 20: dti_band = '2_medium'
    elif dti < 30: dti_band = '3_high'
    else:          dti_band = '4_very_high'

    if loan_amnt < 5000:   lb = '1_micro'
    elif loan_amnt < 10000: lb = '2_small'
    elif loan_amnt < 20000: lb = '3_medium'
    elif loan_amnt < 30000: lb = '4_large'
    else:                   lb = '5_xlarge'

    if emp_length_yrs < 2:  es = '1_junior'
    elif emp_length_yrs < 5: es = '2_mid'
    elif emp_length_yrs < 8: es = '3_senior'
    else:                    es = '4_veteran'

    if int_rate < 8:    irb = '1_prime'
    elif int_rate < 12: irb = '2_near_prime'
    elif int_rate < 16: irb = '3_subprime'
    elif int_rate < 22: irb = '4_deep_subprime'
    else:               irb = '5_distressed'

    def w(feat, val):
        return woe_lookup.get((feat, str(val)), 0.0)

    return {
        'grade_woe'              : w('grade',               grade),
        'fico_band_woe'          : w('fico_band',           fico_band),
        'dti_band_woe'           : w('dti_band',            dti_band),
        'loan_amnt_band_woe'     : w('loan_amnt_band',      lb),
        'emp_stability_woe'      : w('emp_stability',        es),
        'int_rate_band_woe'      : w('int_rate_band',        irb),
        'home_ownership_woe'     : w('home_ownership',       home_ownership),
        'verification_status_woe': w('verification_status',  verification_status),
        'purpose_woe'            : w('purpose',              purpose),
        'application_type_woe'   : w('application_type',     application_type),
    }


def prob_to_score(prob):
    MIN_PD = 0.09
    MAX_PD = 0.90
    score  = 850 - ((prob - MIN_PD) / (MAX_PD - MIN_PD)) * (850 - 300)
    return int(np.clip(round(score), 300, 850))

# ── sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/Credit%20Risk-Scorecard-blue",
                 use_container_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "🏠 Portfolio Overview",
    "🎯 Loan Scorer",
    "📊 Model Performance",
    "🔬 A/B Test Results"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project:** Credit Risk Scorecard  
**Data:** LendingClub 2007-2018  
**Champion:** LR Scorecard  
**Challenger:** XGBoost  
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Portfolio Overview":
    st.title("🏦 Credit Risk Scorecard — Portfolio Overview")
    st.caption("LendingClub Loan Portfolio | 2007-2018 | 1.37M loans")

    stats       = load_summary_stats()
    grade_stats = load_grade_stats()
    vintage     = load_vintage_stats()

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Loans",    f"{int(stats['total_loans']):,}")
    col2.metric("Bad Rate",       f"{stats['bad_rate']}%")
    col3.metric("Avg Loan",       f"${int(stats['avg_loan_amnt']):,}")
    col4.metric("Avg FICO",       f"{int(stats['avg_fico'])}")
    col5.metric("Avg Int Rate",   f"{stats['avg_int_rate']}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bad Rate by Loan Grade")
        fig = px.bar(
            grade_stats, x='grade', y='bad_rate',
            color='bad_rate',
            color_continuous_scale='RdYlGn_r',
            labels={'bad_rate': 'Bad Rate (%)', 'grade': 'Grade'},
            text='bad_rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=380,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Loan Volume by Grade")
        fig = px.bar(
            grade_stats, x='grade', y='total',
            color='total',
            color_continuous_scale='Blues',
            labels={'total': 'Number of Loans', 'grade': 'Grade'}
        )
        fig.update_layout(showlegend=False, height=380,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Vintage Analysis — Bad Rate Over Time")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=vintage['vintage_year'], y=vintage['total'],
               name='Loan Volume', marker_color='#A8C8E8', opacity=0.7),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=vintage['vintage_year'], y=vintage['bad_rate'],
                   name='Bad Rate %', line=dict(color='#E8654C', width=3),
                   mode='lines+markers'),
        secondary_y=True
    )
    fig.update_yaxes(title_text="Loan Volume", secondary_y=False)
    fig.update_yaxes(title_text="Bad Rate (%)", secondary_y=True)
    fig.update_layout(height=380, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: LOAN SCORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Loan Scorer":
    st.title("🎯 Interactive Loan Scorer")
    st.caption("Enter application details to get an instant credit score and decision")

    woe_lookup = load_woe_lookup()
    lr, _      = load_models()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Loan Details")
        loan_amnt   = st.slider("Loan Amount ($)", 1000, 40000, 15000, 500)
        term_months = st.selectbox("Term (months)", [36, 60])
        int_rate    = st.slider("Interest Rate (%)", 5.0, 31.0, 12.0, 0.25)
        grade       = st.selectbox("Grade", ['A','B','C','D','E','F','G'])
        purpose     = st.selectbox("Purpose", [
            'debt_consolidation','credit_card','home_improvement',
            'other','major_purchase','small_business','car',
            'medical','moving','vacation','house','wedding',
            'renewable_energy','educational'
        ])

    with col2:
        st.subheader("Borrower Profile")
        annual_inc      = st.number_input("Annual Income ($)",
                                          10000, 500000, 65000, 5000)
        emp_length_yrs  = st.slider("Employment Length (years)", 0, 10, 5)
        home_ownership  = st.selectbox("Home Ownership",
                                       ['RENT','MORTGAGE','OWN','OTHER'])
        verification_status = st.selectbox("Verification Status",
                                           ['Verified',
                                            'Source Verified',
                                            'Not Verified'])
        application_type = st.selectbox("Application Type",
                                        ['Individual','Joint App'])

    with col3:
        st.subheader("Credit Bureau Data")
        fico_avg       = st.slider("FICO Score", 580, 850, 700)
        dti            = st.slider("DTI Ratio (%)", 0.0, 60.0, 18.0, 0.5)
        revol_util     = st.slider("Revolving Utilization (%)",
                                   0.0, 100.0, 45.0, 1.0)
        open_acc       = st.number_input("Open Accounts", 0, 50, 8)
        total_acc      = st.number_input("Total Accounts", 0, 100, 20)
        delinq_2yrs    = st.number_input("Delinquencies (2yr)", 0, 20, 0)
        pub_rec        = st.number_input("Public Records", 0, 10, 0)
        inq_last_6mths = st.number_input("Inquiries (6mo)", 0, 20, 1)

    st.markdown("---")

    if st.button("🔍 Score This Application", type="primary",
                 use_container_width=True):

        features = engineer_woe(
            loan_amnt, term_months, int_rate, grade,
            emp_length_yrs, annual_inc, home_ownership,
            dti, fico_avg, inq_last_6mths, open_acc,
            pub_rec, revol_util, total_acc, delinq_2yrs,
            verification_status, purpose, application_type,
            woe_lookup
        )

        X    = pd.DataFrame([features])
        prob = float(lr.predict_proba(X)[0, 1])
        score = prob_to_score(prob)

        if score >= 720:
            tier, decision, color = "Prime",         "✅ APPROVED", "approved"
        elif score >= 680:
            tier, decision, color = "Near-Prime",    "✅ APPROVED", "approved"
        elif score >= 640:
            tier, decision, color = "Sub-Prime",     "✅ APPROVED", "approved"
        elif score >= 600:
            tier, decision, color = "Deep Sub-Prime","⚠️ REVIEW",   "review"
        else:
            tier, decision, color = "High Risk",     "❌ DECLINED", "declined"

        # score display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Credit Score",  score)
        col2.metric("Default Prob",  f"{prob:.2%}")
        col3.metric("Risk Tier",     tier)
        col4.metric("Decision",      decision)

        # score gauge
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = score,
            domain= {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Credit Score", 'font': {'size': 20}},
            gauge = {
                'axis'    : {'range': [300, 850]},
                'bar'     : {'color': "#4C9BE8"},
                'steps'   : [
                    {'range': [300, 600], 'color': "#FFCDD2"},
                    {'range': [600, 640], 'color': "#FFE0B2"},
                    {'range': [640, 680], 'color': "#FFF9C4"},
                    {'range': [680, 720], 'color': "#DCEDC8"},
                    {'range': [720, 850], 'color': "#C8E6C9"},
                ],
                'threshold': {
                    'line' : {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # scorecard contribution waterfall
        st.subheader("Score Factor Breakdown")
        factor_names = {
            'grade_woe'              : 'Loan Grade',
            'fico_band_woe'          : 'FICO Score',
            'dti_band_woe'           : 'DTI Ratio',
            'loan_amnt_band_woe'     : 'Loan Amount',
            'emp_stability_woe'      : 'Employment',
            'int_rate_band_woe'      : 'Interest Rate',
            'home_ownership_woe'     : 'Home Ownership',
            'verification_status_woe': 'Verification',
            'purpose_woe'            : 'Loan Purpose',
            'application_type_woe'   : 'App Type'
        }

        sc_pts = pd.read_csv(CHAMPION / "scorecard_points.csv")
        contribs = []
        for feat, woe_val in features.items():
            row = sc_pts[sc_pts['feature'] == feat]
            if not row.empty:
                pts = float(row['points_per_unit_woe'].values[0])
                contribs.append({
                    'Factor'      : factor_names.get(feat, feat),
                    'Contribution': round(woe_val * pts, 2)
                })

        contrib_df = pd.DataFrame(contribs).sort_values(
            'Contribution', ascending=True)
        colors = ['#F44336' if x < 0 else '#4CAF50'
                  for x in contrib_df['Contribution']]

        fig2 = go.Figure(go.Bar(
            x          = contrib_df['Contribution'],
            y          = contrib_df['Factor'],
            orientation= 'h',
            marker_color= colors
        ))
        fig2.update_layout(
            title  = "Score Factor Contributions (positive = helps score)",
            height = 380,
            xaxis_title = "Score Points Contribution"
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    val, oot = load_model_data()
    lr, xgb  = load_models()

    tab1, tab2, tab3 = st.tabs(["ROC Curves", "Score Distribution", "PSI Monitoring"])

    with tab1:
        st.subheader("ROC Curves — Validation vs OOT")
        col1, col2 = st.columns(2)

        for col, (ds, ds_name) in zip([col1, col2],
                                       [(val, 'Validation (2016)'),
                                        (oot, 'OOT (2017-2018)')]):
            with col:
                fig = go.Figure()
                for name, model, feats, color in [
                    ('LR Scorecard', lr,  WOE_FEATURES, '#2196F3'),
                    ('XGBoost',      xgb, RAW_FEATURES, '#FF5722'),
                ]:
                    X    = ds[feats].fillna(0)
                    y    = ds['is_bad']
                    prob = model.predict_proba(X)[:, 1]
                    fpr, tpr, _ = roc_curve(y, prob)
                    auc = roc_auc_score(y, prob)
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, name=f"{name} (AUC={auc:.4f})",
                        line=dict(color=color, width=2)
                    ))
                fig.add_trace(go.Scatter(
                    x=[0,1], y=[0,1], name='Random',
                    line=dict(color='grey', dash='dash')
                ))
                fig.update_layout(
                    title=ds_name, height=380,
                    xaxis_title='FPR', yaxis_title='TPR'
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Score Distribution — Good vs Bad Loans")
        X_val = val[WOE_FEATURES].fillna(0)
        prob_val = lr.predict_proba(X_val)[:, 1]
        MIN_PD = 0.09
        MAX_PD = 0.90
        scores = np.clip(
            850 - ((prob_val - MIN_PD) / (MAX_PD - MIN_PD)) * 550,
            300, 850
        )

        score_df = pd.DataFrame({'score': scores, 'is_bad': val['is_bad']})
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=score_df[score_df['is_bad']==0]['score'],
            name='Good Loans', opacity=0.7,
            marker_color='#4C9BE8', nbinsx=50
        ))
        fig.add_trace(go.Histogram(
            x=score_df[score_df['is_bad']==1]['score'],
            name='Bad Loans', opacity=0.7,
            marker_color='#E8654C', nbinsx=50
        ))
        fig.update_layout(
            barmode='overlay', height=400,
            xaxis_title='Credit Score (300-850)',
            yaxis_title='Count',
            title='Score Distribution by Outcome'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Population Stability Index (PSI)")
        col1, col2 = st.columns(2)

        metrics_data = {
            "Validation PSI": 0.032,
            "OOT PSI"       : 0.078,
        }
        for col, (label, psi_val) in zip([col1, col2], metrics_data.items()):
            with col:
                status = ('🟢 Stable'   if psi_val < 0.10
                          else '🟡 Monitor' if psi_val < 0.25
                          else '🔴 Retrain')
                st.metric(label, f"{psi_val:.4f}", status)

        st.image(str(FIG_DIR / "11_psi_monitoring.png"),
                 caption="PSI — Train vs Validation vs OOT",
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: A/B TEST RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 A/B Test Results":
    st.title("🔬 Champion vs Challenger — A/B Test Results")
    st.caption("Out-of-Time evaluation on 240,506 loans (2017-2018)")

    # summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Champion AUC (OOT)",    "0.6883")
    col2.metric("Challenger AUC (OOT)",  "0.6966", "+0.0083")
    col3.metric("P(Challenger Wins)",    "100.00%")
    col4.metric("EL Reduction",          "$5.6M",   "10.19%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Statistical Test Results")
        test_df = pd.DataFrame([
            ["Z-Score",              "4.8759"],
            ["P-Value",              "0.000001"],
            ["Significant (α=0.05)", "✅ Yes"],
            ["Champion KS",          "0.2738"],
            ["Challenger KS",        "0.2873"],
            ["KS Winner",            "XGBoost ✅"],
            ["P(Challenger Wins)",   "100.00%"],
            ["Expected AUC Uplift",  "+0.008285"],
        ], columns=["Metric", "Value"])
        st.dataframe(test_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Business Impact")
        impact_df = pd.DataFrame([
            ["Approval Rate",        "16.16%",        "16.52%",        "+0.36%"],
            ["Expected Loss",        "$55,135,618",   "$49,514,967",   "-$5,620,651"],
            ["EL Reduction",         "—",             "—",             "10.19%"],
            ["Bad Loans Approved",   "3,129",         "3,016",         "-113"],
        ], columns=["Metric", "Champion", "Challenger", "Delta"])
        st.dataframe(impact_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("A/B Test Dashboard")
    st.image(str(FIG_DIR / "14_ab_test_dashboard.png"),
             caption="Full Champion-Challenger Dashboard",
             use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Decision Memo")
    memo_path = ROOT / "reports" / "decision_memo.txt"
    if memo_path.exists():
        with open(memo_path, 'r', encoding='utf-8') as f:
            st.code(f.read(), language=None)

    st.success("**Recommendation: DEPLOY CHALLENGER** — XGBoost reduces "
               "expected credit loss by $5.6M (10.19%) with statistically "
               "significant improvement (p < 0.000001)")