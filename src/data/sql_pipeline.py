"""
sql_pipeline.py
---------------
Builds the full feature table in SQL using window functions,
CTEs, and business-logic transformations.

Output: features.loan_features — one row per loan, model-ready

Feature groups built here:
  1. Raw numeric features (cleaned)
  2. Derived ratio features
  3. Binned features for WoE encoding
  4. Cohort / vintage features
  5. Train / test / OOT split flags
"""

import duckdb
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "credit_risk.duckdb"


def create_feature_schema(conn):
    conn.execute("CREATE SCHEMA IF NOT EXISTS features")
    log.info("Schema: features")


def build_derived_features(conn):
    """
    CTE-based derived feature engineering in pure SQL.
    Every feature here has a business justification.
    """
    log.info("Building features.loan_features ...")

    conn.execute("""
        CREATE OR REPLACE TABLE features.loan_features AS

        WITH base AS (
            SELECT * FROM staging.loans_clean
        ),

        derived AS (
            SELECT
                *,

                -- 1. DEBT BURDEN FEATURES
                -- monthly debt obligation vs income
                ROUND(installment / NULLIF(annual_inc / 12, 0), 4)
                                                    AS installment_to_income,

                -- loan amount relative to income
                ROUND(loan_amnt / NULLIF(annual_inc, 0), 4)
                                                    AS loan_to_income,

                -- revolving balance relative to income
                ROUND(revol_bal / NULLIF(annual_inc, 0), 4)
                                                    AS revol_bal_to_income,

                -- 2. CREDIT HISTORY FEATURES
                -- average account age proxy: open_acc relative to total_acc
                ROUND(open_acc / NULLIF(total_acc, 0), 4)
                                                    AS open_to_total_acc_ratio,

                -- derogatory mark flag (any serious negative event)
                CASE WHEN pub_rec > 0
                          OR delinq_2yrs > 0
                          OR collections_12_mths_ex_med > 0
                     THEN 1 ELSE 0 END              AS has_derogatory,

                -- 3. FICO FEATURES
                -- FICO band (used for WoE binning)
                CASE
                    WHEN fico_avg < 620  THEN '1_below_620'
                    WHEN fico_avg < 660  THEN '2_620_659'
                    WHEN fico_avg < 700  THEN '3_660_699'
                    WHEN fico_avg < 740  THEN '4_700_739'
                    WHEN fico_avg < 780  THEN '5_740_779'
                    ELSE                      '6_780_plus'
                END                                 AS fico_band,

                -- 4. DTI BANDS
                CASE
                    WHEN dti < 10  THEN '1_low'
                    WHEN dti < 20  THEN '2_medium'
                    WHEN dti < 30  THEN '3_high'
                    ELSE                '4_very_high'
                END                                 AS dti_band,

                -- 5. LOAN AMOUNT BANDS
                CASE
                    WHEN loan_amnt < 5000   THEN '1_micro'
                    WHEN loan_amnt < 10000  THEN '2_small'
                    WHEN loan_amnt < 20000  THEN '3_medium'
                    WHEN loan_amnt < 30000  THEN '4_large'
                    ELSE                         '5_xlarge'
                END                                 AS loan_amnt_band,

                -- 6. EMPLOYMENT STABILITY
                CASE
                    WHEN emp_length_yrs < 2  THEN '1_junior'
                    WHEN emp_length_yrs < 5  THEN '2_mid'
                    WHEN emp_length_yrs < 8  THEN '3_senior'
                    ELSE                          '4_veteran'
                END                                 AS emp_stability,

                -- 7. INTEREST RATE BAND (proxy for risk tier)
                CASE
                    WHEN int_rate < 8   THEN '1_prime'
                    WHEN int_rate < 12  THEN '2_near_prime'
                    WHEN int_rate < 16  THEN '3_subprime'
                    WHEN int_rate < 22  THEN '4_deep_subprime'
                    ELSE                     '5_distressed'
                END                                 AS int_rate_band,

                -- 8. VINTAGE FEATURE (for time-based splits)
                CONCAT(
                    CAST(vintage_year AS VARCHAR), '-Q',
                    CAST(vintage_quarter AS VARCHAR)
                )                                   AS vintage

            FROM base
        ),

        -- 9. TRAIN / VALIDATION / OUT-OF-TIME SPLIT FLAGS
        -- Train:      2007–2015 (in-time sample)
        -- Validation: 2016      (holdout)
        -- OOT:        2017–2018 (out-of-time, simulates production)
        split_flags AS (
            SELECT
                *,
                CASE
                    WHEN vintage_year <= 2015 THEN 'train'
                    WHEN vintage_year  = 2016 THEN 'validation'
                    ELSE                           'oot'
                END                                 AS split

            FROM derived
        )

        SELECT * FROM split_flags
    """)

    count   = conn.execute(
        "SELECT COUNT(*) FROM features.loan_features").fetchone()[0]
    splits  = conn.execute("""
        SELECT split, COUNT(*) as n,
               ROUND(AVG(is_bad)*100,2) as bad_rate
        FROM features.loan_features
        GROUP BY split ORDER BY split
    """).df()

    log.info(f"features.loan_features — {count:,} rows")
    log.info(f"\nSplit summary:\n{splits.to_string(index=False)}")


def build_woe_table(conn):
    """
    Computes Weight of Evidence (WoE) and Information Value (IV)
    for every binned categorical and band feature — all in SQL.

    WoE = ln(Distribution of Events / Distribution of Non-Events)
    IV  = SUM((Dist_Events - Dist_NonEvents) * WoE)

    IV interpretation:
      < 0.02  : useless
      0.02–0.1: weak
      0.1–0.3 : medium
      > 0.3   : strong
    """
    log.info("Computing WoE and IV for all binned features ...")

    bin_features = [
        'grade', 'fico_band', 'dti_band', 'loan_amnt_band',
        'emp_stability', 'int_rate_band', 'home_ownership',
        'verification_status', 'purpose', 'application_type'
    ]

    woe_rows = []

    total_events     = conn.execute(
        "SELECT SUM(is_bad) FROM features.loan_features "
        "WHERE split = 'train'").fetchone()[0]
    total_non_events = conn.execute(
        "SELECT SUM(1 - is_bad) FROM features.loan_features "
        "WHERE split = 'train'").fetchone()[0]

    for feat in bin_features:
        result = conn.execute(f"""
            SELECT
                '{feat}'                                    AS feature,
                {feat}                                      AS bin,
                COUNT(*)                                    AS total,
                SUM(is_bad)                                 AS events,
                SUM(1 - is_bad)                             AS non_events,
                ROUND(SUM(is_bad) * 100.0 / COUNT(*), 2)   AS bad_rate_pct
            FROM features.loan_features
            WHERE split = 'train'
            GROUP BY {feat}
            ORDER BY {feat}
        """).df()

        result['dist_events']     = result['events']     / total_events
        result['dist_non_events'] = result['non_events'] / total_non_events
        result['woe'] = np.log(
            result['dist_events'] / result['dist_non_events']
        ).replace([np.inf, -np.inf], 0).round(4)
        result['iv_component'] = (
            (result['dist_events'] - result['dist_non_events']) * result['woe']
        ).round(6)

        woe_rows.append(result)


    woe_df = pd.concat(woe_rows, ignore_index=True)

    conn.execute("""
        CREATE OR REPLACE TABLE features.woe_table AS
        SELECT * FROM woe_df
    """)

    # IV summary per feature
    iv_summary = (woe_df.groupby('feature')['iv_component']
                        .sum()
                        .reset_index()
                        .rename(columns={'iv_component': 'iv'})
                        .sort_values('iv', ascending=False)
                        .round(4))

    conn.execute("""
        CREATE OR REPLACE TABLE features.iv_summary AS
        SELECT * FROM iv_summary
    """)

    log.info(f"\nInformation Value Summary:\n{iv_summary.to_string(index=False)}")
    return woe_df, iv_summary


def apply_woe_encoding(conn):
    """
    Joins WoE values back onto the feature table.
    Creates features.model_input — the final model-ready table.
    """
    log.info("Applying WoE encoding → features.model_input ...")

    conn.execute("""
        CREATE OR REPLACE TABLE features.model_input AS
        SELECT
            f.id,
            f.is_bad,
            f.split,
            f.vintage_year,
            f.vintage,

            -- raw numeric features (already clean)
            f.loan_amnt,
            f.term_months,
            f.int_rate,
            f.annual_inc,
            f.dti,
            f.fico_avg,
            f.revol_util,
            f.open_acc,
            f.total_acc,
            f.delinq_2yrs,
            f.inq_last_6mths,
            f.pub_rec,
            f.emp_length_yrs,
            f.installment_to_income,
            f.loan_to_income,
            f.revol_bal_to_income,
            f.open_to_total_acc_ratio,
            f.has_derogatory,

            -- WoE encoded features
            g.woe     AS grade_woe,
            fb.woe    AS fico_band_woe,
            db.woe    AS dti_band_woe,
            lb.woe    AS loan_amnt_band_woe,
            es.woe    AS emp_stability_woe,
            ir.woe    AS int_rate_band_woe,
            ho.woe    AS home_ownership_woe,
            vs.woe    AS verification_status_woe,
            pu.woe    AS purpose_woe,
            atype.woe    AS application_type_woe

        FROM features.loan_features f

        LEFT JOIN features.woe_table g
            ON g.feature = 'grade'               AND g.bin = f.grade
        LEFT JOIN features.woe_table fb
            ON fb.feature = 'fico_band'          AND fb.bin = f.fico_band
        LEFT JOIN features.woe_table db
            ON db.feature = 'dti_band'           AND db.bin = f.dti_band
        LEFT JOIN features.woe_table lb
            ON lb.feature = 'loan_amnt_band'     AND lb.bin = f.loan_amnt_band
        LEFT JOIN features.woe_table es
            ON es.feature = 'emp_stability'      AND es.bin = f.emp_stability
        LEFT JOIN features.woe_table ir
            ON ir.feature = 'int_rate_band'      AND ir.bin = f.int_rate_band
        LEFT JOIN features.woe_table ho
            ON ho.feature = 'home_ownership'     AND ho.bin = f.home_ownership
        LEFT JOIN features.woe_table vs
            ON vs.feature = 'verification_status' AND vs.bin = f.verification_status
        LEFT JOIN features.woe_table pu
            ON pu.feature = 'purpose'            AND pu.bin = f.purpose
        LEFT JOIN features.woe_table atype
            ON atype.feature = 'application_type'  AND atype.bin = f.application_type
    """)

    count = conn.execute(
        "SELECT COUNT(*) FROM features.model_input").fetchone()[0]
    log.info(f"features.model_input — {count:,} rows — ready for modelling")


def main():
    conn = duckdb.connect(str(DB_PATH))
    create_feature_schema(conn)
    build_derived_features(conn)
    build_woe_table(conn)
    apply_woe_encoding(conn)
    conn.close()
    log.info("Feature pipeline complete.")


if __name__ == "__main__":
    main()