"""
clean.py
--------
Applies business-rule cleaning on staging.loans:
- Caps outliers using domain knowledge (not just statistics)
- Imputes nulls with business logic
- Removes leakage columns
- Saves clean table back to DuckDB as staging.loans_clean
"""

import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "credit_risk.duckdb"


def clean(conn: duckdb.DuckDBPyConnection) -> None:
    log.info("Building staging.loans_clean ...")

    conn.execute("""
        CREATE OR REPLACE TABLE staging.loans_clean AS
        SELECT
            id,
            is_bad,
            vintage_year,
            vintage_quarter,
            issue_date,
            grade,
            sub_grade,

            -- loan features
            loan_amnt,
            term_months,
            ROUND(int_rate, 2)                              AS int_rate,
            installment,

            -- borrower features
            -- emp_length: impute nulls as median (5 yrs) — business rule
            COALESCE(emp_length_yrs, 5)                     AS emp_length_yrs,
            home_ownership,
            -- cap annual_inc at 99th percentile (~300K)
            LEAST(annual_inc, 300000)                       AS annual_inc,
            verification_status,
            purpose,
            application_type,

            -- credit bureau features
            -- dti: cap at 60 (anything above is data error or extreme)
            CASE WHEN dti < 0 OR dti > 60 THEN NULL
                 ELSE ROUND(dti, 2) END                     AS dti,
            delinq_2yrs,
            fico_avg,
            inq_last_6mths,
            open_acc,
            pub_rec,
            -- revol_util: cap at 100% (892% is a data error)
            LEAST(COALESCE(revol_util, 51.77), 100)         AS revol_util,
            total_acc,
            COALESCE(collections_12_mths_ex_med, 0)         AS collections_12_mths_ex_med,
            COALESCE(acc_now_delinq, 0)                     AS acc_now_delinq,
            -- tot_coll_amt / tot_cur_bal: impute nulls as 0
            revol_bal,
            COALESCE(tot_coll_amt, 0)                       AS tot_coll_amt,
            COALESCE(tot_cur_bal, 0)                        AS tot_cur_bal

        FROM staging.loans
        WHERE
            annual_inc  > 0
            AND loan_amnt > 0
            AND dti IS NOT NULL
            AND fico_avg BETWEEN 580 AND 850
    """)

    count = conn.execute(
        "SELECT COUNT(*) FROM staging.loans_clean").fetchone()[0]
    log.info(f"staging.loans_clean — {count:,} rows")


def main():
    conn = duckdb.connect(str(DB_PATH))
    clean(conn)
    conn.close()
    log.info("Cleaning complete.")


if __name__ == "__main__":
    main()