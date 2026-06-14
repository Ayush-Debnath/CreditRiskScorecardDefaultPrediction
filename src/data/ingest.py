"""
ingest.py
---------
Loads raw Lending Club CSVs into a DuckDB database.
Creates three schemas:
  - raw    : untouched data as-is from Kaggle
  - staging: light type casting, column renaming
  - profile: summary statistics for data quality reporting
"""

import duckdb
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
RAW_DIR     = ROOT / "data" / "raw"
PROCESSED   = ROOT / "data" / "processed"
DB_PATH     = PROCESSED / "credit_risk.duckdb"

ACCEPTED_CSV = RAW_DIR / "accepted_2007_to_2018Q4.csv"
REJECTED_CSV = RAW_DIR / "rejected_2007_to_2018Q4.csv"

PROCESSED.mkdir(parents=True, exist_ok=True)


# ── columns we actually need (keeps memory manageable) ─────────────────────────
ACCEPTED_COLS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "purpose", "title",
    "addr_state", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "initial_list_status", "out_prncp", "total_pymnt",
    "total_rec_prncp", "total_rec_int", "last_pymnt_d", "last_fico_range_high",
    "last_fico_range_low", "collections_12_mths_ex_med", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "application_type"
]

REJECTED_COLS = [
    "Amount Requested", "Application Date", "Loan Title",
    "Risk_Score", "Debt-To-Income Ratio", "Zip Code",
    "State", "Employment Length", "Policy Code"
]


def connect() -> duckdb.DuckDBPyConnection:
    """Return a persistent DuckDB connection."""
    conn = duckdb.connect(str(DB_PATH))
    log.info(f"Connected to DuckDB at {DB_PATH}")
    return conn


def create_schemas(conn: duckdb.DuckDBPyConnection) -> None:
    """Create raw, staging, and profile schemas."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute("CREATE SCHEMA IF NOT EXISTS staging")
    conn.execute("CREATE SCHEMA IF NOT EXISTS profile")
    log.info("Schemas created: raw, staging, profile")


def load_accepted(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Load accepted loans CSV → raw.accepted_loans.
    Reads only the columns we need to keep memory low.
    Skips the first row (LendingClub adds a descriptor row).
    """
    log.info("Loading accepted loans — this takes ~2 minutes for 2.2M rows ...")

    col_list = ", ".join([f'"{c}"' for c in ACCEPTED_COLS])

    conn.execute(f"""
        CREATE OR REPLACE TABLE raw.accepted_loans AS
        SELECT {col_list}
        FROM read_csv_auto(
            '{ACCEPTED_CSV.as_posix()}',
            header      = true,
            ignore_errors = true,
            sample_size = 5000
        )
        WHERE id IS NOT NULL          -- drops the descriptor rows LC adds
          AND loan_status IS NOT NULL
    """)

    count = conn.execute("SELECT COUNT(*) FROM raw.accepted_loans").fetchone()[0]
    log.info(f"raw.accepted_loans loaded — {count:,} rows")


def load_rejected(conn: duckdb.DuckDBPyConnection) -> None:
    """Load rejected loans CSV → raw.rejected_loans."""
    log.info("Loading rejected loans — this takes ~1 minute ...")

    col_list = ", ".join([f'"{c}"' for c in REJECTED_COLS])

    conn.execute(f"""
        CREATE OR REPLACE TABLE raw.rejected_loans AS
        SELECT {col_list}
        FROM read_csv_auto(
            '{REJECTED_CSV.as_posix()}',
            header        = true,
            ignore_errors = true,
            sample_size   = 5000
        )
        WHERE "Amount Requested" IS NOT NULL
    """)

    count = conn.execute("SELECT COUNT(*) FROM raw.rejected_loans").fetchone()[0]
    log.info(f"raw.rejected_loans loaded — {count:,} rows")


def build_staging(conn: duckdb.DuckDBPyConnection) -> None:
    """
    staging.loans
    -------------
    - Cleans column names (lowercase, underscores)
    - Casts types (int_rate % string → float, term → int, issue_d → date)
    - Creates binary target: is_bad (1 = defaulted/charged-off, 0 = fully paid)
    - Adds vintage_year and vintage_quarter for cohort analysis
    """
    log.info("Building staging.loans ...")

    conn.execute("""
        CREATE OR REPLACE TABLE staging.loans AS
        SELECT
            -- identifiers
            id,

            -- loan attributes
            loan_amnt,
            funded_amnt,
            CAST(REPLACE(term, ' months', '') AS INTEGER)           AS term_months,
            CAST(REPLACE(CAST(int_rate AS VARCHAR), '%', '') AS FLOAT) AS int_rate,
            installment,
            grade,
            sub_grade,

            -- borrower attributes
            CASE emp_length
                WHEN '< 1 year'  THEN 0
                WHEN '1 year'    THEN 1
                WHEN '2 years'   THEN 2
                WHEN '3 years'   THEN 3
                WHEN '4 years'   THEN 4
                WHEN '5 years'   THEN 5
                WHEN '6 years'   THEN 6
                WHEN '7 years'   THEN 7
                WHEN '8 years'   THEN 8
                WHEN '9 years'   THEN 9
                WHEN '10+ years' THEN 10
                ELSE NULL
            END                                                      AS emp_length_yrs,
            home_ownership,
            annual_inc,
            verification_status,
            purpose,
            addr_state,

            -- credit attributes
            dti,
            delinq_2yrs,
            fico_range_low,
            fico_range_high,
            ROUND((fico_range_low + fico_range_high) / 2.0, 0)     AS fico_avg,
            inq_last_6mths,
            open_acc,
            pub_rec,
            revol_bal,
            CAST(REPLACE(CAST(revol_util AS VARCHAR), '%', '') AS FLOAT) AS revol_util,
            total_acc,
            collections_12_mths_ex_med,
            acc_now_delinq,
            tot_coll_amt,
            tot_cur_bal,

            -- dates & vintage
            STRPTIME(issue_d, '%b-%Y')                              AS issue_date,
            YEAR(STRPTIME(issue_d, '%b-%Y'))                        AS vintage_year,
            QUARTER(STRPTIME(issue_d, '%b-%Y'))                     AS vintage_quarter,

            -- application type
            application_type,
            loan_status,

            -- TARGET VARIABLE
            -- 1 = bad loan (defaulted / charged off / very late)
            -- 0 = good loan (fully paid)
            -- Loans still current are EXCLUDED — outcome unknown
            CASE
                WHEN loan_status IN (
                    'Charged Off',
                    'Default',
                    'Does not meet the credit policy. Status:Charged Off',
                    'Late (31-120 days)'
                ) THEN 1
                WHEN loan_status IN (
                    'Fully Paid',
                    'Does not meet the credit policy. Status:Fully Paid'
                ) THEN 0
                ELSE NULL   -- Current, In Grace Period, Late 16-30 days → excluded
            END                                                      AS is_bad

        FROM raw.accepted_loans
        WHERE loan_status IN (
            'Charged Off', 'Default', 'Fully Paid',
            'Does not meet the credit policy. Status:Charged Off',
            'Does not meet the credit policy. Status:Fully Paid',
            'Late (31-120 days)'
        )
    """)

    count = conn.execute("SELECT COUNT(*) FROM staging.loans").fetchone()[0]
    bad   = conn.execute("SELECT AVG(is_bad) FROM staging.loans").fetchone()[0]
    log.info(f"staging.loans built — {count:,} rows | bad rate: {bad:.2%}")


def build_data_profile(conn: duckdb.DuckDBPyConnection) -> None:
    """
    profile.column_stats
    --------------------
    For every column in staging.loans, compute:
    null_pct, distinct_count, mean, std, min, max
    This is your data quality report — readable in SQL or exported to CSV.
    """
    log.info("Building data profile ...")

    numeric_cols = [
        "loan_amnt", "funded_amnt", "term_months", "int_rate", "installment",
        "emp_length_yrs", "annual_inc", "dti", "delinq_2yrs", "fico_avg",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
        "total_acc", "collections_12_mths_ex_med", "acc_now_delinq",
        "tot_coll_amt", "tot_cur_bal"
    ]

    rows = []
    total = conn.execute("SELECT COUNT(*) FROM staging.loans").fetchone()[0]

    for col in numeric_cols:
        result = conn.execute(f"""
            SELECT
                COUNT(*)                                    AS total_rows,
                SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
                COUNT(DISTINCT {col})                       AS distinct_count,
                ROUND(AVG({col}), 4)                        AS mean,
                ROUND(STDDEV({col}), 4)                     AS std_dev,
                ROUND(MIN({col}), 4)                        AS min_val,
                ROUND(MAX({col}), 4)                        AS max_val,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}), 4) AS p25,
                ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {col}), 4) AS p50,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}), 4) AS p75
            FROM staging.loans
        """).fetchone()

        rows.append({
            "column"         : col,
            "total_rows"     : result[0],
            "null_count"     : result[1],
            "null_pct"       : round(result[1] / total * 100, 2),
            "distinct_count" : result[2],
            "mean"           : result[3],
            "std_dev"        : result[4],
            "min"            : result[5],
            "max"            : result[6],
            "p25"            : result[7],
            "p50"            : result[8],
            "p75"            : result[9],
        })

    profile_df = pd.DataFrame(rows)

    conn.execute("""
        CREATE OR REPLACE TABLE profile.column_stats AS
        SELECT * FROM profile_df
    """)

    # also save as CSV for easy sharing
    profile_path = PROCESSED / "data_profile.csv"
    profile_df.to_csv(profile_path, index=False)
    log.info(f"Data profile saved → {profile_path}")


def run_validation_queries(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Quick sanity checks printed to terminal.
    These are the first SQL queries you can walk an interviewer through.
    """
    log.info("─" * 60)
    log.info("VALIDATION QUERIES")
    log.info("─" * 60)

    # 1. bad rate by grade
    log.info("\n[1] Bad rate by loan grade:")
    result = conn.execute("""
        SELECT
            grade,
            COUNT(*)                        AS total_loans,
            SUM(is_bad)                     AS bad_loans,
            ROUND(AVG(is_bad) * 100, 2)     AS bad_rate_pct,
            ROUND(AVG(int_rate), 2)         AS avg_interest_rate
        FROM staging.loans
        GROUP BY grade
        ORDER BY grade
    """).df()
    print(result.to_string(index=False))

    # 2. loan volume by vintage year
    log.info("\n[2] Loan volume and bad rate by vintage year:")
    result = conn.execute("""
        SELECT
            vintage_year,
            COUNT(*)                        AS total_loans,
            ROUND(AVG(is_bad) * 100, 2)     AS bad_rate_pct,
            ROUND(AVG(loan_amnt), 0)        AS avg_loan_amount
        FROM staging.loans
        GROUP BY vintage_year
        ORDER BY vintage_year
    """).df()
    print(result.to_string(index=False))

    # 3. target distribution
    log.info("\n[3] Target variable distribution:")
    result = conn.execute("""
        SELECT
            is_bad,
            COUNT(*)                            AS count,
            ROUND(COUNT(*) * 100.0 /
                SUM(COUNT(*)) OVER (), 2)       AS pct
        FROM staging.loans
        GROUP BY is_bad
        ORDER BY is_bad
    """).df()
    print(result.to_string(index=False))

    # 4. null summary for key features
    log.info("\n[4] Null % in key features:")
    result = conn.execute("""
        SELECT * FROM profile.column_stats
        ORDER BY null_pct DESC
        LIMIT 10
    """).df()
    print(result[["column","null_count","null_pct","mean","min","max"]].to_string(index=False))


def main():
    conn = connect()
    create_schemas(conn)
    load_accepted(conn)
    load_rejected(conn)
    build_staging(conn)
    build_data_profile(conn)
    run_validation_queries(conn)
    conn.close()
    log.info("Ingest complete. DB saved at data/processed/credit_risk.duckdb")


if __name__ == "__main__":
    main()