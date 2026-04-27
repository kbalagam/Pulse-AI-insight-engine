"""
Cohort Retention Analysis.

Reads from two pre-processed files in data/processed/:
    cohort_input.csv   - unique (customer_id, cohort_month, txn_month) rows
    cohort_sizes.csv   - number of customers per cohort month

These files are committed to the repo so the module works on Streamlit Cloud
without raw CSV access.

A cohort is a group of customers who signed up in the same calendar month.
Retention = % of that cohort who made at least one purchase in month N.
Month 0 = signup month. Values capped at 100%.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_cohort_inputs() -> tuple:
    """Loads pre-processed cohort activity and cohort sizes."""
    try:
        activity = pd.read_parquet(PROCESSED_DIR / "cohort_input.parquet")
        sizes    = pd.read_parquet(PROCESSED_DIR / "cohort_sizes.parquet")
    except Exception:
        activity = pd.read_csv(PROCESSED_DIR / "cohort_input.csv")
        sizes    = pd.read_csv(PROCESSED_DIR / "cohort_sizes.csv")
    return activity, sizes


def build_cohort_matrix() -> tuple:
    """
    Builds the cohort retention matrix.

    Returns
    -------
    retention_pct : pd.DataFrame
        Pivot of retention rates (0.0-1.0), cohort month x months since signup.
        Values capped at 1.0.
    cohort_sizes : pd.Series
        Number of customers per cohort.
    """
    activity, sizes_df = load_cohort_inputs()

    cohort_sizes = sizes_df.set_index("cohort_month")["cohort_size"]

    # Convert period strings to Period objects for arithmetic
    activity["cohort_month_p"] = pd.PeriodIndex(activity["cohort_month"], freq="M")
    activity["txn_month_p"]    = pd.PeriodIndex(activity["txn_month"],    freq="M")

    activity["months_since_signup"] = (
        activity["txn_month_p"] - activity["cohort_month_p"]
    ).apply(lambda x: x.n if hasattr(x, "n") else np.nan)

    activity = activity[activity["months_since_signup"] >= 0].copy()

    cohort_activity = (
        activity.groupby(["cohort_month","months_since_signup"])["customer_id"]
        .nunique().reset_index(name="active_customers")
    )

    pivot = cohort_activity.pivot_table(
        index="cohort_month", columns="months_since_signup",
        values="active_customers", aggfunc="sum",
    )

    retention_pct = pivot.divide(cohort_sizes, axis=0).clip(upper=1.0)

    max_offset = min(23, int(retention_pct.columns.max()))
    retention_pct = retention_pct[
        [c for c in retention_pct.columns if c <= max_offset]
    ]

    retention_pct.index = retention_pct.index.astype(str)
    cohort_sizes.index  = cohort_sizes.index.astype(str)

    return retention_pct, cohort_sizes


def cohort_summary_for_ai(retention_pct: pd.DataFrame,
                           cohort_sizes: pd.Series) -> dict:
    m1 = retention_pct[1].dropna() if 1 in retention_pct.columns else pd.Series([])
    m3 = retention_pct[3].dropna() if 3 in retention_pct.columns else pd.Series([])
    m6 = retention_pct[6].dropna() if 6 in retention_pct.columns else pd.Series([])
    return {
        "total_cohorts":        len(retention_pct),
        "avg_month1_retention": round(float(m1.mean()*100), 1) if len(m1) else None,
        "avg_month3_retention": round(float(m3.mean()*100), 1) if len(m3) else None,
        "avg_month6_retention": round(float(m6.mean()*100), 1) if len(m6) else None,
        "best_month1_cohort":   str(m1.idxmax()) if len(m1) else "N/A",
        "worst_month1_cohort":  str(m1.idxmin()) if len(m1) else "N/A",
        "avg_cohort_size":      round(float(cohort_sizes.mean()), 0),
    }
