"""
Cohort Retention Analysis.

A cohort is a group of customers who signed up in the same calendar month.

Retention is measured as: what % of each cohort made at least one purchase
in month N after their signup month?

Month 0 = signup month. Month 1 = one month later, etc.

Note on values > 100%:
    In simulated data, some customers transact more in month 1 than month 0
    because they signed up late in the month (fewer days available in month 0).
    All values are capped at 100% for correct interpretation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_cohort_data() -> tuple:
    customers = pd.read_csv(RAW_DIR / "customers.csv")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    txn = pd.read_csv(RAW_DIR / "transactions.csv")
    txn["timestamp"] = pd.to_datetime(txn["timestamp"])
    txn = txn[txn["refund_flag"] == 0].copy()
    return customers, txn


def build_cohort_matrix() -> tuple:
    """
    Builds the cohort retention matrix.

    Returns
    -------
    retention_pct : pd.DataFrame
        Pivot of retention rates (0.0–1.0), cohort month x months since signup.
        Values are capped at 1.0 (100%).
    cohort_sizes : pd.Series
        Number of unique customers in each cohort.
    """
    customers, txn = load_cohort_data()

    customers["cohort_month"] = customers["signup_date"].dt.to_period("M")

    merged = txn.merge(
        customers[["customer_id","cohort_month"]],
        on="customer_id", how="left",
    )
    merged = merged.dropna(subset=["cohort_month"])
    merged["txn_month"] = merged["timestamp"].dt.to_period("M")
    merged["months_since_signup"] = (
        merged["txn_month"] - merged["cohort_month"]
    ).apply(lambda x: x.n if hasattr(x,"n") else np.nan)
    merged = merged[merged["months_since_signup"] >= 0].copy()

    cohort_activity = (
        merged.groupby(["cohort_month","months_since_signup"])["customer_id"]
        .nunique().reset_index(name="active_customers")
    )

    # Cohort size = customers who signed up that month (from customers table)
    cohort_sizes = (
        customers.groupby("cohort_month")["customer_id"]
        .count()
    )

    pivot = cohort_activity.pivot_table(
        index="cohort_month", columns="months_since_signup",
        values="active_customers", aggfunc="sum",
    )

    # Retention % = active / cohort_size — capped at 1.0
    retention_pct = pivot.divide(cohort_sizes, axis=0).clip(upper=1.0)

    # Keep 0–23 months (2 years) for readability
    max_offset = min(23, int(retention_pct.columns.max()))
    retention_pct = retention_pct[
        [c for c in retention_pct.columns if c <= max_offset]
    ]

    retention_pct.index = retention_pct.index.astype(str)
    cohort_sizes.index  = cohort_sizes.index.astype(str)

    return retention_pct, cohort_sizes


def cohort_summary_for_ai(retention_pct: pd.DataFrame, cohort_sizes: pd.Series) -> dict:
    m1 = retention_pct[1].dropna() if 1 in retention_pct.columns else pd.Series([])
    m3 = retention_pct[3].dropna() if 3 in retention_pct.columns else pd.Series([])
    m6 = retention_pct[6].dropna() if 6 in retention_pct.columns else pd.Series([])
    return {
        "total_cohorts":        len(retention_pct),
        "avg_month1_retention": round(float(m1.mean()*100),1) if len(m1) else None,
        "avg_month3_retention": round(float(m3.mean()*100),1) if len(m3) else None,
        "avg_month6_retention": round(float(m6.mean()*100),1) if len(m6) else None,
        "best_month1_cohort":   str(m1.idxmax()) if len(m1) else "N/A",
        "worst_month1_cohort":  str(m1.idxmin()) if len(m1) else "N/A",
        "avg_cohort_size":      round(float(cohort_sizes.mean()),0),
    }
