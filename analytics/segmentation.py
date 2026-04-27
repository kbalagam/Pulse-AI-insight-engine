"""
RFM Customer Segmentation.

Reads from data/processed/rfm_input.csv — a pre-aggregated file containing
one row per customer with recency, frequency, and monetary values already
computed from raw transactions. This file is committed to the repo so the
module works on Streamlit Cloud without raw CSV access.

RFM stands for Recency, Frequency, Monetary:
    Recency   - Days since last purchase (lower = more recent = better)
    Frequency - Number of purchases (higher = better)
    Monetary  - Total revenue generated (higher = better)

Scoring:
    Each dimension scored 1-4 via quartile binning.
    R: inverted — fewer days = score 4
    F: higher count = score 4
    M: higher revenue = score 4

Segments (5 standard business segments):
    Champions  R=4, F>=3   Bought recently and often — best customers
    Loyal      R>=3, F>=3  Consistent buyers with high value
    At Risk    R<=2, F>=3  Used to buy often but haven't recently
    New        R=4, F=1    First-time buyers — recently acquired
    Lost       R<=2, F<=2  Low recency and frequency — likely churned
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_rfm_input() -> pd.DataFrame:
    """Loads pre-aggregated RFM input from processed directory."""
    try:
        return pd.read_parquet(PROCESSED_DIR / "rfm_input.parquet")
    except Exception:
        return pd.read_csv(PROCESSED_DIR / "rfm_input.csv")


def compute_rfm() -> pd.DataFrame:
    """
    Scores and segments customers from the pre-aggregated RFM input.
    Returns one row per customer with R/F/M scores and segment label.
    """
    rfm = load_rfm_input().copy()
    rfm = rfm[rfm["monetary"] > 0].copy()

    rfm["r_score"] = pd.qcut(
        rfm["recency"], q=4, labels=[4,3,2,1]
    ).astype(int)
    rfm["f_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), q=4, labels=[1,2,3,4]
    ).astype(int)
    rfm["m_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"), q=4, labels=[1,2,3,4]
    ).astype(int)

    rfm["segment"] = rfm.apply(_assign_segment, axis=1)
    return rfm


def _assign_segment(row) -> str:
    r, f = row["r_score"], row["f_score"]
    if r == 4 and f >= 3:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r == 4 and f == 1:
        return "New"
    else:
        return "Lost"


SEGMENT_META = {
    "Champions": {
        "icon":  "🏆",
        "color": "#9B1C1C",
        "desc":  "Bought recently, buy often, spend the most. Reward and retain.",
    },
    "Loyal": {
        "icon":  "⭐",
        "color": "#374151",
        "desc":  "Consistent buyers with good value. Upsell and cross-sell.",
    },
    "At Risk": {
        "icon":  "⚠️",
        "color": "#D97706",
        "desc":  "Used to buy often but haven't recently. Re-engagement campaigns needed.",
    },
    "New": {
        "icon":  "🌱",
        "color": "#16A34A",
        "desc":  "Bought recently but only once. Nurture with onboarding offers.",
    },
    "Lost": {
        "icon":  "💤",
        "color": "#6B7280",
        "desc":  "Low recency and frequency. Consider win-back campaigns or accept churn.",
    },
}


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = rfm.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_recency=("recency",    "mean"),
        avg_frequency=("frequency","mean"),
        avg_monetary=("monetary",  "mean"),
        total_revenue=("monetary", "sum"),
    ).reset_index()
    summary["avg_recency"]   = summary["avg_recency"].round(0).astype(int)
    summary["avg_frequency"] = summary["avg_frequency"].round(1)
    summary["avg_monetary"]  = summary["avg_monetary"].round(2)
    summary["total_revenue"] = summary["total_revenue"].round(2)
    summary["pct_customers"] = (
        summary["customer_count"] / summary["customer_count"].sum() * 100
    ).round(1)
    return summary


def rfm_summary_for_ai(rfm: pd.DataFrame, summary: pd.DataFrame) -> dict:
    top = summary.sort_values("total_revenue", ascending=False).iloc[0]
    at_risk = summary[summary["segment"] == "At Risk"]
    champ   = summary[summary["segment"] == "Champions"]
    lost    = summary[summary["segment"] == "Lost"]
    return {
        "total_customers":     len(rfm),
        "top_segment":         top["segment"],
        "top_segment_revenue": round(float(top["total_revenue"]), 2),
        "at_risk_count":       int(at_risk["customer_count"].values[0]) if len(at_risk) else 0,
        "champions_pct":       float(champ["pct_customers"].values[0]) if len(champ) else 0,
        "lost_pct":            float(lost["pct_customers"].values[0]) if len(lost) else 0,
    }
