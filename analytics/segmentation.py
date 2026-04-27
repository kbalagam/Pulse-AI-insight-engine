"""
RFM Customer Segmentation.

RFM stands for Recency, Frequency, Monetary:
    Recency   - How recently did the customer make a purchase? (days ago)
    Frequency - How many times have they purchased?
    Monetary  - How much total revenue have they generated?

Scoring:
    Each dimension is scored 1-4 using quartile binning.
    R: lower days = better = score 4 (inverted quartile)
    F: higher count = better = score 4
    M: higher revenue = better = score 4

Segments (5 standard business segments):
    Champions     R=4, F>=3         Best customers — bought recently and often
    Loyal         R>=3, F>=3        Consistent buyers, high value
    At Risk       R<=2, F>=3        Used to buy often but haven't recently
    New           R=4, F=1          Bought recently but only once
    Lost          R<=2, F<=2        Haven't bought in a long time, low frequency
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_transactions_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "transactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["refund_flag"] == 0].copy()
    df = df[df["gross_revenue"] > 0].copy()
    return df


def compute_rfm(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes R, F, M values per customer.
    Reference date = last transaction date in the dataset.
    """
    if df is None:
        df = load_transactions_raw()

    reference_date = df["timestamp"].max()

    rfm = df.groupby("customer_id").agg(
        recency=("timestamp",    lambda x: (reference_date - x.max()).days),
        frequency=("transaction_id", "count"),
        monetary=("gross_revenue",   "sum"),
    ).reset_index()

    # Remove edge cases
    rfm = rfm[rfm["monetary"] > 0].copy()

    # Score each dimension 1-4 using quartiles
    rfm["r_score"] = pd.qcut(rfm["recency"],   q=4, labels=[4,3,2,1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=4, labels=[1,2,3,4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  q=4, labels=[1,2,3,4]).astype(int)

    rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)

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
    """Returns per-segment aggregate stats for dashboard display."""
    summary = rfm.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_recency=("recency",   "mean"),
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
    """Compact context dict for Gemini prompt."""
    top_seg = summary.sort_values("total_revenue", ascending=False).iloc[0]
    at_risk = summary[summary["segment"] == "At Risk"]
    return {
        "total_customers":     len(rfm),
        "top_segment":         top_seg["segment"],
        "top_segment_revenue": round(float(top_seg["total_revenue"]), 2),
        "at_risk_count":       int(at_risk["customer_count"].values[0]) if len(at_risk) else 0,
        "champions_pct":       float(summary[summary["segment"]=="Champions"]["pct_customers"].values[0]) if len(summary[summary["segment"]=="Champions"]) else 0,
        "lost_pct":            float(summary[summary["segment"]=="Lost"]["pct_customers"].values[0]) if len(summary[summary["segment"]=="Lost"]) else 0,
    }
