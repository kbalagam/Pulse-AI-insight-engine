"""
Metric computation layer.

Computes rolling statistics and week-over-week comparisons for all KPIs
in fact_daily_metrics. Returns enriched DataFrames used by the anomaly
detector and insight engine.

All rolling windows use min_periods=1 so early dates are not dropped.
"""

import datetime
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TRACKED_METRICS = ["revenue", "orders", "aov", "conversion_rate", "cac", "roas", "spend"]


def load_daily_metrics() -> pd.DataFrame:
    try:
        return pd.read_parquet(PROCESSED_DIR / "fact_daily_metrics.parquet")
    except Exception:
        df = pd.read_csv(PROCESSED_DIR / "fact_daily_metrics.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df


def load_channel_metrics() -> pd.DataFrame:
    try:
        return pd.read_parquet(PROCESSED_DIR / "fact_marketing_channel.parquet")
    except Exception:
        df = pd.read_csv(PROCESSED_DIR / "fact_marketing_channel.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df


def load_product_sales() -> pd.DataFrame:
    try:
        return pd.read_parquet(PROCESSED_DIR / "fact_product_sales.parquet")
    except Exception:
        df = pd.read_csv(PROCESSED_DIR / "fact_product_sales.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df


def compute_rolling_stats(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for metric in TRACKED_METRICS:
        if metric not in df.columns:
            continue
        roll = df[metric].rolling(window=window, min_periods=1)
        df[f"{metric}_roll_mean"] = roll.mean().round(4)
        df[f"{metric}_roll_std"]  = roll.std(ddof=0).round(4)
        df[f"{metric}_prev_day"]  = df[metric].shift(1)
    return df


def compute_wow_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for metric in TRACKED_METRICS:
        if metric not in df.columns:
            continue
        prior = df[metric].shift(7)
        df[f"{metric}_wow_pct"] = np.where(
            prior > 0,
            ((df[metric] - prior) / prior).round(4),
            np.nan,
        )
    return df


def compute_period_summary(
    df: pd.DataFrame,
    days: int = 7,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
) -> dict:
    """
    Returns a summary dict for the selected date range vs the prior equivalent period.
    If start_date and end_date are provided, uses them directly.
    Otherwise falls back to trailing `days` rows.
    """
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    if start_date is not None and end_date is not None:
        start_dt   = pd.Timestamp(start_date)
        end_dt     = pd.Timestamp(end_date)
        period_len = (end_dt - start_dt).days + 1
        prior_end  = start_dt - pd.Timedelta(days=1)
        prior_start = prior_end - pd.Timedelta(days=period_len - 1)
        recent = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        prior  = df[(df["date"] >= prior_start) & (df["date"] <= prior_end)]
    else:
        recent = df.tail(days)
        prior  = df.iloc[-(days * 2):-days]

    summary = {"period_days": len(recent)}
    for metric in TRACKED_METRICS:
        if metric not in df.columns:
            continue
        r_val = recent[metric].mean()
        p_val = prior[metric].mean()
        pct_change = ((r_val - p_val) / p_val) if p_val and p_val > 0 else None
        summary[metric] = {
            "recent_avg": round(r_val, 2) if not np.isnan(r_val) else None,
            "prior_avg":  round(p_val, 2) if not np.isnan(p_val) else None,
            "pct_change": round(pct_change, 4) if pct_change is not None else None,
        }
    return summary


def compute_top_products(
    df: pd.DataFrame,
    n: int = 5,
    days: int = 7,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
) -> pd.DataFrame:
    """Returns the top N products by revenue over the selected date range."""
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    if start_date is not None and end_date is not None:
        recent = df[
            (df["date"].dt.date >= start_date) &
            (df["date"].dt.date <= end_date)
        ]
    else:
        recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=days - 1)]

    return (
        recent.groupby(["product_id", "product_label", "category"])
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(n)
    )


def compute_channel_summary(
    df: pd.DataFrame,
    days: int = 7,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
) -> pd.DataFrame:
    """Returns per-channel performance summary for the selected date range."""
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    if start_date is not None and end_date is not None:
        recent = df[
            (df["date"].dt.date >= start_date) &
            (df["date"].dt.date <= end_date)
        ]
    else:
        recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=days - 1)]

    summary = recent.groupby("channel").agg(
        spend=("spend", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
        revenue_attributed=("revenue_attributed", "sum"),
    ).reset_index()
    summary["roas"] = np.where(
        summary["spend"] > 0,
        (summary["revenue_attributed"] / summary["spend"]).round(2),
        np.nan,
    )
    summary["conversion_rate"] = np.where(
        summary["clicks"] > 0,
        (summary["conversions"] / summary["clicks"]).round(4),
        np.nan,
    )
    return summary.sort_values("revenue_attributed", ascending=False)


def enrich_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all metric enrichments in sequence. Returns a fully enriched DataFrame."""
    df = compute_rolling_stats(df)
    df = compute_wow_change(df)
    return df
