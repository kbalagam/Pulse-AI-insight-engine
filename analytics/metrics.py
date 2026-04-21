"""
Metric computation layer.

Computes rolling statistics and week-over-week comparisons for all KPIs
in fact_daily_metrics. Returns enriched DataFrames used by the anomaly
detector and insight engine.

All rolling windows use min_periods=1 so early dates are not dropped.
"""

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
    """
    Adds rolling mean, rolling std, and prior-day value for each tracked metric.
    Columns added: {metric}_roll_mean, {metric}_roll_std, {metric}_prev_day
    """
    df = df.sort_values("date").copy()
    for metric in TRACKED_METRICS:
        if metric not in df.columns:
            continue
        roll = df[metric].rolling(window=window, min_periods=1)
        df[f"{metric}_roll_mean"] = roll.mean().round(4)
        df[f"{metric}_roll_std"] = roll.std(ddof=0).round(4)
        df[f"{metric}_prev_day"] = df[metric].shift(1)
    return df


def compute_wow_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds week-over-week percentage change for each tracked metric.
    Column added: {metric}_wow_pct  (e.g. 0.12 = +12%)
    """
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


def compute_period_summary(df: pd.DataFrame, days: int = 7) -> dict:
    """
    Returns a summary dict for the most recent N days vs the prior N days.
    Used by the AI layer to give context for narrative generation.
    """
    df = df.sort_values("date")
    recent = df.tail(days)
    prior = df.iloc[-(days * 2):-days]

    summary = {"period_days": days}
    for metric in TRACKED_METRICS:
        if metric not in df.columns:
            continue
        r_val = recent[metric].mean()
        p_val = prior[metric].mean()
        pct_change = ((r_val - p_val) / p_val) if p_val and p_val > 0 else None
        summary[metric] = {
            "recent_avg": round(r_val, 2) if not np.isnan(r_val) else None,
            "prior_avg": round(p_val, 2) if not np.isnan(p_val) else None,
            "pct_change": round(pct_change, 4) if pct_change is not None else None,
        }
    return summary


def compute_top_products(df: pd.DataFrame, n: int = 5, days: int = 7) -> pd.DataFrame:
    """Returns the top N products by revenue over the last N days."""
    df = df.sort_values("date")
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=days - 1)]
    return (
        recent.groupby(["product_id", "product_label", "category"])
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(n)
    )


def compute_channel_summary(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Returns per-channel performance summary for the last N days."""
    df = df.sort_values("date")
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
