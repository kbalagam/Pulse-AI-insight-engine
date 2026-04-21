"""
Anomaly detection layer.

Detection logic:
    A metric is flagged as an anomaly when it deviates from its 7-day
    rolling mean by more than THRESHOLD standard deviations, subject to
    a minimum absolute change to avoid flagging noise on stable metrics.

    Single-day spikes >= SPIKE_THRESHOLD are flagged regardless of std.

Business rules applied (from requirements):
    - Significant change: >= 15% deviation from 7-day rolling average
    - Single-day spike: >= 30% change from prior day
    - Anomaly confirmed: 2 consecutive flagged days OR 1-day spike >= 30%
"""

import pandas as pd
import numpy as np

ROLLING_THRESHOLD = 1.8      # standard deviations from rolling mean
PERCENT_THRESHOLD = 0.15     # 15% minimum deviation from rolling mean
SPIKE_THRESHOLD = 0.30       # 30% single-day change
CONSECUTIVE_DAYS = 2         # days before a moderate deviation is confirmed


def _pct_deviation(value: float, mean: float) -> float:
    if mean and mean != 0:
        return abs((value - mean) / mean)
    return 0.0


def detect_metric_anomalies(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Returns rows where the given metric is anomalous.
    Adds columns: anomaly_type, deviation_pct, direction.
    """
    col_mean = f"{metric}_roll_mean"
    col_std = f"{metric}_roll_std"
    col_prev = f"{metric}_prev_day"

    required = [metric, col_mean, col_std, col_prev]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df.copy().sort_values("date").reset_index(drop=True)

    # Percentage deviation from rolling mean
    df["_dev_pct"] = df.apply(
        lambda r: _pct_deviation(r[metric], r[col_mean]), axis=1
    )

    # Single-day spike from prior day
    df["_spike_pct"] = np.where(
        df[col_prev] > 0,
        abs((df[metric] - df[col_prev]) / df[col_prev]),
        0.0,
    )

    # Std-based flag
    df["_std_flag"] = (
        (df["_dev_pct"] >= PERCENT_THRESHOLD) &
        (df[col_std] > 0) &
        (abs(df[metric] - df[col_mean]) >= ROLLING_THRESHOLD * df[col_std])
    )

    # Spike flag
    df["_spike_flag"] = df["_spike_pct"] >= SPIKE_THRESHOLD

    # Consecutive flag: rolling sum of std_flag over 2 days
    df["_consec"] = df["_std_flag"].astype(int).rolling(
        window=CONSECUTIVE_DAYS, min_periods=CONSECUTIVE_DAYS
    ).sum()

    df["_is_anomaly"] = df["_spike_flag"] | (df["_consec"] >= CONSECUTIVE_DAYS)

    anomalies = df[df["_is_anomaly"]].copy()
    if anomalies.empty:
        return pd.DataFrame()

    anomalies["metric"] = metric
    anomalies["deviation_pct"] = anomalies["_dev_pct"].round(4)
    anomalies["spike_pct"] = anomalies["_spike_pct"].round(4)
    anomalies["direction"] = np.where(
        anomalies[metric] > anomalies[col_mean], "spike_up", "spike_down"
    )
    anomalies["anomaly_type"] = np.where(
        anomalies["_spike_flag"], "single_day_spike", "sustained_deviation"
    )

    return anomalies[["date", "metric", metric, col_mean, "deviation_pct",
                       "spike_pct", "direction", "anomaly_type"]]


def detect_all_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs anomaly detection across all tracked metrics.
    Returns a unified DataFrame of all anomalous events, sorted by date desc.
    """
    from analytics.metrics import TRACKED_METRICS

    frames = []
    for metric in TRACKED_METRICS:
        result = detect_metric_anomalies(df, metric)
        if not result.empty:
            frames.append(result)

    if not frames:
        return pd.DataFrame()

    all_anomalies = pd.concat(frames, ignore_index=True)
    all_anomalies = all_anomalies.sort_values("date", ascending=False).reset_index(drop=True)
    return all_anomalies


def get_recent_anomalies(df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    """Returns anomalies from the most recent N days."""
    if df.empty:
        return df
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].reset_index(drop=True)
