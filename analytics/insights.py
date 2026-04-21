"""
Rule-based insight generation layer.

Converts structured metric data and anomalies into prioritized,
human-readable insight objects before they reach the AI layer.

Each insight is a dict with:
    metric        - which KPI triggered it
    impact        - "high" | "medium" | "low"
    direction     - "positive" | "negative" | "neutral"
    finding       - one-sentence factual statement
    context       - supporting data point
    priority      - numeric sort key (lower = more important)

The AI layer receives these as structured context, not raw numbers.
This separation keeps the AI prompt focused and the logic testable.
"""

import pandas as pd
import numpy as np


IMPACT_RULES = {
    "revenue": {
        "high": 0.20,
        "medium": 0.10,
    },
    "cac": {
        "high": 0.25,
        "medium": 0.12,
    },
    "conversion_rate": {
        "high": 0.15,
        "medium": 0.08,
    },
    "roas": {
        "high": 0.20,
        "medium": 0.10,
    },
    "aov": {
        "high": 0.15,
        "medium": 0.08,
    },
    "orders": {
        "high": 0.20,
        "medium": 0.10,
    },
    "spend": {
        "high": 0.25,
        "medium": 0.12,
    },
}

METRIC_LABELS = {
    "revenue": "Revenue",
    "orders": "Orders",
    "aov": "Average Order Value",
    "conversion_rate": "Conversion Rate",
    "cac": "Customer Acquisition Cost",
    "roas": "Return on Ad Spend",
    "spend": "Marketing Spend",
}

# Metrics where an increase is negative (cost metrics)
INVERSE_METRICS = {"cac", "spend"}


def _classify_impact(metric: str, pct_change: float) -> str:
    rules = IMPACT_RULES.get(metric, {"high": 0.20, "medium": 0.10})
    abs_change = abs(pct_change)
    if abs_change >= rules["high"]:
        return "high"
    if abs_change >= rules["medium"]:
        return "medium"
    return "low"


def _classify_direction(metric: str, pct_change: float) -> str:
    if pct_change is None or np.isnan(pct_change):
        return "neutral"
    is_increase = pct_change > 0
    if metric in INVERSE_METRICS:
        return "negative" if is_increase else "positive"
    return "positive" if is_increase else "negative"


def generate_metric_insights(period_summary: dict) -> list[dict]:
    """
    Generates one insight per metric that has a meaningful week-over-week change.
    Skips metrics with no change or missing data.
    """
    insights = []
    priority_map = {"high": 1, "medium": 2, "low": 3}

    for metric, data in period_summary.items():
        if metric == "period_days" or not isinstance(data, dict):
            continue

        pct = data.get("pct_change")
        recent = data.get("recent_avg")
        prior = data.get("prior_avg")

        if pct is None or recent is None or prior is None:
            continue
        if abs(pct) < 0.03:
            continue

        label = METRIC_LABELS.get(metric, metric)
        impact = _classify_impact(metric, pct)
        direction = _classify_direction(metric, pct)
        pct_display = f"{'+' if pct > 0 else ''}{pct * 100:.1f}%"

        if metric == "conversion_rate":
            finding = (
                f"Conversion rate {pct_display} to {recent * 100:.2f}% "
                f"from {prior * 100:.2f}% in the prior period."
            )
        elif metric in ("cac",):
            finding = (
                f"CAC moved {pct_display} to ${recent:.2f} "
                f"from ${prior:.2f} — "
                f"{'higher acquisition cost per new customer' if pct > 0 else 'improved acquisition efficiency'}."
            )
        elif metric == "roas":
            finding = (
                f"ROAS {pct_display} to {recent:.2f}x "
                f"from {prior:.2f}x."
            )
        else:
            finding = (
                f"{label} is {pct_display} vs the prior 7-day average "
                f"(${recent:,.2f} vs ${prior:,.2f})."
                if metric in ("revenue", "aov", "spend")
                else f"{label} moved {pct_display} ({recent:,.0f} vs {prior:,.0f})."
            )

        insights.append({
            "metric": metric,
            "impact": impact,
            "direction": direction,
            "finding": finding,
            "context": {
                "recent_avg": recent,
                "prior_avg": prior,
                "pct_change": pct,
            },
            "priority": priority_map[impact],
        })

    return sorted(insights, key=lambda x: x["priority"])


def generate_anomaly_insights(anomalies: pd.DataFrame) -> list[dict]:
    """
    Converts anomaly detection output into structured insight objects.
    Only processes the most recent anomaly per metric to avoid duplication.
    """
    if anomalies.empty:
        return []

    insights = []
    seen_metrics = set()

    for _, row in anomalies.iterrows():
        metric = row["metric"]
        if metric in seen_metrics:
            continue
        seen_metrics.add(metric)

        label = METRIC_LABELS.get(metric, metric)
        direction = row.get("direction", "")
        atype = row.get("anomaly_type", "")
        dev_pct = row.get("deviation_pct", 0)
        dev_display = f"{dev_pct * 100:.1f}%"

        if atype == "single_day_spike":
            finding = (
                f"Anomaly detected: {label} had a single-day "
                f"{'surge' if 'up' in direction else 'drop'} of {dev_display} "
                f"on {row['date'].date() if hasattr(row['date'], 'date') else row['date']}."
            )
        else:
            finding = (
                f"Sustained deviation: {label} has been "
                f"{'above' if 'up' in direction else 'below'} its 7-day average "
                f"by {dev_display} for multiple consecutive days."
            )

        insights.append({
            "metric": metric,
            "impact": "high" if dev_pct >= 0.20 else "medium",
            "direction": "negative" if "down" in direction and metric not in INVERSE_METRICS else "positive",
            "finding": finding,
            "context": {
                "anomaly_type": atype,
                "deviation_pct": dev_pct,
                "direction": direction,
                "date": str(row["date"]),
            },
            "priority": 0,
        })

    return insights


def generate_channel_insights(channel_summary: pd.DataFrame) -> list[dict]:
    """
    Identifies best and worst performing channels by ROAS and conversion rate.
    """
    if channel_summary.empty:
        return []

    insights = []
    ch = channel_summary.dropna(subset=["roas"])

    if not ch.empty:
        best = ch.loc[ch["roas"].idxmax()]
        worst = ch.loc[ch["roas"].idxmin()]

        insights.append({
            "metric": "roas",
            "impact": "medium",
            "direction": "positive",
            "finding": (
                f"{best['channel']} is the top-performing channel "
                f"with a {best['roas']:.2f}x ROAS over the last 7 days."
            ),
            "context": {"channel": best["channel"], "roas": best["roas"]},
            "priority": 2,
        })

        if worst["channel"] != best["channel"] and worst["roas"] < 1.5:
            insights.append({
                "metric": "roas",
                "impact": "high",
                "direction": "negative",
                "finding": (
                    f"{worst['channel']} has a low ROAS of {worst['roas']:.2f}x — "
                    f"spend may not be generating sufficient return."
                ),
                "context": {"channel": worst["channel"], "roas": worst["roas"]},
                "priority": 1,
            })

    return insights


def compile_all_insights(
    period_summary: dict,
    anomalies: pd.DataFrame,
    channel_summary: pd.DataFrame,
) -> list[dict]:
    """
    Merges metric, anomaly, and channel insights into a single prioritized list.
    Anomaly insights are prepended as they are highest priority.
    """
    anomaly_ins = generate_anomaly_insights(anomalies)
    metric_ins = generate_metric_insights(period_summary)
    channel_ins = generate_channel_insights(channel_summary)

    all_insights = anomaly_ins + metric_ins + channel_ins
    return sorted(all_insights, key=lambda x: (x["priority"], x["metric"]))
