"""
Daily report generator.

Assembles a plain-text daily insight report from analytics output
and optionally enriches it with Gemini-generated narratives.

The report is structured in sections so it can be read top-to-bottom
by a non-technical stakeholder in under 2 minutes.

Output: reports/output/report_YYYY-MM-DD.txt

Design note:
    The report runs in two modes:
    - AI mode: calls Gemini for narrative, explanation, and recommendations
    - Fallback mode: uses rule-based findings only (no API key required)
    This ensures the report always runs, even without a live API key.
"""

import os
from datetime import datetime, date
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SEPARATOR = "-" * 72


def _fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_currency(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_number(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_header(report_date: date) -> str:
    return "\n".join([
        SEPARATOR,
        "  AI GROWTH ANALYST — DAILY INSIGHT REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Reporting period: 7-day window ending {report_date}",
        SEPARATOR,
    ])


def _build_executive_summary(narrative: str) -> str:
    lines = [
        "",
        "EXECUTIVE SUMMARY",
        SEPARATOR,
    ]
    if narrative:
        lines.append(narrative)
    else:
        lines.append("AI narrative unavailable. See metric detail below.")
    return "\n".join(lines)


def _build_kpi_snapshot(period_summary: dict) -> str:
    METRIC_LABELS = {
        "revenue":         ("Revenue (avg/day)",      "currency"),
        "orders":          ("Orders (avg/day)",        "number"),
        "aov":             ("Avg Order Value",         "currency"),
        "conversion_rate": ("Conversion Rate",         "pct_raw"),
        "cac":             ("CAC",                     "currency"),
        "roas":            ("ROAS",                    "multiplier"),
        "spend":           ("Marketing Spend (avg/day)", "currency"),
    }

    lines = [
        "",
        "KPI SNAPSHOT  (last 7 days vs prior 7 days)",
        SEPARATOR,
        f"  {'Metric':<28} {'Recent':>12} {'Prior':>12} {'Change':>10}",
        f"  {'-'*28} {'-'*12} {'-'*12} {'-'*10}",
    ]

    for metric, (label, fmt) in METRIC_LABELS.items():
        data = period_summary.get(metric)
        if not isinstance(data, dict):
            continue
        recent = data.get("recent_avg")
        prior = data.get("prior_avg")
        pct = data.get("pct_change")

        if fmt == "currency":
            r_str = _fmt_currency(recent)
            p_str = _fmt_currency(prior)
        elif fmt == "number":
            r_str = _fmt_number(recent)
            p_str = _fmt_number(prior)
        elif fmt == "pct_raw":
            r_str = f"{float(recent)*100:.2f}%" if recent else "N/A"
            p_str = f"{float(prior)*100:.2f}%" if prior else "N/A"
        elif fmt == "multiplier":
            r_str = f"{float(recent):.2f}x" if recent else "N/A"
            p_str = f"{float(prior):.2f}x" if prior else "N/A"
        else:
            r_str = str(recent)
            p_str = str(prior)

        pct_str = _fmt_pct(pct)
        arrow = "^" if pct and pct > 0 else ("v" if pct and pct < 0 else "-")
        lines.append(f"  {label:<28} {r_str:>12} {p_str:>12} {arrow} {pct_str:>8}")

    return "\n".join(lines)


def _build_anomalies_section(anomalies) -> str:
    lines = [
        "",
        "ANOMALIES DETECTED",
        SEPARATOR,
    ]

    if anomalies is None or (hasattr(anomalies, "empty") and anomalies.empty):
        lines.append("  No anomalies detected in the reporting window.")
        return "\n".join(lines)

    METRIC_LABELS = {
        "revenue": "Revenue", "orders": "Orders", "aov": "Avg Order Value",
        "conversion_rate": "Conversion Rate", "cac": "CAC",
        "roas": "ROAS", "spend": "Marketing Spend",
    }

    for _, row in anomalies.iterrows():
        metric_label = METRIC_LABELS.get(row["metric"], row["metric"])
        direction = "UP" if "up" in str(row.get("direction", "")) else "DOWN"
        atype = str(row.get("anomaly_type", "")).replace("_", " ").title()
        dev = row.get("deviation_pct", 0)
        row_date = row["date"]
        if hasattr(row_date, "date"):
            row_date = row_date.date()
        lines.append(
            f"  [{direction}] {metric_label} — {atype} "
            f"({dev*100:.1f}% deviation) on {row_date}"
        )

    return "\n".join(lines)


def _build_insights_section(insights: list[dict], anomaly_explanation: str = None) -> str:
    lines = [
        "",
        "KEY FINDINGS",
        SEPARATOR,
    ]

    if not insights:
        lines.append("  No significant findings in the reporting window.")
        return "\n".join(lines)

    for ins in insights:
        impact = ins.get("impact", "").upper()
        direction_symbol = "+" if ins.get("direction") == "positive" else "-"
        lines.append(f"  [{impact}] {direction_symbol} {ins['finding']}")

    if anomaly_explanation:
        lines += [
            "",
            "  ANOMALY DEEP-DIVE (AI Analysis):",
            "",
        ]
        for line in anomaly_explanation.strip().split("\n"):
            lines.append(f"  {line}")

    return "\n".join(lines)


def _build_channel_section(channel_summary) -> str:
    lines = [
        "",
        "CHANNEL PERFORMANCE  (last 7 days)",
        SEPARATOR,
        f"  {'Channel':<22} {'Spend':>10} {'Clicks':>8} {'Conv':>6} {'Rev Attr':>12} {'ROAS':>8}",
        f"  {'-'*22} {'-'*10} {'-'*8} {'-'*6} {'-'*12} {'-'*8}",
    ]

    if channel_summary is None or (hasattr(channel_summary, "empty") and channel_summary.empty):
        lines.append("  Channel data unavailable.")
        return "\n".join(lines)

    for _, row in channel_summary.iterrows():
        roas_str = f"{row['roas']:.2f}x" if str(row.get("roas", "")) not in ("nan", "None", "") else "N/A"
        lines.append(
            f"  {str(row['channel']):<22} "
            f"${row['spend']:>9,.0f} "
            f"{row['clicks']:>8,.0f} "
            f"{row['conversions']:>6,.0f} "
            f"${row['revenue_attributed']:>11,.0f} "
            f"{roas_str:>8}"
        )

    return "\n".join(lines)


def _build_top_products_section(top_products) -> str:
    lines = [
        "",
        "TOP PRODUCTS  (last 7 days by revenue)",
        SEPARATOR,
    ]

    if top_products is None or (hasattr(top_products, "empty") and top_products.empty):
        lines.append("  Product data unavailable.")
        return "\n".join(lines)

    for i, row in enumerate(top_products.itertuples(), 1):
        lines.append(
            f"  {i}. {row.product_label:<35} "
            f"({row.category})  "
            f"Revenue: ${row.revenue:,.2f}  "
            f"Units: {row.units_sold:.0f}"
        )

    return "\n".join(lines)


def _build_recommendations_section(recommendations: str) -> str:
    lines = [
        "",
        "RECOMMENDED ACTIONS",
        SEPARATOR,
    ]

    if not recommendations:
        lines.append("  Recommendations unavailable (AI layer not configured).")
        return "\n".join(lines)

    for line in recommendations.strip().split("\n"):
        if line.strip():
            lines.append(f"  {line.strip()}")

    return "\n".join(lines)


def _build_footer() -> str:
    return "\n".join([
        "",
        SEPARATOR,
        "  Report generated by AI Growth Analyst",
        "  AI narratives powered by Google Gemini 1.5 Flash",
        "  Data: fact_daily_metrics, fact_marketing_channel, fact_product_sales",
        SEPARATOR,
        "",
    ])


# ---------------------------------------------------------------------------
# Main report assembly
# ---------------------------------------------------------------------------

def generate_report(
    period_summary: dict,
    insights: list[dict],
    anomalies,
    channel_summary,
    top_products,
    report_date: date = None,
    ai_narrative: str = None,
    ai_anomaly_explanation: str = None,
    ai_recommendations: str = None,
    save: bool = True,
) -> str:
    """
    Assembles the full daily report as a string and optionally saves it.

    Parameters
    ----------
    period_summary        : output of compute_period_summary()
    insights              : output of compile_all_insights()
    anomalies             : output of get_recent_anomalies()
    channel_summary       : output of compute_channel_summary()
    top_products          : output of compute_top_products()
    report_date           : date to stamp on the report (defaults to today)
    ai_narrative          : Gemini daily narrative string (optional)
    ai_anomaly_explanation: Gemini anomaly explanation string (optional)
    ai_recommendations    : Gemini recommendations string (optional)
    save                  : write report to reports/output/ if True
    """
    if report_date is None:
        report_date = date.today()

    sections = [
        _build_header(report_date),
        _build_executive_summary(ai_narrative),
        _build_kpi_snapshot(period_summary),
        _build_anomalies_section(anomalies),
        _build_insights_section(insights, ai_anomaly_explanation),
        _build_channel_section(channel_summary),
        _build_top_products_section(top_products),
        _build_recommendations_section(ai_recommendations),
        _build_footer(),
    ]

    report_text = "\n".join(sections)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = OUTPUT_DIR / f"report_{report_date}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report saved: {filename}")

    return report_text
