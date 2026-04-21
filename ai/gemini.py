"""
Gemini AI layer.

Calls the Gemini REST API directly using the requests library.
This approach has no SDK dependency, works with any Python environment,
and always targets the stable v1 endpoint — not v1beta.

Endpoint: https://generativelanguage.googleapis.com/v1/models/{model}:generateContent

Design decisions:
    - Direct REST over SDK: eliminates SDK version conflicts entirely.
      The v1beta issue was caused by google-generativeai defaulting to
      the beta endpoint regardless of version. The REST call targets v1
      explicitly.
    - gemini-2.0-flash: free tier (1,500 req/day), fast, sufficient for
      a dashboard that makes a few calls per session.
    - Temperature 0.3: consistent factual output without being robotic.
    - Each public function is independently callable so the dashboard
      requests only what it needs per render.
"""

import os
import textwrap
import requests
from dotenv import load_dotenv

load_dotenv()

_MODEL = "gemini-2.0-flash"
_TEMPERATURE = 0.3
_MAX_TOKENS = 1024
_API_BASE = "https://generativelanguage.googleapis.com/v1/models"


def _call(prompt: str) -> str:
    """
    Calls the Gemini v1 REST API directly.
    Raises a clear error if the API key is missing or the call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )

    url = f"{_API_BASE}/{_MODEL}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": _TEMPERATURE,
            "maxOutputTokens": _MAX_TOKENS,
        },
    }

    response = requests.post(url, json=payload, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text}"
        )

    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_insights_for_prompt(insights: list[dict]) -> str:
    """Converts insight dicts into a compact, readable prompt block."""
    lines = []
    for i, ins in enumerate(insights, 1):
        impact = ins.get("impact", "").upper()
        direction = ins.get("direction", "")
        finding = ins.get("finding", "")
        lines.append(f"{i}. [{impact}][{direction}] {finding}")
    return "\n".join(lines)


def _format_period_summary_for_prompt(summary: dict) -> str:
    """Formats the period summary dict into a readable metrics block."""
    lines = []
    for metric, data in summary.items():
        if metric == "period_days" or not isinstance(data, dict):
            continue
        pct = data.get("pct_change")
        recent = data.get("recent_avg")
        prior = data.get("prior_avg")
        if pct is None or recent is None:
            continue
        pct_str = f"{pct * 100:+.1f}%"
        lines.append(f"  {metric}: {recent} (vs {prior} prior, {pct_str})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_daily_narrative(
    insights: list[dict],
    period_summary: dict,
    date_label: str = "the most recent period",
) -> str:
    """
    Generates a 3-5 sentence executive summary of business performance.
    Suitable for the top of the dashboard or a daily digest email.
    """
    insights_text = _format_insights_for_prompt(insights[:6])
    metrics_text = _format_period_summary_for_prompt(period_summary)

    prompt = textwrap.dedent(f"""
        You are a senior business analyst writing a concise daily performance brief
        for a non-technical executive. Write 3 to 5 sentences only.

        Do not use bullet points. Do not use headers. Write in plain prose.
        Be direct and specific. Reference actual numbers where available.
        Do not start with "I" or "Here is".

        Time period: {date_label}

        Key metrics (recent vs prior 7-day average):
        {metrics_text}

        Detected insights (prioritized):
        {insights_text}

        Write the executive summary now:
    """).strip()

    return _call(prompt)


def generate_anomaly_explanation(
    metric: str,
    direction: str,
    deviation_pct: float,
    anomaly_type: str,
    period_summary: dict,
) -> str:
    """
    Generates a probable-cause explanation and 3 action recommendations
    for a specific anomaly. Returns a structured plain-text response.
    """
    metrics_text = _format_period_summary_for_prompt(period_summary)
    dev_display = f"{deviation_pct * 100:.1f}%"
    metric_labels = {
        "revenue": "Revenue",
        "orders": "Orders",
        "aov": "Average Order Value",
        "conversion_rate": "Conversion Rate",
        "cac": "Customer Acquisition Cost",
        "roas": "Return on Ad Spend",
        "spend": "Marketing Spend",
    }
    metric_label = metric_labels.get(metric, metric)
    move = "increased" if direction == "spike_up" else "decreased"

    prompt = textwrap.dedent(f"""
        You are a growth analyst investigating a metric anomaly.

        Anomaly: {metric_label} {move} by {dev_display} ({anomaly_type.replace('_', ' ')}).

        Other metrics context:
        {metrics_text}

        Respond in exactly this format, no extra text:

        LIKELY CAUSES:
        1. [cause one]
        2. [cause two]
        3. [cause three]

        RECOMMENDED ACTIONS:
        1. [action one]
        2. [action two]
        3. [action three]

        Keep each point to one sentence. Be specific and actionable.
    """).strip()

    return _call(prompt)


def generate_recommendations(
    insights: list[dict],
    channel_summary,
    period_summary: dict,
) -> str:
    """
    Generates 3-5 prioritized strategic recommendations based on the
    current state of all metrics and channel performance.
    """
    insights_text = _format_insights_for_prompt(insights[:8])
    metrics_text = _format_period_summary_for_prompt(period_summary)

    channel_lines = []
    try:
        for _, row in channel_summary.iterrows():
            roas = f"{row['roas']:.2f}x" if row.get("roas") and str(row["roas"]) != "nan" else "N/A"
            channel_lines.append(
                f"  {row['channel']}: spend=${row['spend']:.0f}, "
                f"conversions={row['conversions']:.0f}, ROAS={roas}"
            )
    except Exception:
        channel_lines = ["  Channel data unavailable"]

    channel_text = "\n".join(channel_lines)

    prompt = textwrap.dedent(f"""
        You are a growth strategy advisor. Based on the business data below,
        provide 3 to 5 prioritized, specific, actionable recommendations.

        Number each recommendation. Start each with an action verb.
        Be specific — name channels, metrics, or thresholds where relevant.
        Do not use vague language like "consider" or "think about".

        Current metric performance:
        {metrics_text}

        Channel performance (last 7 days):
        {channel_text}

        Detected insights:
        {insights_text}

        Recommendations:
    """).strip()

    return _call(prompt)


def generate_weekly_report_narrative(
    insights: list[dict],
    period_summary: dict,
    top_products: list[dict],
) -> str:
    """
    Generates a longer weekly narrative (6-8 sentences) suitable for
    export as a text report or stakeholder email.
    """
    insights_text = _format_insights_for_prompt(insights)
    metrics_text = _format_period_summary_for_prompt(period_summary)

    product_lines = "\n".join(
        f"  {p.get('product_label', 'Unknown')} ({p.get('category', '')}): "
        f"${p.get('revenue', 0):,.2f} revenue"
        for p in top_products[:5]
    )

    prompt = textwrap.dedent(f"""
        You are a senior analyst writing a weekly business performance report.
        Write 6 to 8 sentences in plain prose. No headers. No bullet points.
        Include metric trends, notable anomalies, channel performance,
        and top products. End with one forward-looking observation.

        Metrics:
        {metrics_text}

        Top products this week:
        {product_lines}

        Key findings:
        {insights_text}

        Weekly report:
    """).strip()

    return _call(prompt)
