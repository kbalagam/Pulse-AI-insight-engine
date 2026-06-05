"""
Pulse — Anomalies
Master-detail anomaly investigation with AI explanation per event.
"""

import sys
import os
import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))
from utils import inject_styles
inject_styles()

from analytics.metrics import load_daily_metrics, enrich_daily_metrics
from analytics.anomaly import detect_all_anomalies, get_recent_anomalies

# ── Helpers ───────────────────────────────────────────────────────────────────

def _days(window: str) -> int:
    return {"Last 7 days": 7, "Last 14 days": 14,
            "Last 30 days": 30, "Last 90 days": 90}.get(window, 14)


METRIC_LABELS = {
    "revenue":         "Revenue",
    "orders":          "Orders",
    "aov":             "Average order value",
    "conversion_rate": "Conversion rate",
    "cac":             "Customer acquisition cost",
    "roas":            "Return on ad spend",
    "spend":           "Marketing spend",
}

METRIC_FORMAT = {
    "revenue":         lambda v: f"${v:,.0f}",
    "orders":          lambda v: f"{v:,.0f}",
    "aov":             lambda v: f"${v:,.2f}",
    "conversion_rate": lambda v: f"{v*100:.2f}%",
    "cac":             lambda v: f"${v:,.2f}",
    "roas":            lambda v: f"{v:.2f}x",
    "spend":           lambda v: f"${v:,.0f}",
}


def _fmt(metric: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return METRIC_FORMAT.get(metric, lambda v: f"{v:.2f}")(value)
    except Exception:
        return "N/A"


def _gemini_explain(metric, direction, deviation_pct, anomaly_type, period_summary) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return ""
    try:
        from ai.gemini import generate_anomaly_explanation
        return generate_anomaly_explanation(
            metric=metric,
            direction=direction,
            deviation_pct=deviation_pct,
            anomaly_type=anomaly_type,
            period_summary=period_summary,
        )
    except Exception as e:
        return f"Error generating explanation: {e}"


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading metrics...")
def _load():
    return enrich_daily_metrics(load_daily_metrics())


@st.cache_data(show_spinner="Running anomaly detection...")
def _anomalies(_df):
    return detect_all_anomalies(_df)


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-title">Anomalies</div>
<div class="page-sub">Statistical deviations detected across all tracked metrics</div>
""", unsafe_allow_html=True)

days    = _days(st.session_state.get("date_window", "Last 14 days"))
has_key = bool(st.session_state.get("gemini_key", ""))
ai_on   = st.session_state.get("ai_on", False)

df       = _load()
all_anom = _anomalies(df)
recent   = get_recent_anomalies(all_anom, days=days)

# ── Empty state ───────────────────────────────────────────────────────────────

if recent.empty:
    st.info(f"No anomalies detected in the last {days} days. Try expanding the date window.")
    st.stop()

# ── Filter bar ────────────────────────────────────────────────────────────────

col_filter, col_count = st.columns([3, 1])
with col_filter:
    anom_type_filter = st.radio(
        "Filter by type",
        ["All", "Single-day spike", "Sustained deviation"],
        horizontal=True,
        label_visibility="collapsed",
    )
with col_count:
    st.markdown(
        f"<div style='text-align:right; font-size:0.75rem; color:#9CA3AF; "
        f"padding-top:0.5rem;'>{len(recent)} event{'s' if len(recent) != 1 else ''} detected</div>",
        unsafe_allow_html=True,
    )

if anom_type_filter == "Single-day spike":
    recent = recent[recent["anomaly_type"] == "single_day_spike"]
elif anom_type_filter == "Sustained deviation":
    recent = recent[recent["anomaly_type"] == "sustained_deviation"]

if recent.empty:
    st.info("No anomalies match this filter.")
    st.stop()

# ── Master-detail layout ──────────────────────────────────────────────────────

log_col, detail_col = st.columns([1, 2.2], gap="small")

# ── Left: anomaly log ─────────────────────────────────────────────────────────

with log_col:
    st.markdown(
        '<div class="section-header">Anomaly log</div>',
        unsafe_allow_html=True,
    )

    selected_idx = st.session_state.get("selected_anomaly_idx", 0)
    selected_idx = min(selected_idx, len(recent) - 1)

    for i, (_, row) in enumerate(recent.iterrows()):
        metric    = row["metric"]
        direction = row.get("direction", "")
        atype     = row.get("anomaly_type", "")
        dev       = row.get("deviation_pct", 0)
        date_str  = pd.Timestamp(row["date"]).strftime("%b %d, %Y")

        badge_cls  = "badge-up" if "up" in direction else "badge-down"
        sign       = "+" if "up" in direction else "-"
        type_label = "Single-day spike" if atype == "single_day_spike" else "Sustained"
        is_sel     = i == selected_idx
        bg         = "#F0F4FF" if is_sel else "transparent"
        border_l   = "3px solid #1D4ED8" if is_sel else "3px solid transparent"

        if st.button(
            f"{METRIC_LABELS.get(metric, metric)} · {sign}{dev*100:.0f}% · {date_str}",
            key=f"anom_btn_{i}",
            use_container_width=True,
        ):
            st.session_state["selected_anomaly_idx"] = i
            st.rerun()

# ── Right: detail panel ───────────────────────────────────────────────────────

with detail_col:
    row = recent.iloc[selected_idx]

    metric     = row["metric"]
    direction  = row.get("direction", "")
    atype      = row.get("anomaly_type", "")
    dev_pct    = row.get("deviation_pct", 0)
    spike_pct  = row.get("spike_pct", 0)
    date_val   = pd.Timestamp(row["date"])
    col_mean   = f"{metric}_roll_mean"
    actual_val = row.get(metric)
    mean_val   = row.get(col_mean)

    is_up      = "up" in direction
    badge_cls  = "badge-up" if is_up else "badge-down"
    sign       = "+" if is_up else "-"
    type_label = "Single-day spike" if atype == "single_day_spike" else "Sustained deviation"
    delta_col  = "#15803D" if is_up else "#B91C1C"

    # ── Stat row ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">'
        f'{METRIC_LABELS.get(metric, metric)} &nbsp;'
        f'<span class="{badge_cls}">{type_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4, gap="small")

    with s1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Actual value</div>
            <div class="kpi-value" style="color:{delta_col};">{_fmt(metric, actual_val)}</div>
            <div class="kpi-hint">On {date_val.strftime('%b %d, %Y')}</div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">7-day average</div>
            <div class="kpi-value">{_fmt(metric, mean_val)}</div>
            <div class="kpi-hint">Rolling baseline</div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Deviation</div>
            <div class="kpi-value" style="color:{delta_col};">{sign}{dev_pct*100:.1f}%</div>
            <div class="kpi-hint">From rolling mean</div>
        </div>
        """, unsafe_allow_html=True)

    with s4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Detection type</div>
            <div class="kpi-value" style="font-size:1rem;">{type_label}</div>
            <div class="kpi-hint">{'Single day' if atype == 'single_day_spike' else '2+ consecutive days'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Metric chart ──────────────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="chart-title">{METRIC_LABELS.get(metric, metric)} · last {days} days</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chart-sub">Flagged day highlighted · rolling average shown</div>',
        unsafe_allow_html=True,
    )

    chart_df = df.sort_values("date").tail(days).copy()
    colors   = [
        "#E24B4A" if pd.Timestamp(d).date() == date_val.date() else "#BFDBFE"
        for d in chart_df["date"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=chart_df["date"],
        y=chart_df[metric],
        name=METRIC_LABELS.get(metric, metric),
        marker_color=colors,
        hovertemplate=f"<b>%{{x|%b %d}}</b><br>{METRIC_LABELS.get(metric,metric)}: %{{y:.2f}}<extra></extra>",
    ))
    if col_mean in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df["date"],
            y=chart_df[col_mean],
            name="7-day avg",
            line=dict(color="#1D4ED8", width=2),
            hovertemplate="<b>%{x|%b %d}</b><br>7d avg: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=11), tickformat="%b %d"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", tickfont=dict(size=11)),
        hovermode="x unified",
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── AI explanation ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">AI analysis</div>',
        unsafe_allow_html=True,
    )

    if not has_key:
        st.markdown("""
        <div class="ai-placeholder">
            Add a Gemini API key in the sidebar to generate probable causes
            and recommended actions for this anomaly.
        </div>
        """, unsafe_allow_html=True)
    else:
        explain_key = f"anomaly_explain_{metric}_{date_val.date()}"

        if explain_key not in st.session_state:
            if st.button("Generate AI analysis", key=f"gen_{explain_key}"):
                with st.spinner("Analysing anomaly..."):
                    from analytics.metrics import compute_period_summary
                    period_ctx = compute_period_summary(df, days=7)
                    st.session_state[explain_key] = _gemini_explain(
                        metric=metric,
                        direction=direction,
                        deviation_pct=dev_pct,
                        anomaly_type=atype,
                        period_summary=period_ctx,
                    )

        if explain_key in st.session_state:
            explanation = st.session_state[explain_key]
            if explanation:
                sections = explanation.split("\n\n")
                for section in sections:
                    lines = section.strip().split("\n")
                    if not lines:
                        continue
                    header = lines[0].strip()
                    items  = lines[1:]

                    if header:
                        st.markdown(
                            f"<div style='font-size:0.75rem; font-weight:600; "
                            f"color:#374151; margin-bottom:0.35rem; margin-top:0.6rem;'>"
                            f"{header}</div>",
                            unsafe_allow_html=True,
                        )
                    for item in items:
                        if item.strip():
                            st.markdown(
                                f"<div style='font-size:0.78rem; color:#6B7280; "
                                f"line-height:1.65; padding:0.2rem 0 0.2rem 0.8rem; "
                                f"border-left:2px solid #E5E7EB; margin-bottom:0.3rem;'>"
                                f"{item.strip()}</div>",
                                unsafe_allow_html=True,
                            )
