"""
Pulse — Forecast
90-day revenue forecast using Holt-Winters Exponential Smoothing.
"""

import sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analytics.forecasting import build_forecast, forecast_summary
from analytics.metrics import (
    load_daily_metrics,
    enrich_daily_metrics,
    compute_period_summary,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_currency(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"${v:,.0f}"


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


def _kpi_card(label, value, delta_str="", delta_class="kpi-delta-neu", hint=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div>
            {"<span class='" + delta_class + "'>" + delta_str + "</span>" if delta_str else ""}
            {"<div class='kpi-hint'>" + hint + "</div>" if hint else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _ai_chart_insight(key: str, context: str, ai_on: bool, has_key: bool):
    with st.expander("AI insight for this chart"):
        if not has_key:
            st.markdown(
                '<div class="ai-placeholder">Add a Gemini API key in the sidebar to generate insights.</div>',
                unsafe_allow_html=True,
            )
            return
        if not ai_on:
            st.markdown(
                '<div class="ai-placeholder">Turn on "Show AI insights on charts" in the sidebar.</div>',
                unsafe_allow_html=True,
            )
            return
        btn_key = f"btn_{key}"
        res_key = f"res_{key}"
        if st.button("Generate insight", key=btn_key):
            with st.spinner("Analysing..."):
                try:
                    import requests, os
                    prompt = (
                        "You are a revenue forecasting analyst. Give 2-3 concise, "
                        "specific, actionable observations about this forecast in "
                        "plain English. Comment on the trend, seasonality if visible, "
                        "and confidence range. No bullet points. Do not start with 'I'.\n\n"
                        f"Context: {context}"
                    )
                    url = (
                        "https://generativelanguage.googleapis.com/v1/models/"
                        f"gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}"
                    )
                    r = requests.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.3,
                                "maxOutputTokens": 300,
                            },
                        },
                        timeout=30,
                    )
                    result = (
                        r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if r.status_code == 200
                        else f"API error {r.status_code}"
                    )
                except Exception as e:
                    result = f"Error: {e}"
            st.session_state[res_key] = result
        if res_key in st.session_state:
            st.markdown(
                f'<div class="ai-insight-box">{st.session_state[res_key]}</div>',
                unsafe_allow_html=True,
            )


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building forecast model...")
def _load_forecast(horizon: int):
    fc  = build_forecast(horizon_days=horizon)
    fcs = forecast_summary(fc)
    return fc, fcs


@st.cache_data(show_spinner="Loading daily metrics...")
def _load_metrics():
    return enrich_daily_metrics(load_daily_metrics())


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-title">Forecast</div>
<div class="page-sub">Revenue forecast using Holt-Winters exponential smoothing</div>
""", unsafe_allow_html=True)

has_key = bool(st.session_state.get("gemini_key", ""))
ai_on   = st.session_state.get("ai_on", False)

# ── Horizon selector ──────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Forecast horizon</div>',
    unsafe_allow_html=True,
)

horizon_opt = st.radio(
    "Forecast horizon",
    ["30 days", "60 days", "90 days"],
    index=2,
    horizontal=True,
    label_visibility="collapsed",
)
horizon = int(horizon_opt.split()[0])

fc_df, fcs = _load_forecast(horizon)
df         = _load_metrics()
period     = compute_period_summary(df, days=7)

# ── AI page summary ───────────────────────────────────────────────────────────

if has_key:
    fc_sum_key = f"forecast_summary_{horizon}"
    if fc_sum_key not in st.session_state:
        with st.spinner("Generating AI summary..."):
            try:
                import requests, os
                prompt = (
                    "You are a revenue forecasting analyst. Write 2-3 sentences "
                    "summarising this forecast for a non-technical executive. "
                    "Be specific about the projected trend and confidence range. "
                    "No bullet points. Do not start with 'I'.\n\n"
                    f"Forecast data: {fcs}\n"
                    f"Recent metrics: {period}"
                )
                url = (
                    "https://generativelanguage.googleapis.com/v1/models/"
                    f"gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}"
                )
                r = requests.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.3,
                            "maxOutputTokens": 200,
                        },
                    },
                    timeout=30,
                )
                st.session_state[fc_sum_key] = (
                    r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if r.status_code == 200 else None
                )
            except Exception:
                st.session_state[fc_sum_key] = None

    narrative = st.session_state.get(fc_sum_key)
    if narrative:
        st.markdown(f"""
        <div class="ai-strip">
            <div class="ai-strip-label">AI observation</div>
            {narrative}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="ai-placeholder">
        Add a Gemini API key in the sidebar to unlock AI forecast observations.
    </div>
    """, unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────

last_actual   = fcs.get("last_actual_revenue")
avg_actual_30 = fcs.get("avg_actual_30d")
avg_forecast  = fcs.get("avg_forecast")
fc_change_pct = fcs.get("forecast_change_pct")
lower_avg     = fcs.get("lower_bound_avg")
upper_avg     = fcs.get("upper_bound_avg")

delta_cls = (
    "kpi-delta-up" if fc_change_pct and fc_change_pct > 0
    else "kpi-delta-down" if fc_change_pct and fc_change_pct < 0
    else "kpi-delta-neu"
)

k1, k2, k3, k4 = st.columns(4, gap="small")

with k1:
    _kpi_card(
        "Last actual (daily)",
        _fmt_currency(last_actual),
        hint="Most recent data point",
    )
with k2:
    _kpi_card(
        "30-day actual avg",
        _fmt_currency(avg_actual_30),
        hint="Recent baseline",
    )
with k3:
    _kpi_card(
        f"Avg forecast ({horizon}d)",
        _fmt_currency(avg_forecast),
        delta_str=_fmt_pct(fc_change_pct),
        delta_class=delta_cls,
        hint="vs 30-day actual avg",
    )
with k4:
    _kpi_card(
        "95% CI range (daily)",
        f"{_fmt_currency(lower_avg)} – {_fmt_currency(upper_avg)}",
        hint="Based on residual std",
    )

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ── Main forecast chart ───────────────────────────────────────────────────────

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown(
    f'<div class="chart-title">Revenue forecast · {horizon}-day horizon</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="chart-sub">'
    'Actual (solid blue) · Forecast (dashed) · 95% confidence band (shaded)'
    '</div>',
    unsafe_allow_html=True,
)

hist    = fc_df[fc_df["actual"].notna()].copy()
future  = fc_df[fc_df["actual"].isna()].copy()

# Show last 90 days of history for context
hist_plot = hist.tail(90)

fig = go.Figure()

# Confidence band
fig.add_trace(go.Scatter(
    x=pd.concat([future["date"], future["date"].iloc[::-1]]),
    y=pd.concat([future["upper_ci"], future["lower_ci"].iloc[::-1]]),
    fill="toself",
    fillcolor="rgba(29, 78, 216, 0.08)",
    line=dict(color="rgba(0,0,0,0)"),
    name="95% CI",
    hoverinfo="skip",
))

# Actual revenue
fig.add_trace(go.Scatter(
    x=hist_plot["date"],
    y=hist_plot["actual"].round(0),
    name="Actual",
    line=dict(color="#1D4ED8", width=2),
    hovertemplate="<b>%{x|%b %d}</b><br>Actual: $%{y:,.0f}<extra></extra>",
))

# Historical fitted values
fig.add_trace(go.Scatter(
    x=hist_plot["date"],
    y=hist_plot["forecast"].round(0),
    name="Model fit",
    line=dict(color="#93C5FD", width=1.5, dash="dot"),
    hovertemplate="<b>%{x|%b %d}</b><br>Fitted: $%{y:,.0f}<extra></extra>",
))

# Future forecast
fig.add_trace(go.Scatter(
    x=future["date"],
    y=future["forecast"].round(0),
    name="Forecast",
    line=dict(color="#1D4ED8", width=2, dash="dash"),
    hovertemplate="<b>%{x|%b %d}</b><br>Forecast: $%{y:,.0f}<extra></extra>",
))

# Divider line at forecast start
if not hist.empty and not future.empty:
    split_date = future["date"].iloc[0]
    fig.add_vline(
        x=split_date,
        line_dash="dot",
        line_color="#E5E7EB",
        line_width=1.5,
        annotation_text="Forecast start",
        annotation_font_size=10,
        annotation_font_color="#9CA3AF",
    )

fig.update_layout(
    height=380,
    margin=dict(l=0, r=0, t=8, b=0),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, font=dict(size=11),
    ),
    xaxis=dict(
        showgrid=False,
        tickfont=dict(size=11),
        tickformat="%b %d",
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#F3F4F6",
        tickfont=dict(size=11),
        tickprefix="$",
        tickformat=",.0f",
    ),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

_ai_chart_insight(
    key="forecast_main",
    context=(
        f"Revenue forecast chart showing last 90 days of actual data and "
        f"{horizon}-day forward forecast using Holt-Winters exponential smoothing "
        f"with weekly seasonality. "
        f"Last actual daily revenue: {_fmt_currency(last_actual)}. "
        f"Avg forecast: {_fmt_currency(avg_forecast)} "
        f"({_fmt_pct(fc_change_pct)} vs 30-day actual avg). "
        f"95% CI: {_fmt_currency(lower_avg)} to {_fmt_currency(upper_avg)} per day."
    ),
    ai_on=ai_on,
    has_key=has_key,
)

# ── Weekly seasonality chart ──────────────────────────────────────────────────

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

seas_l, seas_r = st.columns(2, gap="small")

with seas_l:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-title">Average revenue by day of week</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chart-sub">Historical actuals · weekly seasonality pattern</div>',
        unsafe_allow_html=True,
    )

    hist_copy = hist.copy()
    hist_copy["dow"] = pd.to_datetime(hist_copy["date"]).dt.day_name()
    dow_order  = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]
    dow_avg    = (
        hist_copy.groupby("dow")["actual"]
        .mean()
        .reindex(dow_order)
        .round(0)
    )
    overall_avg = dow_avg.mean()
    bar_colors  = [
        "#1D4ED8" if v >= overall_avg else "#BFDBFE"
        for v in dow_avg.values
    ]

    fig2 = go.Figure(go.Bar(
        x=dow_avg.index,
        y=dow_avg.values,
        marker_color=bar_colors,
        hovertemplate="<b>%{x}</b><br>Avg revenue: $%{y:,.0f}<extra></extra>",
    ))
    fig2.add_hline(
        y=overall_avg,
        line_dash="dash",
        line_color="#E5E7EB",
        line_width=1.5,
        annotation_text=f"Avg ${overall_avg:,.0f}",
        annotation_font_size=10,
        annotation_font_color="#9CA3AF",
    )
    fig2.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(
            showgrid=True, gridcolor="#F3F4F6",
            tickfont=dict(size=11), tickprefix="$", tickformat=",.0f",
        ),
        bargap=0.3,
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    _ai_chart_insight(
        key="forecast_dow",
        context=(
            "Bar chart of average daily revenue by day of week across all "
            "historical data. Dark bars are above the weekly average, "
            f"light bars are below. Overall daily avg: ${overall_avg:,.0f}. "
            f"Peak day: {dow_avg.idxmax()} at ${dow_avg.max():,.0f}. "
            f"Lowest day: {dow_avg.idxmin()} at ${dow_avg.min():,.0f}."
        ),
        ai_on=ai_on,
        has_key=has_key,
    )

with seas_r:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-title">30-day rolling average revenue</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chart-sub">Smoothed trend · all historical data</div>',
        unsafe_allow_html=True,
    )

    roll30 = hist.copy()
    roll30["roll30"] = roll30["actual"].rolling(window=30, min_periods=1).mean().round(0)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=roll30["date"],
        y=roll30["actual"].round(0),
        name="Daily actual",
        line=dict(color="#BFDBFE", width=1),
        hovertemplate="<b>%{x|%b %d}</b><br>Actual: $%{y:,.0f}<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=roll30["date"],
        y=roll30["roll30"],
        name="30-day avg",
        line=dict(color="#1D4ED8", width=2),
        hovertemplate="<b>%{x|%b %d}</b><br>30d avg: $%{y:,.0f}<extra></extra>",
    ))
    fig3.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=11), tickformat="%b %y"),
        yaxis=dict(
            showgrid=True, gridcolor="#F3F4F6",
            tickfont=dict(size=11), tickprefix="$", tickformat=",.0f",
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    _ai_chart_insight(
        key="forecast_rolling",
        context=(
            "Line chart showing daily actual revenue (light blue) and "
            "30-day rolling average (dark blue) across all historical data. "
            f"Most recent 30-day avg: {_fmt_currency(avg_actual_30)}."
        ),
        ai_on=ai_on,
        has_key=has_key,
    )

# ── Methodology note ──────────────────────────────────────────────────────────

st.markdown("""
<div class="home-note" style="margin-top:1rem;">
Forecast model: Holt-Winters exponential smoothing with additive trend and
additive weekly seasonality (period = 7). Confidence intervals are estimated
using the standard deviation of in-sample residuals scaled by 1.96.
Model is fit on all available historical data and re-computed when the
horizon changes.
</div>
""", unsafe_allow_html=True)
