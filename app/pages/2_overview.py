"""
Pulse — Overview
Daily KPI summary, revenue trend, anomaly snapshot, and top products.
"""

import sys
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

from analytics.metrics import (
    load_daily_metrics,
    load_product_sales,
    enrich_daily_metrics,
    compute_period_summary,
    compute_top_products,
    load_channel_metrics,
    compute_channel_summary,
)
from analytics.anomaly import detect_all_anomalies, get_recent_anomalies
from analytics.insights import compile_all_insights

# ── Helpers ───────────────────────────────────────────────────────────────────

def _days(window: str) -> int:
    return {"Last 7 days": 7, "Last 14 days": 14,
            "Last 30 days": 30, "Last 90 days": 90}.get(window, 7)

def _fmt_currency(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"${v:,.0f}"

def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.1f}%"

def _fmt_x(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.2f}x"

def _delta_class(v, inverse: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "kpi-delta-neu"
    positive = v > 0
    if inverse:
        positive = not positive
    return "kpi-delta-up" if positive else "kpi-delta-down"

def _kpi_card(label, value, delta_str, delta_class, hint=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div>
            <span class="{delta_class}">{delta_str}</span>
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
                        "You are a business analyst. Give 2-3 concise, specific, "
                        "actionable observations about this chart in plain English. "
                        "No bullet points. Do not start with 'I'.\n\n"
                        f"Chart context: {context}"
                    )
                    url = (
                        "https://generativelanguage.googleapis.com/v1/models/"
                        f"gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}"
                    )
                    r = requests.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 300},
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

@st.cache_data(show_spinner="Loading data...")
def _load():
    df   = enrich_daily_metrics(load_daily_metrics())
    ch   = load_channel_metrics()
    prod = load_product_sales()
    return df, ch, prod

@st.cache_data(show_spinner="Running anomaly detection...")
def _anomalies(_df):
    return detect_all_anomalies(_df)

# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-title">Overview</div>
<div class="page-sub">Daily performance summary</div>
""", unsafe_allow_html=True)

days    = _days(st.session_state.get("date_window", "Last 7 days"))
top_n   = st.session_state.get("top_n", 5)
has_key = bool(st.session_state.get("gemini_key", ""))
ai_on   = st.session_state.get("ai_on", False)

df, ch, prod = _load()
all_anom     = _anomalies(df)
recent_anom  = get_recent_anomalies(all_anom, days=days)
period       = compute_period_summary(df, days=days)
ch_sum       = compute_channel_summary(ch, days=days)
top_prod     = compute_top_products(prod, n=top_n, days=days)
insights     = compile_all_insights(period, recent_anom, ch_sum)

# ── AI page summary ───────────────────────────────────────────────────────────

if has_key:
    sum_key = f"overview_summary_{days}"
    if sum_key not in st.session_state:
        with st.spinner("Generating AI summary..."):
            try:
                from ai.gemini import generate_daily_narrative
                st.session_state[sum_key] = generate_daily_narrative(
                    insights, period, date_label=f"the last {days} days",
                )
            except Exception:
                st.session_state[sum_key] = None
    narrative = st.session_state.get(sum_key)
    if narrative:
        st.markdown(f"""
        <div class="ai-strip">
            <div class="ai-strip-label">AI summary</div>
            {narrative}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="ai-placeholder">
        Add a Gemini API key in the sidebar to unlock the AI page summary.
    </div>
    """, unsafe_allow_html=True)

# ── KPI row 1 ─────────────────────────────────────────────────────────────────

rev  = period.get("revenue", {})
ord_ = period.get("orders", {})
cr   = period.get("conversion_rate", {})
cac  = period.get("cac", {})
roas = period.get("roas", {})
aov  = period.get("aov", {})
spend = period.get("spend", {})

k1, k2, k3, k4 = st.columns(4, gap="small")
with k1:
    _kpi_card("Revenue", _fmt_currency(rev.get("recent_avg")),
              _fmt_pct(rev.get("pct_change")), _delta_class(rev.get("pct_change")),
              hint=f"vs {_fmt_currency(rev.get('prior_avg'))} prior")
with k2:
    _kpi_card("Orders",
              f"{ord_.get('recent_avg', 0):,.0f}" if ord_.get("recent_avg") else "N/A",
              _fmt_pct(ord_.get("pct_change")), _delta_class(ord_.get("pct_change")),
              hint=f"vs {ord_.get('prior_avg', 0):,.0f} prior" if ord_.get("prior_avg") else "")
with k3:
    cr_r = cr.get("recent_avg")
    cr_p = cr.get("prior_avg")
    _kpi_card("Conversion rate",
              f"{cr_r * 100:.2f}%" if cr_r else "N/A",
              _fmt_pct(cr.get("pct_change")), _delta_class(cr.get("pct_change")),
              hint=f"vs {cr_p * 100:.2f}% prior" if cr_p else "")
with k4:
    _kpi_card("CAC", _fmt_currency(cac.get("recent_avg")),
              _fmt_pct(cac.get("pct_change")), _delta_class(cac.get("pct_change"), inverse=True),
              hint=f"vs {_fmt_currency(cac.get('prior_avg'))} prior")

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

k5, k6, k7, k8 = st.columns(4, gap="small")
with k5:
    _kpi_card("ROAS", _fmt_x(roas.get("recent_avg")),
              _fmt_pct(roas.get("pct_change")), _delta_class(roas.get("pct_change")),
              hint=f"vs {_fmt_x(roas.get('prior_avg'))} prior")
with k6:
    _kpi_card("AOV", _fmt_currency(aov.get("recent_avg")),
              _fmt_pct(aov.get("pct_change")), _delta_class(aov.get("pct_change")),
              hint=f"vs {_fmt_currency(aov.get('prior_avg'))} prior")
with k7:
    _kpi_card("Total spend", _fmt_currency(spend.get("recent_avg")),
              _fmt_pct(spend.get("pct_change")), _delta_class(spend.get("pct_change"), inverse=True),
              hint=f"vs {_fmt_currency(spend.get('prior_avg'))} prior")
with k8:
    n_anom = len(recent_anom) if not recent_anom.empty else 0
    _kpi_card("Anomalies detected", str(n_anom),
              f"last {days} days", "kpi-delta-neu",
              hint="See Anomalies page for detail")

# ── Revenue chart + Anomaly snapshot ─────────────────────────────────────────

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
ch_left, ch_right = st.columns([1.6, 1], gap="small")

with ch_left:
    st.markdown('<div class="chart-title">Daily revenue</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-sub">Last {days} days with 7-day rolling average</div>',
                unsafe_allow_html=True)
    recent_df = df.sort_values("date").tail(days)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=recent_df["date"], y=recent_df["revenue"], name="Revenue",
        marker_color="#BFDBFE",
        hovertemplate="<b>%{x|%b %d}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=recent_df["date"], y=recent_df["revenue_roll_mean"], name="7-day avg",
        line=dict(color="#1D4ED8", width=2),
        hovertemplate="<b>%{x|%b %d}</b><br>7d avg: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(showgrid=False, tickfont=dict(size=11), tickformat="%b %d"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", tickfont=dict(size=11),
                   tickprefix="$", tickformat=",.0f"),
        hovermode="x unified", bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    _ai_chart_insight(
        key="overview_revenue",
        context=(f"Daily revenue bar chart for the last {days} days with 7-day rolling average. "
                 f"Recent avg: {_fmt_currency(rev.get('recent_avg'))}, "
                 f"prior avg: {_fmt_currency(rev.get('prior_avg'))}, "
                 f"WoW change: {_fmt_pct(rev.get('pct_change'))}."),
        ai_on=ai_on, has_key=has_key,
    )

with ch_right:
    st.markdown('<div class="chart-title">Anomalies detected</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-sub">Last {days} days · all metrics</div>',
                unsafe_allow_html=True)
    if recent_anom.empty:
        st.markdown("<div style='padding:1rem 0; color:#9CA3AF; font-size:0.8rem;'>"
                    "No anomalies detected in this window.</div>", unsafe_allow_html=True)
    else:
        for _, row in recent_anom.head(6).iterrows():
            direction  = row.get("direction", "")
            atype      = row.get("anomaly_type", "")
            metric     = row.get("metric", "")
            dev        = row.get("deviation_pct", 0)
            date_str   = pd.Timestamp(row["date"]).strftime("%b %d")
            badge_cls  = "badge-up" if "up" in direction else "badge-down"
            sign       = "+" if "up" in direction else "-"
            type_label = "Single-day spike" if atype == "single_day_spike" else "Sustained"
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between;
                        padding:0.45rem 0; border-bottom:0.5px solid #F3F4F6;">
                <div>
                    <div style="font-size:0.78rem; font-weight:600; color:#111827;">
                        {metric.replace('_',' ').title()}</div>
                    <div style="font-size:0.65rem; color:#9CA3AF;">{date_str} · {type_label}</div>
                </div>
                <span class="{badge_cls}">{sign}{dev*100:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)

# ── Conversion rate trend ─────────────────────────────────────────────────────

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="chart-title">Conversion rate & AOV trend</div>', unsafe_allow_html=True)
st.markdown(f'<div class="chart-sub">Last {days} days · dual axis</div>', unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=recent_df["date"], y=(recent_df["conversion_rate"] * 100).round(2),
    name="Conversion rate (%)", line=dict(color="#1D4ED8", width=2), yaxis="y1",
    hovertemplate="<b>%{x|%b %d}</b><br>Conv rate: %{y:.2f}%<extra></extra>",
))
fig2.add_trace(go.Scatter(
    x=recent_df["date"], y=recent_df["aov"].round(2),
    name="AOV ($)", line=dict(color="#059669", width=2, dash="dot"), yaxis="y2",
    hovertemplate="<b>%{x|%b %d}</b><br>AOV: $%{y:,.2f}<extra></extra>",
))
fig2.update_layout(
    height=280, margin=dict(l=0, r=40, t=8, b=0),
    plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11)),
    xaxis=dict(showgrid=False, tickfont=dict(size=11), tickformat="%b %d"),
    yaxis=dict(showgrid=True, gridcolor="#F3F4F6", tickfont=dict(size=11),
               ticksuffix="%", title="Conv rate", title_font=dict(size=11)),
    yaxis2=dict(overlaying="y", side="right", tickfont=dict(size=11),
                tickprefix="$", title="AOV", title_font=dict(size=11), showgrid=False),
    hovermode="x unified",
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
_ai_chart_insight(
    key="overview_cr_aov",
    context=(f"Dual-axis line chart showing conversion rate and AOV over the last {days} days. "
             f"Conversion rate recent avg: {cr.get('recent_avg', 0)*100:.2f}%, "
             f"AOV recent avg: {_fmt_currency(aov.get('recent_avg'))}."),
    ai_on=ai_on, has_key=has_key,
)

# ── Top products ──────────────────────────────────────────────────────────────

st.markdown(f'<div class="section-header">Top {top_n} products · last {days} days</div>',
            unsafe_allow_html=True)
if top_prod.empty:
    st.info("No product data available for this window.")
else:
    top_prod_display = top_prod.copy()
    top_prod_display["revenue"]    = top_prod_display["revenue"].apply(lambda x: f"${x:,.0f}")
    top_prod_display["units_sold"] = top_prod_display["units_sold"].apply(lambda x: f"{x:,.0f}")
    top_prod_display = top_prod_display.rename(columns={
        "product_label": "Product", "category": "Category",
        "revenue": "Revenue", "units_sold": "Units sold",
    })[["Product", "Category", "Revenue", "Units sold"]]
    st.dataframe(top_prod_display, use_container_width=True, hide_index=True)

_ai_chart_insight(
    key="overview_products",
    context=(f"Table of top {top_n} products by revenue over the last {days} days. "
             f"Top product: {top_prod.iloc[0]['product_label'] if not top_prod.empty else 'N/A'} "
             f"with revenue {_fmt_currency(top_prod.iloc[0]['revenue']) if not top_prod.empty else 'N/A'}."),
    ai_on=ai_on, has_key=has_key,
)
