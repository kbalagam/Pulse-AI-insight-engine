"""
Pulse — Marketing
Channel performance, spend vs return, and ROAS breakdown.
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
    load_channel_metrics, load_daily_metrics, enrich_daily_metrics,
    compute_channel_summary, compute_period_summary,
)
from analytics.insights import compile_all_insights, generate_channel_insights
from analytics.anomaly import detect_all_anomalies, get_recent_anomalies

# ── Helpers ───────────────────────────────────────────────────────────────────

def _days(window: str) -> int:
    return {"Last 7 days": 7, "Last 14 days": 14,
            "Last 30 days": 30, "Last 90 days": 90}.get(window, 7)

def _fmt_currency(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"${v:,.0f}"

def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.1f}%"

def _fmt_x(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v:.2f}x"

def _delta_class(v, inverse: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "kpi-delta-neu"
    positive = v > 0
    if inverse: positive = not positive
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
            st.markdown('<div class="ai-placeholder">Add a Gemini API key in the sidebar.</div>',
                        unsafe_allow_html=True)
            return
        if not ai_on:
            st.markdown('<div class="ai-placeholder">Turn on "Show AI insights on charts" in the sidebar.</div>',
                        unsafe_allow_html=True)
            return
        btn_key = f"btn_{key}"
        res_key = f"res_{key}"
        if st.button("Generate insight", key=btn_key):
            with st.spinner("Analysing..."):
                try:
                    import requests, os
                    prompt = (
                        "You are a marketing analyst. Give 2-3 concise, specific, "
                        "actionable observations about this chart in plain English. "
                        "No bullet points. Do not start with 'I'.\n\n"
                        f"Chart context: {context}"
                    )
                    url = (
                        "https://generativelanguage.googleapis.com/v1/models/"
                        f"gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}"
                    )
                    r = requests.post(url, json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 300},
                    }, timeout=30)
                    result = (r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                              if r.status_code == 200 else f"API error {r.status_code}")
                except Exception as e:
                    result = f"Error: {e}"
            st.session_state[res_key] = result
        if res_key in st.session_state:
            st.markdown(f'<div class="ai-insight-box">{st.session_state[res_key]}</div>',
                        unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading marketing data...")
def _load():
    ch = load_channel_metrics()
    df = enrich_daily_metrics(load_daily_metrics())
    return ch, df

@st.cache_data(show_spinner="Running anomaly detection...")
def _anomalies(_df):
    return detect_all_anomalies(_df)

# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-title">Marketing</div>
<div class="page-sub">Channel performance, spend efficiency, and return on ad spend</div>
""", unsafe_allow_html=True)

days    = _days(st.session_state.get("date_window", "Last 7 days"))
has_key = bool(st.session_state.get("gemini_key", ""))
ai_on   = st.session_state.get("ai_on", False)

ch, df   = _load()
all_anom = _anomalies(df)
recent   = get_recent_anomalies(all_anom, days=days)
period   = compute_period_summary(df, days=days)
ch_sum   = compute_channel_summary(ch, days=days)
ch_ins   = generate_channel_insights(ch_sum)

# ── AI page summary ───────────────────────────────────────────────────────────

if has_key and ch_ins:
    sum_key = f"marketing_summary_{days}"
    if sum_key not in st.session_state:
        with st.spinner("Generating AI summary..."):
            try:
                from ai.gemini import generate_recommendations
                all_ins = compile_all_insights(period, recent, ch_sum)
                st.session_state[sum_key] = generate_recommendations(
                    insights=all_ins, channel_summary=ch_sum, period_summary=period,
                )
            except Exception:
                st.session_state[sum_key] = None
    narrative = st.session_state.get(sum_key)
    if narrative:
        st.markdown(f"""
        <div class="ai-strip">
            <div class="ai-strip-label">AI recommendations</div>
            {narrative}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="ai-placeholder">
        Add a Gemini API key in the sidebar to unlock AI channel recommendations.
    </div>
    """, unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────

total_spend  = ch_sum["spend"].sum() if not ch_sum.empty else 0
total_rev    = ch_sum["revenue_attributed"].sum() if not ch_sum.empty else 0
total_clicks = ch_sum["clicks"].sum() if not ch_sum.empty else 0
total_conv   = ch_sum["conversions"].sum() if not ch_sum.empty else 0
overall_roas = (total_rev / total_spend) if total_spend > 0 else None
overall_cr   = (total_conv / total_clicks) if total_clicks > 0 else None
spend_period = period.get("spend", {})
roas_period  = period.get("roas", {})

k1, k2, k3, k4 = st.columns(4, gap="small")
with k1:
    _kpi_card("Total spend", _fmt_currency(total_spend),
              _fmt_pct(spend_period.get("pct_change")),
              _delta_class(spend_period.get("pct_change"), inverse=True),
              hint=f"Last {days} days")
with k2:
    _kpi_card("Revenue attributed", _fmt_currency(total_rev),
              _fmt_pct(roas_period.get("pct_change")),
              _delta_class(roas_period.get("pct_change")),
              hint=f"Last {days} days")
with k3:
    _kpi_card("Overall ROAS", _fmt_x(overall_roas),
              _fmt_pct(roas_period.get("pct_change")),
              _delta_class(roas_period.get("pct_change")),
              hint="Revenue / spend")
with k4:
    _kpi_card("Overall conversion rate",
              f"{overall_cr*100:.2f}%" if overall_cr else "N/A",
              f"{total_conv:,.0f} conversions", "kpi-delta-neu",
              hint=f"From {total_clicks:,.0f} clicks")

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ── Channel performance table ─────────────────────────────────────────────────

st.markdown(f'<div class="section-header">Channel performance · last {days} days</div>',
            unsafe_allow_html=True)

if ch_sum.empty:
    st.info("No channel data available for this window.")
else:
    ch_display = ch_sum.copy()
    ch_display["ROAS"]            = ch_display["roas"].apply(lambda v: _fmt_x(v) if pd.notna(v) else "N/A")
    ch_display["Conversion rate"] = ch_display["conversion_rate"].apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "N/A")
    ch_display["Spend"]           = ch_display["spend"].apply(_fmt_currency)
    ch_display["Revenue"]         = ch_display["revenue_attributed"].apply(_fmt_currency)
    ch_display["Clicks"]          = ch_display["clicks"].apply(lambda v: f"{v:,.0f}")
    ch_display["Conversions"]     = ch_display["conversions"].apply(lambda v: f"{v:,.0f}")
    ch_display["Channel"]         = ch_display["channel"]
    st.dataframe(
        ch_display[["Channel", "Spend", "Revenue", "ROAS",
                    "Clicks", "Conversions", "Conversion rate"]],
        use_container_width=True, hide_index=True,
    )

# ── Spend vs Revenue chart ────────────────────────────────────────────────────

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
chart_l, chart_r = st.columns(2, gap="small")

with chart_l:
    st.markdown('<div class="chart-title">Spend vs revenue attributed</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-sub">By channel · last {days} days</div>', unsafe_allow_html=True)
    if not ch_sum.empty:
        ch_sorted = ch_sum.sort_values("spend", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ch_sorted["channel"], x=ch_sorted["spend"], name="Spend",
            orientation="h", marker_color="#BFDBFE",
            hovertemplate="<b>%{y}</b><br>Spend: $%{x:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=ch_sorted["channel"], x=ch_sorted["revenue_attributed"], name="Revenue",
            orientation="h", marker_color="#1D4ED8",
            hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=8, b=0),
            plot_bgcolor="white", paper_bgcolor="white", barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=11)),
            xaxis=dict(showgrid=True, gridcolor="#F3F4F6", tickfont=dict(size=11),
                       tickprefix="$", tickformat=",.0f"),
            yaxis=dict(showgrid=False, tickfont=dict(size=11)),
            hovermode="y unified", bargap=0.25, bargroupgap=0.1,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    _ai_chart_insight(
        key="marketing_spend_rev",
        context=(f"Horizontal grouped bar chart comparing spend vs revenue attributed "
                 f"by channel over the last {days} days. "
                 f"Total spend: {_fmt_currency(total_spend)}, "
                 f"total revenue: {_fmt_currency(total_rev)}, "
                 f"overall ROAS: {_fmt_x(overall_roas)}."),
        ai_on=ai_on, has_key=has_key,
    )

with chart_r:
    st.markdown('<div class="chart-title">ROAS by channel</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-sub">Last {days} days · 1.0x = break even</div>',
                unsafe_allow_html=True)
    if not ch_sum.empty:
        roas_df = ch_sum.dropna(subset=["roas"]).sort_values("roas", ascending=True)
        bar_colors = [
            "#E24B4A" if v < 1.5 else "#BFDBFE" if v < 3.0 else "#1D4ED8"
            for v in roas_df["roas"]
        ]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=roas_df["channel"], x=roas_df["roas"], orientation="h",
            marker_color=bar_colors,
            hovertemplate="<b>%{y}</b><br>ROAS: %{x:.2f}x<extra></extra>",
        ))
        fig2.add_vline(x=1.0, line_dash="dash", line_color="#E5E7EB", line_width=1.5,
                       annotation_text="Break even", annotation_font_size=10,
                       annotation_font_color="#9CA3AF")
        fig2.update_layout(
            height=300, margin=dict(l=0, r=0, t=8, b=0),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                       tickfont=dict(size=11), ticksuffix="x"),
            yaxis=dict(showgrid=False, tickfont=dict(size=11)),
            hovermode="y unified", bargap=0.3,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    _ai_chart_insight(
        key="marketing_roas",
        context=(f"Horizontal bar chart of ROAS by channel over the last {days} days. "
                 f"Bars below 1.5x are red, 1.5-3x light blue, above 3x dark blue."),
        ai_on=ai_on, has_key=has_key,
    )

# ── Daily spend trend ─────────────────────────────────────────────────────────

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="chart-title">Daily spend by channel</div>', unsafe_allow_html=True)
st.markdown(f'<div class="chart-sub">Last {days} days · stacked</div>', unsafe_allow_html=True)

recent_ch = ch.copy()
recent_ch["date"] = pd.to_datetime(recent_ch["date"])
cutoff    = recent_ch["date"].max() - pd.Timedelta(days=days - 1)
recent_ch = recent_ch[recent_ch["date"] >= cutoff]
channels  = recent_ch["channel"].unique()
palette   = ["#1D4ED8", "#3B82F6", "#93C5FD", "#BFDBFE", "#DBEAFE", "#EFF6FF"]

fig3 = go.Figure()
for i, channel in enumerate(channels):
    ch_data = recent_ch[recent_ch["channel"] == channel].sort_values("date")
    fig3.add_trace(go.Bar(
        x=ch_data["date"], y=ch_data["spend"], name=channel,
        marker_color=palette[i % len(palette)],
        hovertemplate=f"<b>%{{x|%b %d}}</b><br>{channel}: $%{{y:,.0f}}<extra></extra>",
    ))
fig3.update_layout(
    height=300, margin=dict(l=0, r=0, t=8, b=0),
    plot_bgcolor="white", paper_bgcolor="white", barmode="stack",
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11)),
    xaxis=dict(showgrid=False, tickfont=dict(size=11), tickformat="%b %d"),
    yaxis=dict(showgrid=True, gridcolor="#F3F4F6", tickfont=dict(size=11),
               tickprefix="$", tickformat=",.0f"),
    hovermode="x unified", bargap=0.25,
)
st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
_ai_chart_insight(
    key="marketing_daily_spend",
    context=(f"Stacked bar chart of daily spend by channel over the last {days} days. "
             f"Channels: {', '.join(channels)}. "
             f"Total spend in window: {_fmt_currency(total_spend)}."),
    ai_on=ai_on, has_key=has_key,
)
