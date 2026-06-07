"""
Pulse — Reports
AI-generated daily and weekly insight reports with live preview and download.
"""

import sys
import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
APP  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

from utils import inject_styles
inject_styles()

from analytics.metrics import (
    load_daily_metrics, enrich_daily_metrics, compute_period_summary,
    compute_top_products, compute_channel_summary,
    load_channel_metrics, load_product_sales,
)
from analytics.anomaly import detect_all_anomalies, get_recent_anomalies
from analytics.insights import compile_all_insights
from reports.daily_report import generate_report

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_currency(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"${v:,.2f}"

def _fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.1f}%"

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
<div class="page-title">Reports</div>
<div class="page-sub">Generate, preview, and download AI-written insight reports</div>
""", unsafe_allow_html=True)

start_date = st.session_state.get("start_date", datetime.date(2023, 12, 2))
end_date   = st.session_state.get("end_date",   datetime.date(2023, 12, 31))
days       = st.session_state.get("days", 30)
has_key    = bool(st.session_state.get("gemini_key", ""))

df, ch, prod = _load()
all_anom     = _anomalies(df)

df["date"]  = pd.to_datetime(df["date"])
recent_anom = get_recent_anomalies(all_anom, start_date=start_date, end_date=end_date)
period      = compute_period_summary(df, days=days, start_date=start_date, end_date=end_date)
ch_sum      = compute_channel_summary(ch, days=days, start_date=start_date, end_date=end_date)
top_prod    = compute_top_products(prod, n=5, days=days, start_date=start_date, end_date=end_date)
insights    = compile_all_insights(period, recent_anom, ch_sum)

# ── Controls ──────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Report settings</div>', unsafe_allow_html=True)

ctrl_l, ctrl_r = st.columns([1, 2], gap="small")

with ctrl_l:
    report_type = st.selectbox(
        "Report type",
        ["Daily insight report", "Weekly summary report"],
        index=0,
    )
    report_date = st.date_input("Reporting date", value=end_date)

with ctrl_r:
    st.markdown(f"""
    <div style="background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:12px;
                padding:1rem 1.2rem;">
        <div style="font-size:0.7rem; font-weight:600; color:#9CA3AF;
                    text-transform:uppercase; letter-spacing:0.08em;
                    margin-bottom:0.6rem;">What this report includes</div>
        <div style="font-size:0.8rem; color:#374151; line-height:1.8;">
            Executive summary &nbsp;&middot;&nbsp;
            Key metric changes &nbsp;&middot;&nbsp;
            Anomaly log<br>
            Channel performance &nbsp;&middot;&nbsp;
            Top products &nbsp;&middot;&nbsp;
            AI recommendations
        </div>
        <div style="font-size:0.72rem; color:#9CA3AF; margin-top:0.6rem;">
            Window: {start_date.strftime('%b %d, %Y')} — {end_date.strftime('%b %d, %Y')} ({days}d)
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ── Generate button ───────────────────────────────────────────────────────────

gen_col, _ = st.columns([1, 3])
with gen_col:
    generate_clicked = st.button("Generate report", type="primary", use_container_width=True)

if generate_clicked:
    report_key = f"report_{report_type}_{report_date}_{start_date}_{end_date}"
    with st.spinner("Generating report..."):
        try:
            top_prod_df = top_prod if not top_prod.empty else pd.DataFrame()

            ai_narrative           = None
            ai_anomaly_explanation = None
            ai_recommendations     = None

            if has_key:
                try:
                    from ai.gemini import (
                        generate_daily_narrative, generate_anomaly_explanation,
                        generate_recommendations, generate_weekly_report_narrative,
                    )
                    is_weekly = report_type == "Weekly summary report"
                    if is_weekly:
                        top_prod_list = top_prod.to_dict("records") if not top_prod.empty else []
                        ai_narrative = generate_weekly_report_narrative(
                            insights=insights, period_summary=period,
                            top_products=top_prod_list,
                        )
                    else:
                        ai_narrative = generate_daily_narrative(
                            insights=insights, period_summary=period,
                            date_label=f"{start_date.strftime('%b %d')} to {end_date.strftime('%b %d, %Y')}",
                        )
                    if not recent_anom.empty:
                        top_anom = recent_anom.iloc[0]
                        ai_anomaly_explanation = generate_anomaly_explanation(
                            metric=top_anom["metric"],
                            direction=top_anom.get("direction", ""),
                            deviation_pct=top_anom.get("deviation_pct", 0),
                            anomaly_type=top_anom.get("anomaly_type", ""),
                            period_summary=period,
                        )
                    ai_recommendations = generate_recommendations(
                        insights=insights, channel_summary=ch_sum, period_summary=period,
                    )
                except Exception as ai_err:
                    st.warning(f"AI generation partial failure: {ai_err}")

            report_text = generate_report(
                period_summary=period, insights=insights,
                anomalies=recent_anom, channel_summary=ch_sum,
                top_products=top_prod_df, report_date=report_date,
                ai_narrative=ai_narrative,
                ai_anomaly_explanation=ai_anomaly_explanation,
                ai_recommendations=ai_recommendations,
                save=False,
            )
            st.session_state[report_key]          = report_text
            st.session_state["active_report_key"]  = report_key
            st.session_state["active_report_type"] = report_type
        except Exception as e:
            st.error(f"Report generation failed: {e}")

# ── Preview ───────────────────────────────────────────────────────────────────

active_key = st.session_state.get("active_report_key")

if active_key and active_key in st.session_state:
    report_text  = st.session_state[active_key]
    report_type_ = st.session_state.get("active_report_type", "Report")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Report preview</div>', unsafe_allow_html=True)

    prev_l, prev_r = st.columns([2.2, 1], gap="small")

    with prev_l:
        for block in report_text.strip().split("\n\n"):
            lines = block.strip().split("\n")
            if not lines:
                continue
            first = lines[0].strip()
            if set(first) == {"-"} and len(first) > 10:
                st.markdown(
                    "<hr style='border:none; border-top:0.5px solid #E5E7EB; margin:0.5rem 0;'>",
                    unsafe_allow_html=True)
                continue
            if first.isupper() and len(first) > 3 and not first.startswith(" "):
                st.markdown(
                    f"<div style='font-size:0.72rem; font-weight:600; color:#374151; "
                    f"text-transform:uppercase; letter-spacing:0.08em; margin:1rem 0 0.4rem;'>"
                    f"{first}</div>", unsafe_allow_html=True)
                body_lines = lines[1:]
            else:
                body_lines = lines
            for line in body_lines:
                line = line.strip()
                if not line:
                    continue
                if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
                    st.markdown(
                        f"<div style='font-size:0.8rem; color:#374151; line-height:1.7; "
                        f"padding:0.15rem 0 0.15rem 0.8rem; border-left:2px solid #E5E7EB; "
                        f"margin-bottom:0.25rem;'>{line}</div>", unsafe_allow_html=True)
                elif ":" in line and len(line.split(":")[0]) < 30:
                    parts = line.split(":", 1)
                    st.markdown(
                        f"<div style='font-size:0.78rem; color:#374151; line-height:1.7; "
                        f"margin-bottom:0.1rem;'><span style='font-weight:600; color:#111827;'>"
                        f"{parts[0]}:</span>{parts[1]}</div>", unsafe_allow_html=True)
                elif line.startswith("["):
                    is_up   = "[UP]" in line
                    is_down = "[DOWN]" in line
                    color   = "#15803D" if is_up else "#B91C1C" if is_down else "#92400E"
                    bg      = "#DCFCE7" if is_up else "#FEE2E2" if is_down else "#FEF3C7"
                    st.markdown(
                        f"<div style='font-size:0.78rem; color:{color}; background:{bg}; "
                        f"border-radius:6px; padding:0.3rem 0.7rem; margin-bottom:0.25rem; "
                        f"line-height:1.6;'>{line}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='font-size:0.8rem; color:#374151; line-height:1.75; "
                        f"margin-bottom:0.3rem;'>{line}</div>", unsafe_allow_html=True)

    with prev_r:
        n_anom    = len(recent_anom) if not recent_anom.empty else 0
        n_insight = len(insights)
        rev_data  = period.get("revenue", {})
        rev_chg   = rev_data.get("pct_change")

        st.markdown(f"""
        <div style="background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:12px;
                    padding:1rem 1.2rem; margin-bottom:0.6rem;">
            <div style="font-size:0.65rem; font-weight:600; color:#9CA3AF;
                        text-transform:uppercase; letter-spacing:0.08em;
                        margin-bottom:0.8rem;">Report summary</div>
            <div style="display:flex; flex-direction:column; gap:0.55rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;
                            border-bottom:0.5px solid #F3F4F6; padding-bottom:0.4rem;">
                    <span style="color:#6B7280;">Type</span>
                    <span style="color:#111827; font-weight:500;">{report_type_}</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;
                            border-bottom:0.5px solid #F3F4F6; padding-bottom:0.4rem;">
                    <span style="color:#6B7280;">Window</span>
                    <span style="color:#111827; font-weight:500;">
                        {start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;
                            border-bottom:0.5px solid #F3F4F6; padding-bottom:0.4rem;">
                    <span style="color:#6B7280;">Anomalies</span>
                    <span style="color:#111827; font-weight:500;">{n_anom} detected</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;
                            border-bottom:0.5px solid #F3F4F6; padding-bottom:0.4rem;">
                    <span style="color:#6B7280;">Insights</span>
                    <span style="color:#111827; font-weight:500;">{n_insight} findings</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                    <span style="color:#6B7280;">Revenue WoW</span>
                    <span style="color:{'#15803D' if rev_chg and rev_chg > 0 else '#B91C1C' if rev_chg and rev_chg < 0 else '#6B7280'};
                                font-weight:500;">{_fmt_pct(rev_chg)}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ai_status = "Enabled" if has_key else "Disabled"
        ai_color  = "#15803D" if has_key else "#9CA3AF"
        ai_bg     = "#DCFCE7" if has_key else "#F3F4F6"
        st.markdown(f"""
        <div style="background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:12px;
                    padding:1rem 1.2rem; margin-bottom:0.6rem;">
            <div style="font-size:0.65rem; font-weight:600; color:#9CA3AF;
                        text-transform:uppercase; letter-spacing:0.08em;
                        margin-bottom:0.6rem;">AI narrative</div>
            <span style="background:{ai_bg}; color:{ai_color}; font-size:0.7rem;
                         font-weight:600; padding:3px 10px; border-radius:20px;">
                {ai_status}
            </span>
            <div style="font-size:0.72rem; color:#9CA3AF; margin-top:0.6rem; line-height:1.6;">
                {"AI-generated executive summary and recommendations included."
                 if has_key else
                 "Add a Gemini API key to include AI narrative."}
            </div>
        </div>
        """, unsafe_allow_html=True)

        file_name = f"pulse_report_{report_date.strftime('%Y-%m-%d')}.txt"
        st.download_button(
            label="Download report (.txt)", data=report_text,
            file_name=file_name, mime="text/plain", use_container_width=True,
        )
        with st.expander("View raw text"):
            st.code(report_text, language=None)

else:
    st.markdown("""
    <div style="background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:12px;
                padding:2.5rem 2rem; text-align:center; margin-top:0.5rem;">
        <div style="font-size:0.9rem; font-weight:600; color:#374151; margin-bottom:0.4rem;">
            No report generated yet</div>
        <div style="font-size:0.8rem; color:#9CA3AF; line-height:1.7;
                    max-width:360px; margin:0 auto;">
            Select a report type and date above, then click
            <strong>Generate report</strong> to build a preview.
        </div>
    </div>
    """, unsafe_allow_html=True)
