"""
Pulse — Reports
AI-generated daily and weekly insight reports with live preview and download.
"""

import sys
import os
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
    load_daily_metrics,
    enrich_daily_metrics,
    compute_period_summary,
    compute_top_products,
    compute_channel_summary,
    load_channel_metrics,
    load_product_sales,
)
from analytics.anomaly import detect_all_anomalies, get_recent_anomalies
from analytics.insights import compile_all_insights
from reports.daily_report import generate_report

# ── Helpers ───────────────────────────────────────────────────────────────────

def _days(window: str) -> int:
    return {"Last 7 days": 7, "Last 14 days": 14,
            "Last 30 days": 30, "Last 90 days": 90}.get(window, 7)


def _fmt_currency(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"${v:,.2f}"


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
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

days    = _days(st.session_state.get("date_window", "Last 7 days"))
has_key = bool(st.session_state.get("gemini_key", ""))
ai_on   = st.session_state.get("ai_on", False)

df, ch, prod = _load()
all_anom     = _anomalies(df)
recent_anom  = get_recent_anomalies(all_anom, days=days)
period       = compute_period_summary(df, days=days)
ch_sum       = compute_channel_summary(ch, days=days)
top_prod     = compute_top_products(prod, n=5, days=days)
insights     = compile_all_insights(period, recent_anom, ch_sum)

# ── Controls ──────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Report settings</div>',
    unsafe_allow_html=True,
)

ctrl_l, ctrl_r = st.columns([1, 2], gap="small")

with ctrl_l:
    report_type = st.selectbox(
        "Report type",
        ["Daily insight report", "Weekly summary report"],
        index=0,
    )
    report_date = st.date_input(
        "Reporting date",
        value=date.today(),
    )

with ctrl_r:
    st.markdown("""
    <div style="background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:12px;
                padding:1rem 1.2rem; height:100%;">
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
            AI narrative requires a Gemini API key. Without one, the report
            uses rule-based findings only — all sections still populate.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ── Generate button ───────────────────────────────────────────────────────────

gen_col, _ = st.columns([1, 3])
with gen_col:
    generate_clicked = st.button(
        "Generate report",
        type="primary",
        use_container_width=True,
    )

if generate_clicked:
    report_key = f"report_{report_type}_{report_date}_{days}"
    with st.spinner("Generating report..."):
        try:
            # Pass DataFrame directly — generate_report uses itertuples internally
            top_prod_df = top_prod if not top_prod.empty else pd.DataFrame()

            # ── Generate AI strings if key is available ────────────────────
            ai_narrative           = None
            ai_anomaly_explanation = None
            ai_recommendations     = None

            if has_key:
                try:
                    from ai.gemini import (
                        generate_daily_narrative,
                        generate_anomaly_explanation,
                        generate_recommendations,
                        generate_weekly_report_narrative,
                    )

                    is_weekly = report_type == "Weekly summary report"

                    if is_weekly:
                        top_prod_list = (
                            top_prod.to_dict("records")
                            if not top_prod.empty else []
                        )
                        ai_narrative = generate_weekly_report_narrative(
                            insights=insights,
                            period_summary=period,
                            top_products=top_prod_list,
                        )
                    else:
                        ai_narrative = generate_daily_narrative(
                            insights=insights,
                            period_summary=period,
                            date_label=f"the last {days} days",
                        )

                    # Use most recent anomaly for deep-dive explanation
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
                        insights=insights,
                        channel_summary=ch_sum,
                        period_summary=period,
                    )

                except Exception as ai_err:
                    st.warning(f"AI generation partial failure: {ai_err}")

            # ── Assemble report ────────────────────────────────────────────
            report_text = generate_report(
                period_summary=period,
                insights=insights,
                anomalies=recent_anom,
                channel_summary=ch_sum,
                top_products=top_prod_df,
                report_date=report_date,
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
    st.markdown(
        '<div class="section-header">Report preview</div>',
        unsafe_allow_html=True,
    )

    prev_l, prev_r = st.columns([2.2, 1], gap="small")

    with prev_l:
        sections = report_text.strip().split("\n\n")

        for block in sections:
            lines = block.strip().split("\n")
            if not lines:
                continue

            first = lines[0].strip()

            # Separator lines
            if set(first) == {"-"} and len(first) > 10:
                st.markdown(
                    "<hr style='border:none; border-top:0.5px solid #E5E7EB;"
                    " margin:0.5rem 0;'>",
                    unsafe_allow_html=True,
                )
                continue

            # Section headings — all caps
            if first.isupper() and len(first) > 3 and not first.startswith(" "):
                st.markdown(
                    f"<div style='font-size:0.72rem; font-weight:600; "
                    f"color:#374151; text-transform:uppercase; "
                    f"letter-spacing:0.08em; margin:1rem 0 0.4rem;'>"
                    f"{first}</div>",
                    unsafe_allow_html=True,
                )
                body_lines = lines[1:]
            else:
                body_lines = lines

            for line in body_lines:
                line = line.strip()
                if not line:
                    continue

                # Numbered list items
                if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
                    st.markdown(
                        f"<div style='font-size:0.8rem; color:#374151; "
                        f"line-height:1.7; padding:0.15rem 0 0.15rem 0.8rem; "
                        f"border-left:2px solid #E5E7EB; margin-bottom:0.25rem;'>"
                        f"{line}</div>",
                        unsafe_allow_html=True,
                    )

                # Key: value lines
                elif ":" in line and len(line.split(":")[0]) < 30:
                    parts = line.split(":", 1)
                    st.markdown(
                        f"<div style='font-size:0.78rem; color:#374151; "
                        f"line-height:1.7; margin-bottom:0.1rem;'>"
                        f"<span style='font-weight:600; color:#111827;'>"
                        f"{parts[0]}:</span>{parts[1]}</div>",
                        unsafe_allow_html=True,
                    )

                # Anomaly flags — lines starting with [
                elif line.startswith("["):
                    is_up   = "[UP]" in line
                    is_down = "[DOWN]" in line
                    color   = "#15803D" if is_up else "#B91C1C" if is_down else "#92400E"
                    bg      = "#DCFCE7" if is_up else "#FEE2E2" if is_down else "#FEF3C7"
                    st.markdown(
                        f"<div style='font-size:0.78rem; color:{color}; "
                        f"background:{bg}; border-radius:6px; "
                        f"padding:0.3rem 0.7rem; margin-bottom:0.25rem; "
                        f"line-height:1.6;'>{line}</div>",
                        unsafe_allow_html=True,
                    )

                # Regular prose
                else:
                    st.markdown(
                        f"<div style='font-size:0.8rem; color:#374151; "
                        f"line-height:1.75; margin-bottom:0.3rem;'>"
                        f"{line}</div>",
                        unsafe_allow_html=True,
                    )

    with prev_r:
        # ── Report metadata card ──────────────────────────────────────────────
        n_anom    = len(recent_anom) if not recent_anom.empty else 0
        n_insight = len(insights)
        rev_data  = period.get("revenue", {})
        rev_chg   = rev_data.get("pct_change")

        st.markdown(f"""
        <div style="background:#FFFFFF; border:0.5px solid #E5E7EB;
                    border-radius:12px; padding:1rem 1.2rem;
                    margin-bottom:0.6rem;">
            <div style="font-size:0.65rem; font-weight:600; color:#9CA3AF;
                        text-transform:uppercase; letter-spacing:0.08em;
                        margin-bottom:0.8rem;">Report summary</div>
            <div style="display:flex; flex-direction:column; gap:0.55rem;">
                <div style="display:flex; justify-content:space-between;
                            font-size:0.78rem; border-bottom:0.5px solid #F3F4F6;
                            padding-bottom:0.4rem;">
                    <span style="color:#6B7280;">Type</span>
                    <span style="color:#111827; font-weight:500;">{report_type_}</span>
                </div>
                <div style="display:flex; justify-content:space-between;
                            font-size:0.78rem; border-bottom:0.5px solid #F3F4F6;
                            padding-bottom:0.4rem;">
                    <span style="color:#6B7280;">Date</span>
                    <span style="color:#111827; font-weight:500;">{report_date.strftime('%b %d, %Y')}</span>
                </div>
                <div style="display:flex; justify-content:space-between;
                            font-size:0.78rem; border-bottom:0.5px solid #F3F4F6;
                            padding-bottom:0.4rem;">
                    <span style="color:#6B7280;">Window</span>
                    <span style="color:#111827; font-weight:500;">Last {days} days</span>
                </div>
                <div style="display:flex; justify-content:space-between;
                            font-size:0.78rem; border-bottom:0.5px solid #F3F4F6;
                            padding-bottom:0.4rem;">
                    <span style="color:#6B7280;">Anomalies</span>
                    <span style="color:#111827; font-weight:500;">{n_anom} detected</span>
                </div>
                <div style="display:flex; justify-content:space-between;
                            font-size:0.78rem; border-bottom:0.5px solid #F3F4F6;
                            padding-bottom:0.4rem;">
                    <span style="color:#6B7280;">Insights</span>
                    <span style="color:#111827; font-weight:500;">{n_insight} findings</span>
                </div>
                <div style="display:flex; justify-content:space-between;
                            font-size:0.78rem;">
                    <span style="color:#6B7280;">Revenue WoW</span>
                    <span style="color:{'#15803D' if rev_chg and rev_chg > 0 else '#B91C1C' if rev_chg and rev_chg < 0 else '#6B7280'};
                                font-weight:500;">{_fmt_pct(rev_chg)}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── AI status card ────────────────────────────────────────────────────
        ai_status = "Enabled" if has_key else "Disabled"
        ai_color  = "#15803D" if has_key else "#9CA3AF"
        ai_bg     = "#DCFCE7" if has_key else "#F3F4F6"

        st.markdown(f"""
        <div style="background:#FFFFFF; border:0.5px solid #E5E7EB;
                    border-radius:12px; padding:1rem 1.2rem;
                    margin-bottom:0.6rem;">
            <div style="font-size:0.65rem; font-weight:600; color:#9CA3AF;
                        text-transform:uppercase; letter-spacing:0.08em;
                        margin-bottom:0.6rem;">AI narrative</div>
            <span style="background:{ai_bg}; color:{ai_color};
                         font-size:0.7rem; font-weight:600;
                         padding:3px 10px; border-radius:20px;">
                {ai_status}
            </span>
            <div style="font-size:0.72rem; color:#9CA3AF; margin-top:0.6rem;
                        line-height:1.6;">
                {"AI-generated executive summary and recommendations are included in this report."
                 if has_key else
                 "Add a Gemini API key in the sidebar to include AI narrative in the report."}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Download button ───────────────────────────────────────────────────
        file_name = f"pulse_report_{report_date.strftime('%Y-%m-%d')}.txt"
        st.download_button(
            label="Download report (.txt)",
            data=report_text,
            file_name=file_name,
            mime="text/plain",
            use_container_width=True,
        )

        # ── Raw text toggle ───────────────────────────────────────────────────
        with st.expander("View raw text"):
            st.code(report_text, language=None)

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#FFFFFF; border:0.5px solid #E5E7EB;
                border-radius:12px; padding:2.5rem 2rem;
                text-align:center; margin-top:0.5rem;">
        <div style="font-size:0.9rem; font-weight:600; color:#374151;
                    margin-bottom:0.4rem;">No report generated yet</div>
        <div style="font-size:0.8rem; color:#9CA3AF; line-height:1.7;
                    max-width:360px; margin:0 auto;">
            Select a report type and date above, then click
            <strong>Generate report</strong> to build a preview.
            The report will appear here before you download it.
        </div>
    </div>
    """, unsafe_allow_html=True)
