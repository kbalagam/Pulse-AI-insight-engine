"""
Pulse — AI-powered insight engine.
DM Sans font, consistent KPI tiles, always-visible AI expanders,
no funnel, fixed chart margins, consistent product layout, top-N filter.
"""

import os
import sys
import textwrap
import requests
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics.metrics import (
    load_daily_metrics, load_channel_metrics, load_product_sales,
    enrich_daily_metrics, compute_period_summary,
    compute_top_products, compute_channel_summary,
)
from analytics.anomaly import detect_all_anomalies, get_recent_anomalies
from analytics.insights import compile_all_insights
from reports.daily_report import generate_report

st.set_page_config(
    page_title="Pulse — AI Insight Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

CRIMSON      = "#9B1C1C"
CRIMSON_DEEP = "#7F1D1D"
RED_ACCENT   = "#DC2626"
RED_LIGHT    = "#FEF2F2"
CHARCOAL     = "#1C1C1C"
CHARCOAL_MID = "#374151"
GRAY         = "#6B7280"
GRAY_LIGHT   = "#F3F4F6"
BORDER       = "#E5E7EB"
WHITE        = "#FFFFFF"
BG           = "#F9FAFB"
TEXT_DARK    = "#111827"
TEXT_MID     = "#6B7280"
TEXT_LIGHT   = "#9CA3AF"
GREEN        = "#16A34A"
GREEN_LIGHT  = "#DCFCE7"
AMBER        = "#D97706"
AMBER_LIGHT  = "#FEF3C7"
BLUE         = "#1D4ED8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], button, input, textarea, select, p, div, span, label {{
  font-family: 'DM Sans', sans-serif !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
  background: {CHARCOAL} !important;
}}
section[data-testid="stSidebar"] > div {{
  background: {CHARCOAL} !important;
}}
section[data-testid="stSidebar"] * {{
  color: #E5E7EB !important;
}}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stDateInput input,
section[data-testid="stSidebar"] .stTextArea textarea {{
  background: rgba(255,255,255,0.1) !important;
  border: 1px solid rgba(255,255,255,0.2) !important;
  border-radius: 7px !important;
  color: #F9FAFB !important;
}}
section[data-testid="stSidebar"] label {{
  color: #D1D5DB !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
}}
section[data-testid="stSidebar"] hr {{
  border-color: rgba(255,255,255,0.12) !important;
}}
section[data-testid="stSidebar"] .stButton > button {{
  background: {CRIMSON} !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}}

/* ── KPI cards — fixed height so all tiles are uniform ── */
.kpi-card {{
  background: {WHITE};
  border-radius: 12px;
  padding: 1rem 1.1rem 0.9rem 1.1rem;
  border: 1px solid {BORDER};
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  height: 150px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-sizing: border-box;
  margin-bottom: 0.5rem;
}}
.kpi-icon  {{ font-size: 1.1rem; line-height: 1; }}
.kpi-label {{
  font-size: 0.67rem; color: {TEXT_MID}; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem;
}}
.kpi-value {{
  font-size: 1.5rem; font-weight: 700; color: {TEXT_DARK}; line-height: 1.15;
}}
.kpi-bottom {{ display: flex; flex-direction: column; gap: 0.15rem; }}
.kpi-delta-up   {{ display:inline-block; font-size:0.68rem; font-weight:600; color:{GREEN};      background:{GREEN_LIGHT};  padding:2px 7px; border-radius:20px; }}
.kpi-delta-down {{ display:inline-block; font-size:0.68rem; font-weight:600; color:{RED_ACCENT}; background:{RED_LIGHT};    padding:2px 7px; border-radius:20px; }}
.kpi-delta-neu  {{ display:inline-block; font-size:0.68rem; font-weight:600; color:{GRAY};       background:{GRAY_LIGHT};   padding:2px 7px; border-radius:20px; }}
.kpi-hint {{ font-size: 0.65rem; color: {TEXT_LIGHT}; }}

/* ── Section ── */
.section-header {{ font-size:0.94rem; font-weight:700; color:{TEXT_DARK}; margin:1.1rem 0 0.2rem 0; }}
.section-sub    {{ font-size:0.76rem; color:{TEXT_MID}; margin-bottom:0.55rem; }}

/* ── AI ── */
.ai-box {{
  background:{WHITE}; border:1px solid {BORDER};
  border-left:4px solid {CRIMSON}; border-radius:10px;
  padding:0.9rem 1.1rem; line-height:1.7;
  color:{TEXT_DARK}; font-size:0.85rem; margin:0.4rem 0 0.7rem 0;
}}
.ai-placeholder {{
  background:{GRAY_LIGHT}; border:1.5px dashed {BORDER};
  border-radius:10px; padding:0.8rem 1rem;
  color:{TEXT_MID}; font-size:0.81rem;
  margin:0.4rem 0 0.6rem 0; text-align:center; line-height:1.6;
}}

/* ── Insight cards ── */
.insight-card {{
  background:{WHITE}; border-radius:10px; padding:0.72rem 0.9rem;
  border:1px solid {BORDER}; margin-bottom:0.38rem;
  border-left:4px solid {CRIMSON};
}}
.insight-high   {{ border-left-color:{RED_ACCENT}; }}
.insight-medium {{ border-left-color:{AMBER}; }}
.insight-low    {{ border-left-color:{GREEN}; }}
.badge {{ display:inline-block; font-size:0.61rem; font-weight:700; padding:2px 6px; border-radius:20px; text-transform:uppercase; letter-spacing:0.05em; margin-right:4px; }}
.badge-high   {{ background:{RED_LIGHT};   color:{RED_ACCENT}; }}
.badge-medium {{ background:{AMBER_LIGHT}; color:{AMBER}; }}
.badge-low    {{ background:{GREEN_LIGHT}; color:{GREEN}; }}

/* ── Anomaly stat cards ── */
.anomaly-stat-card {{
  background:{WHITE}; border-radius:12px; padding:0.85rem 0.95rem;
  border:1px solid {BORDER}; text-align:center;
}}
.anom-val   {{ font-size:1.65rem; font-weight:700; }}
.anom-label {{ font-size:0.71rem; color:{TEXT_MID}; margin-top:0.12rem; }}

/* ── Callout ── */
.callout {{
  background:{RED_LIGHT}; border-radius:10px; padding:0.72rem 0.95rem;
  font-size:0.81rem; color:{CRIMSON_DEEP}; margin-bottom:0.7rem;
  border:1px solid #FECACA;
}}

/* ── Hero ── */
.hero {{
  background:linear-gradient(135deg,{CRIMSON_DEEP} 0%,{CHARCOAL} 100%);
  border-radius:14px; padding:1.3rem 1.6rem; color:white; margin-bottom:1rem;
}}
.hero-title {{ font-size:1.2rem; font-weight:700; }}
.hero-sub   {{ font-size:0.78rem; opacity:0.7; margin-top:0.1rem; }}
.hero-stats {{ display:flex; gap:2rem; margin-top:0.75rem; flex-wrap:wrap; }}
.hero-stat-val   {{ font-size:1.1rem; font-weight:700; }}
.hero-stat-label {{ font-size:0.63rem; opacity:0.62; text-transform:uppercase; letter-spacing:0.05em; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
  gap:3px; background:{WHITE}; border-radius:9px;
  padding:3px; border:1px solid {BORDER}; margin-bottom:0.75rem;
}}
.stTabs [data-baseweb="tab"] {{
  border-radius:6px; font-size:0.77rem; font-weight:600;
  color:{GRAY}; padding:5px 11px;
}}
.stTabs [aria-selected="true"] {{
  background:{CRIMSON} !important; color:white !important;
}}

/* ── Buttons ── */
.stButton > button {{
  background:{CRIMSON}; color:white; border:none;
  border-radius:8px; font-weight:600; transition:opacity 0.2s;
}}
.stButton > button:hover {{ opacity:0.84; }}

/* ── Product tile ── */
.prod-tile {{
  background:{WHITE}; border-radius:9px;
  padding:0.6rem 0.85rem; border:1px solid {BORDER};
  margin-bottom:0.32rem; box-shadow:0 1px 2px rgba(0,0,0,0.03);
}}

/* ── Landing ── */
.landing-hero {{
  background:linear-gradient(135deg,{CRIMSON_DEEP} 0%,{CHARCOAL} 100%);
  border-radius:16px; padding:2.2rem 1.8rem; color:white;
  margin-bottom:1.6rem; text-align:center;
}}
.landing-title {{ font-size:2.2rem; font-weight:700; margin-bottom:0.35rem; }}
.landing-tag   {{ font-size:0.92rem; opacity:0.72; margin-bottom:1rem; }}
.vs-row {{ display:flex; gap:0.55rem; padding:0.45rem 0; border-bottom:1px solid {BORDER}; font-size:0.8rem; }}
.vs-bad  {{ color:{RED_ACCENT}; flex:1; }}
.vs-good {{ color:{GREEN}; flex:1; }}

.block-container {{ padding-top:1rem; }}
.stDataFrame {{ border-radius:9px; overflow:hidden; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KPI_META = {
    "revenue":         {"label":"revenue",        "icon":"💰","hint":"Total sales per day",                        "fmt":lambda v:f"${v:,.0f}",   "good":"up"},
    "orders":          {"label":"orders",          "icon":"🛍️","hint":"Number of purchases made",                 "fmt":lambda v:f"{v:,.0f}",     "good":"up"},
    "aov":             {"label":"aov",             "icon":"🧾","hint":"Average spend per order",                  "fmt":lambda v:f"${v:,.2f}",    "good":"up"},
    "conversion_rate": {"label":"conversion_rate", "icon":"🎯","hint":"% of clicks resulting in a purchase",     "fmt":lambda v:f"{v*100:.2f}%", "good":"up"},
    "roas":            {"label":"roas",            "icon":"📣","hint":"Revenue per $1 of ad spend",              "fmt":lambda v:f"{v:.2f}x",     "good":"up"},
    "spend":           {"label":"spend",           "icon":"💸","hint":"Daily marketing spend",                   "fmt":lambda v:f"${v:,.0f}",    "good":"neutral"},
}

METRIC_LABELS = {
    "revenue":"Revenue","orders":"Orders","aov":"AOV",
    "conversion_rate":"Conversion Rate","cac":"CAC","roas":"ROAS","spend":"Spend",
}

CHANNEL_COLORS = {
    "Paid Search":CRIMSON,"Social":CHARCOAL_MID,"Email":GRAY,
    "Affiliate":GREEN,"Display":AMBER,"Organic / Direct":BLUE,
}
CATEGORY_COLORS = {
    "Electronics":CRIMSON,"Fashion":CHARCOAL_MID,"Sports":GREEN,
    "Home":AMBER,"Beauty":RED_ACCENT,"Grocery":BLUE,
}
CATEGORY_ICONS = {
    "Electronics":"💻","Fashion":"👗","Sports":"🏃",
    "Home":"🏠","Beauty":"✨","Grocery":"🛒",
}
ANOMALY_METRIC_COLORS = {
    "revenue":CRIMSON,"orders":CHARCOAL_MID,"aov":AMBER,
    "conversion_rate":GREEN,"cac":RED_ACCENT,"roas":BLUE,"spend":GRAY,
}

# ---------------------------------------------------------------------------
# Chart layout — generous margins prevent cropping
# ---------------------------------------------------------------------------

def _base_layout(height=320, x_title="", y_title=""):
    return dict(
        height=height,
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        font=dict(color=TEXT_DARK, family="DM Sans", size=11),
        xaxis=dict(
            title=x_title, gridcolor=GRAY_LIGHT, showgrid=True,
            zeroline=False, linecolor=BORDER, automargin=True,
            title_font=dict(size=11, color=TEXT_MID),
        ),
        yaxis=dict(
            title=y_title, gridcolor=GRAY_LIGHT, showgrid=True,
            zeroline=False, linecolor=BORDER, automargin=True,
            title_font=dict(size=11, color=TEXT_MID),
        ),
        margin=dict(l=64, r=36, t=36, b=64),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER, font=dict(family="DM Sans")),
    )

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_and_enrich():
    df   = load_daily_metrics()
    df   = enrich_daily_metrics(df)
    ch   = load_channel_metrics()
    prod = load_product_sales()
    return df, ch, prod

@st.cache_data(show_spinner="Running anomaly detection...")
def get_all_anomalies(_df):
    return detect_all_anomalies(_df)

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def _api_key():
    return os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

def _has_key():
    return bool(_api_key())

def _gemini_post(prompt: str, max_tokens: int = 400) -> str:
    key = _api_key()
    if not key:
        return ""
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
    try:
        r = requests.post(
            f"{url}?key={key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
            },
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return f"API error {r.status_code}"
    except Exception as e:
        return f"Error: {e}"

def call_gemini(fn: str, *args, **kwargs):
    if not _has_key():
        return None
    os.environ["GEMINI_API_KEY"] = _api_key()
    try:
        from ai.gemini import (
            generate_daily_narrative, generate_anomaly_explanation,
            generate_recommendations, generate_weekly_report_narrative,
        )
        fmap = {
            "narrative":       generate_daily_narrative,
            "anomaly":         generate_anomaly_explanation,
            "recommendations": generate_recommendations,
            "weekly":          generate_weekly_report_narrative,
        }
        return fmap[fn](*args, **kwargs)
    except Exception as e:
        st.warning(f"AI unavailable: {e}")
        return None

def ai_placeholder(message: str = "AI Insight available"):
    st.markdown(f"""
    <div class="ai-placeholder">
      🔒 <strong>{message}</strong><br>
      <span style="font-size:0.75rem;">
        Add your free Gemini API key in the sidebar to unlock this.
        Get one at <a href="https://aistudio.google.com" target="_blank"
        style="color:{CRIMSON};">aistudio.google.com</a>.
      </span>
    </div>""", unsafe_allow_html=True)

def graph_ai_expander(graph_id: str, context: str, ai_on: bool):
    """Always visible below every chart regardless of key or toggle state."""
    with st.expander("🤖 AI Insight for this chart"):
        if not _has_key():
            ai_placeholder("Add your Gemini API key to generate insight for this chart")
            return
        if not ai_on:
            st.markdown(f"""
            <div class="ai-placeholder">
              💡 <strong>AI Insights are ready for this chart.</strong><br>
              <span style="font-size:0.75rem;">
                Turn on <strong>"Show AI Insights on graphs"</strong> in the sidebar,
                then click Generate.
              </span>
            </div>""", unsafe_allow_html=True)
            return
        btn_k = f"ai_btn_{graph_id}"
        res_k = f"ai_res_{graph_id}"
        if st.button("Generate Insight", key=btn_k):
            with st.spinner("Analysing..."):
                result = _gemini_post(
                    "You are a business analyst. Give 2-3 concise actionable "
                    "observations about this chart in plain English. No bullet points. "
                    f"Do not start with 'I'.\n\nChart context: {context}",
                    max_tokens=300,
                )
            st.session_state[res_k] = result
        if res_k in st.session_state:
            st.markdown(f'<div class="ai-box">{st.session_state[res_k]}</div>',
                        unsafe_allow_html=True)
        else:
            st.caption("Click Generate Insight to analyse this chart.")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "show_landing" not in st.session_state:
    st.session_state["show_landing"] = True

# ---------------------------------------------------------------------------
# LANDING PAGE
# ---------------------------------------------------------------------------

if st.session_state["show_landing"]:
    st.markdown(f"""
    <div class="landing-hero">
      <div class="landing-title">⚡ Pulse</div>
      <div class="landing-tag">AI-Powered Insight Engine for Any Data-Driven Domain</div>
      <div style="font-size:0.84rem;opacity:0.68;max-width:540px;margin:0 auto;">
        Pulse automatically analyses your business data, detects anomalies,
        and delivers plain-English recommendations — no analyst required.
      </div>
    </div>""", unsafe_allow_html=True)

    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("### What Pulse Does")
        st.markdown(f"""
        <div style="background:{WHITE};border-radius:12px;padding:1rem;border:1px solid {BORDER};margin-bottom:0.8rem;">
          <div style="font-size:0.84rem;color:{TEXT_DARK};line-height:1.7;">
            Built for <strong>executives, analysts, and domain leads</strong>
            who need fast, clear answers from their data.<br><br>
            Pulse automatically:
            <ul style="margin:0.4rem 0 0 1rem;color:{TEXT_MID};font-size:0.81rem;">
              <li>Detects meaningful metric changes and anomalies</li>
              <li>Ranks findings by business impact</li>
              <li>Generates plain-English explanations and recommendations</li>
              <li>Exports shareable daily insight reports</li>
            </ul>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("### Who It's For")
        for icon, role, desc in [
            ("📊","Revenue & Finance","Daily revenue, AOV, order trends, ROAS"),
            ("📣","Marketing Teams","Channel ROI, CAC, campaign anomalies"),
            ("📦","Product Managers","Category shifts, top products, units sold"),
            ("🏢","Executives","High-level summaries and daily reports"),
            ("🔬","Data Scientists","Anomaly detection, trend signals, deep analytics"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:0.55rem;align-items:flex-start;
                 padding:0.42rem 0;border-bottom:1px solid {BORDER};">
              <div style="font-size:0.95rem;">{icon}</div>
              <div>
                <div style="font-weight:600;font-size:0.81rem;">{role}</div>
                <div style="font-size:0.74rem;color:{TEXT_MID};">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with lc2:
        st.markdown("### AI Features")
        for icon, title, desc in [
            ("✨","Daily Narrative","Executive summary of performance in plain English."),
            ("🚨","Anomaly Explanations","When a metric spikes or drops, Pulse explains why and what to do."),
            ("💡","Recommendations","3–5 prioritised, specific actions ranked by revenue impact."),
            ("📊","Per-Graph Insights","Every chart has an AI Insight button for targeted analysis."),
            ("💬","Ask Anything","Type any question and get a direct, data-grounded answer."),
            ("📄","AI Reports","Full written reports with AI narratives — ready to share."),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:0.55rem;align-items:flex-start;
                 padding:0.47rem 0;border-bottom:1px solid {BORDER};">
              <div style="font-size:0.95rem;min-width:1.3rem;">{icon}</div>
              <div>
                <div style="font-weight:600;font-size:0.81rem;">{title}</div>
                <div style="font-size:0.74rem;color:{TEXT_MID};line-height:1.5;">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Pulse vs Static Dashboards")
        for bad, good in [
            ("❌ Shows numbers, no explanation","✅ Explains what changed and why"),
            ("❌ Analyst needed to find anomalies","✅ Anomalies detected automatically"),
            ("❌ Hours to build a report","✅ One-click AI report in seconds"),
            ("❌ No recommended actions","✅ Specific next steps every session"),
        ]:
            st.markdown(f'<div class="vs-row"><div class="vs-bad">{bad}</div>'
                        f'<div class="vs-good">{good}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    _lc1, _lc2, _lc3 = st.columns([2,1,2])
    with _lc2:
        if st.button("Open Dashboard →", type="primary", use_container_width=True):
            st.session_state["show_landing"] = False
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

df_full, ch_full, prod_full = load_and_enrich()
DATA_MIN = df_full["date"].min().date()
DATA_MAX = df_full["date"].max().date()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style='padding:0.3rem 0 0.6rem 0;'>
      <div style='font-size:1.05rem;font-weight:700;color:#FFFFFF;'>⚡ Pulse</div>
      <div style='font-size:0.67rem;color:#9CA3AF;margin-top:1px;'>AI Insight Engine</div>
    </div>""", unsafe_allow_html=True)

    if st.button("← About Pulse", use_container_width=False):
        st.session_state["show_landing"] = True
        st.rerun()

    st.markdown("---")
    st.markdown("**📅 Date Range**")
    preset = st.selectbox("Quick select", [
        "Last 30 days","Last 90 days","Last 6 months",
        "This year","All time","Custom range",
    ], index=0)

    if   preset == "Last 30 days":  d_start,d_end = DATA_MAX-timedelta(29),  DATA_MAX
    elif preset == "Last 90 days":  d_start,d_end = DATA_MAX-timedelta(89),  DATA_MAX
    elif preset == "Last 6 months": d_start,d_end = DATA_MAX-timedelta(179), DATA_MAX
    elif preset == "This year":     d_start,d_end = date(DATA_MAX.year,1,1), DATA_MAX
    elif preset == "All time":      d_start,d_end = DATA_MIN, DATA_MAX
    else:                           d_start,d_end = None, None

    if preset == "Custom range":
        d_start = st.date_input("From", value=DATA_MAX-timedelta(29),
                                 min_value=DATA_MIN, max_value=DATA_MAX)
        d_end   = st.date_input("To",   value=DATA_MAX,
                                 min_value=DATA_MIN, max_value=DATA_MAX)
        if d_start > d_end:
            st.error("Start must be before end.")
            d_start, d_end = DATA_MAX-timedelta(29), DATA_MAX
    else:
        d_start = max(d_start, DATA_MIN)
        d_end   = min(d_end,   DATA_MAX)

    # Date range display — always fully visible
    st.markdown(
        f"<div style='font-size:0.76rem;color:#F3F4F6;font-weight:500;"
        f"background:rgba(255,255,255,0.1);border-radius:6px;"
        f"padding:5px 9px;margin-top:4px;border:1px solid rgba(255,255,255,0.15);'>"
        f"📅 {d_start.strftime('%b %d, %Y')} → {d_end.strftime('%b %d, %Y')}"
        f"</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🤖 AI Features**")
    gemini_key = st.text_input("Gemini API Key", type="password",
                                placeholder="Paste key to enable AI",
                                help="Free at aistudio.google.com")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        st.markdown(
            "<div style='font-size:0.71rem;background:rgba(22,163,74,0.22);"
            "border-radius:7px;padding:4px 9px;margin-top:3px;color:#D1FAE5;'>"
            "✅ AI active</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='font-size:0.71rem;color:#9CA3AF;margin-top:3px;'>"
            "Add key to unlock AI features</div>", unsafe_allow_html=True)

    ai_toggle = st.toggle(
        "Show AI Insights on graphs",
        value=False,
        help="Expanders always show — toggle enables the Generate button inside",
    )

    st.markdown("---")
    st.markdown("**💬 Ask Anything**")
    st.markdown(
        "<div style='font-size:0.68rem;color:#9CA3AF;margin-bottom:0.35rem;'>"
        "Ask any question about your current data</div>", unsafe_allow_html=True)

    sidebar_q = st.text_area("Q", height=72,
                              placeholder="e.g. Why did revenue drop?\nWhich channel has best ROI?",
                              label_visibility="collapsed",
                              key="sidebar_question")
    if st.button("Ask Pulse 🤖", use_container_width=True,
                  disabled=not _has_key(), key="sidebar_ask"):
        if sidebar_q.strip():
            st.session_state["sidebar_answer_pending"] = sidebar_q.strip()
        else:
            st.warning("Type a question first.")
    if not _has_key():
        st.markdown(
            "<div style='font-size:0.67rem;color:#6B7280;margin-top:-0.2rem;'>"
            "Requires API key above</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        with st.spinner("Re-running ETL..."):
            from etl.pipeline import run as run_etl
            run_etl()
            st.cache_data.clear()
        st.success("Done — reload page.")

    st.markdown(
        f"<div style='font-size:0.63rem;color:#6B7280;margin-top:0.7rem;'>"
        f"Dataset: {DATA_MIN.strftime('%b %Y')} – {DATA_MAX.strftime('%b %Y')}"
        f"</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filtered data
# ---------------------------------------------------------------------------

ts_start  = pd.Timestamp(d_start)
ts_end    = pd.Timestamp(d_end)
days_sel  = (d_end - d_start).days + 1
cmp_days  = min(7, days_sel)

df   = df_full[(df_full["date"]>=ts_start)&(df_full["date"]<=ts_end)].copy()
ch   = ch_full[(ch_full["date"]>=ts_start)&(ch_full["date"]<=ts_end)].copy()
prod = prod_full[(prod_full["date"]>=ts_start)&(prod_full["date"]<=ts_end)].copy()

summary       = compute_period_summary(df, days=cmp_days)
all_anomalies = get_all_anomalies(df_full)
rec_anomalies = get_recent_anomalies(all_anomalies, days=min(days_sel, 60))
ch_summary    = compute_channel_summary(ch, days=cmp_days)
insights      = compile_all_insights(summary, rec_anomalies, ch_summary)

total_rev    = df["revenue"].sum()
total_orders = int(df["orders"].sum()) if "orders" in df.columns else 0
high_count   = sum(1 for i in insights if i.get("impact") == "high")
period_label = f"{d_start.strftime('%b %d')} – {d_end.strftime('%b %d, %Y')}"

# ---------------------------------------------------------------------------
# Sidebar Ask Anything — process
# ---------------------------------------------------------------------------

if "sidebar_answer_pending" in st.session_state:
    q = st.session_state.pop("sidebar_answer_pending")
    m_lines, ch_lines, ins_lines = [], [], []
    for metric, data in summary.items():
        if metric == "period_days" or not isinstance(data, dict): continue
        pct = data.get("pct_change"); recent = data.get("recent_avg")
        if pct is None or recent is None: continue
        m_lines.append(f"  {metric}: {recent} (WoW:{pct*100:+.1f}%)")
    try:
        for _, row in ch_summary.iterrows():
            roas = f"{row['roas']:.2f}x" if pd.notna(row.get("roas")) else "N/A"
            ch_lines.append(f"  {row['channel']}: spend=${row['spend']:.0f}, ROAS={roas}")
    except: pass
    ins_lines = [f"  [{i.get('impact','').upper()}] {i.get('finding','')}" for i in insights[:6]]
    prompt = textwrap.dedent(f"""
        You are a data analytics advisor.
        Answer using only the data below. Be specific and concise (3-5 sentences max).
        USER QUESTION: {q}
        METRICS: {chr(10).join(m_lines)}
        CHANNELS: {chr(10).join(ch_lines)}
        INSIGHTS: {chr(10).join(ins_lines)}
    """).strip()
    with st.spinner("Pulse is thinking..."):
        answer = _gemini_post(prompt, max_tokens=400)
    st.session_state["sidebar_answer"] = (q, answer)

if "sidebar_answer" in st.session_state:
    q_shown, a_shown = st.session_state["sidebar_answer"]
    st.markdown(f"""
    <div style='background:{GRAY_LIGHT};border-radius:9px;padding:0.8rem 0.95rem;
         margin-bottom:0.9rem;border-left:3px solid {CRIMSON};'>
      <div style='font-size:0.69rem;font-weight:700;color:{CRIMSON};
           text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem;'>
        You asked:</div>
      <div style='font-size:0.8rem;color:{TEXT_DARK};margin-bottom:0.45rem;
           font-style:italic;'>"{q_shown}"</div>
      <div style='font-size:0.8rem;color:{TEXT_DARK};line-height:1.6;'>{a_shown}</div>
    </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="hero">
  <div class="hero-title">⚡ Pulse — Insight Engine</div>
  <div class="hero-sub">{period_label} · {days_sel} days</div>
  <div class="hero-stats">
    <div class="hero-stat"><div class="hero-stat-val">${total_rev:,.0f}</div><div class="hero-stat-label">Total Revenue</div></div>
    <div class="hero-stat"><div class="hero-stat-val">{total_orders:,}</div><div class="hero-stat-label">Total Orders</div></div>
    <div class="hero-stat"><div class="hero-stat-val">{len(insights)}</div><div class="hero-stat-label">Insights</div></div>
    <div class="hero-stat"><div class="hero-stat-val">{high_count}</div><div class="hero-stat-label">Need Attention</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview","🚨 Anomalies","📣 Channels",
    "📦 Products","🤖 AI Insights","📄 Report",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">KPI Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Last {cmp_days} days vs prior {cmp_days} days</div>',
                unsafe_allow_html=True)

    kpi_cols = st.columns(len(KPI_META))
    for col, (metric, meta) in zip(kpi_cols, KPI_META.items()):
        data    = summary.get(metric, {})
        recent  = data.get("recent_avg") if isinstance(data, dict) else None
        pct     = data.get("pct_change") if isinstance(data, dict) else None
        val_str = meta["fmt"](recent) if recent is not None else "—"
        good    = meta["good"]
        if pct is None:
            delta = '<span class="kpi-delta-neu">No comparison</span>'
        else:
            arrow   = "▲" if pct > 0 else "▼"
            lbl     = f"{arrow} {abs(pct)*100:.1f}% vs prior"
            is_good = (pct>0 and good=="up") or (pct<0 and good=="down")
            cls     = "kpi-delta-up" if is_good else ("kpi-delta-down" if good!="neutral" else "kpi-delta-neu")
            delta   = f'<span class="{cls}">{lbl}</span>'
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div>
                <div class="kpi-icon">{meta['icon']}</div>
                <div class="kpi-label">{meta['label']}</div>
                <div class="kpi-value">{val_str}</div>
              </div>
              <div class="kpi-bottom">
                {delta}
                <div class="kpi-hint">{meta['hint']}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # Revenue trend
    st.markdown('<div class="section-header">Revenue Trend</div>', unsafe_allow_html=True)
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=df["date"], y=df["revenue"], mode="lines", name="Daily Revenue",
        line=dict(color=CRIMSON, width=2.5),
        fill="tozeroy", fillcolor="rgba(155,28,28,0.07)",
    ))
    if "revenue_roll_mean" in df.columns:
        fig_rev.add_trace(go.Scatter(
            x=df["date"], y=df["revenue_roll_mean"], mode="lines",
            name="7-day avg", line=dict(color=CHARCOAL_MID, width=1.8, dash="dot"),
        ))
    fig_rev.update_layout(**_base_layout(height=300, x_title="Date", y_title="Revenue (USD)"))
    fig_rev.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_rev, use_container_width=True)
    graph_ai_expander("rev_trend",
        f"Daily revenue {d_start} to {d_end}. Total ${total_rev:,.0f} over {days_sel} days.",
        ai_toggle)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">conversion_rate</div>', unsafe_allow_html=True)
        fig_cr = go.Figure()
        fig_cr.add_trace(go.Scatter(
            x=df["date"], y=df["conversion_rate"]*100, mode="lines",
            line=dict(color=GREEN, width=2.5),
            fill="tozeroy", fillcolor="rgba(22,163,74,0.07)", name="conversion_rate",
        ))
        fig_cr.update_layout(**_base_layout(height=260, x_title="Date", y_title="Conversion Rate (%)"))
        fig_cr.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_cr, use_container_width=True)
        cr_val = (summary.get("conversion_rate",{}).get("recent_avg") or 0)*100
        graph_ai_expander("conv_rate",
            f"Conversion rate. Avg: {cr_val:.2f}%. Period: {d_start} to {d_end}.", ai_toggle)

    with col_b:
        st.markdown('<div class="section-header">New vs Returning Customers</div>', unsafe_allow_html=True)
        fig_cust = go.Figure()
        fig_cust.add_trace(go.Scatter(
            x=df["date"], y=df["new_customers"], mode="lines",
            name="new_customers", line=dict(color=CRIMSON, width=2),
            stackgroup="one", fillcolor="rgba(155,28,28,0.25)",
        ))
        fig_cust.add_trace(go.Scatter(
            x=df["date"], y=df["returning_customers"], mode="lines",
            name="returning_customers", line=dict(color=CHARCOAL_MID, width=2),
            stackgroup="one", fillcolor="rgba(55,65,81,0.25)",
        ))
        fig_cust.update_layout(**_base_layout(height=260, x_title="Date", y_title="Customer Count"))
        st.plotly_chart(fig_cust, use_container_width=True)
        graph_ai_expander("cust_mix",
            f"New vs returning customers. Period: {d_start} to {d_end}.", ai_toggle)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-header">aov</div>', unsafe_allow_html=True)
        fig_aov = go.Figure()
        fig_aov.add_trace(go.Scatter(
            x=df["date"], y=df["aov"], mode="lines",
            line=dict(color=AMBER, width=2.5),
            fill="tozeroy", fillcolor="rgba(217,119,6,0.07)", name="aov",
        ))
        fig_aov.update_layout(**_base_layout(height=260, x_title="Date", y_title="AOV (USD)"))
        fig_aov.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_aov, use_container_width=True)
        aov_val = summary.get("aov",{}).get("recent_avg") or 0
        graph_ai_expander("aov",
            f"AOV trend. Avg: ${aov_val:.2f}. Period: {d_start} to {d_end}.", ai_toggle)

    with col_d:
        st.markdown('<div class="section-header">roas</div>', unsafe_allow_html=True)
        fig_roas = go.Figure()
        fig_roas.add_trace(go.Scatter(
            x=df["date"], y=df["roas"], mode="lines",
            line=dict(color=RED_ACCENT, width=2.5),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.07)", name="roas",
        ))
        fig_roas.add_hline(y=2, line_dash="dot", line_color=CHARCOAL_MID,
                           annotation_text="Min target 2x",
                           annotation_font_color=CHARCOAL_MID)
        fig_roas.update_layout(**_base_layout(height=260, x_title="Date", y_title="ROAS (x)"))
        fig_roas.update_yaxes(ticksuffix="x")
        st.plotly_chart(fig_roas, use_container_width=True)
        roas_val = summary.get("roas",{}).get("recent_avg") or 0
        graph_ai_expander("roas",
            f"ROAS trend. Avg: {roas_val:.2f}x. Target 2x. Period: {d_start} to {d_end}.", ai_toggle)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Anomaly Summary</div>', unsafe_allow_html=True)

    most_affected="—"; worst_dev=0.0; worst_date="—"
    if not rec_anomalies.empty:
        mc = rec_anomalies["metric"].value_counts()
        most_affected = METRIC_LABELS.get(mc.index[0], mc.index[0]) if len(mc) else "—"
        wr = rec_anomalies.loc[rec_anomalies["deviation_pct"].idxmax()]
        worst_dev  = wr["deviation_pct"]
        wd = wr["date"]
        worst_date = wd.date().strftime("%b %d, %Y") if hasattr(wd,"date") else str(wd)

    sc1,sc2,sc3,sc4 = st.columns(4)
    for col,val,label,color in [
        (sc1, len(rec_anomalies),       "Total Anomalies",         CRIMSON),
        (sc2, most_affected,            "Most Affected Metric",    CHARCOAL_MID),
        (sc3, f"{worst_dev*100:.0f}%",  "Largest Deviation",       RED_ACCENT),
        (sc4, worst_date,               "Date of Worst Anomaly",   AMBER),
    ]:
        with col:
            st.markdown(f"""
            <div class="anomaly-stat-card">
              <div class="anom-val" style="color:{color};">{val}</div>
              <div class="anom-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)

    if rec_anomalies.empty:
        st.markdown(f"""
        <div style='text-align:center;padding:2rem;background:{WHITE};
             border-radius:12px;border:1px solid {BORDER};'>
          <div style='font-size:1.7rem;'>✅</div>
          <div style='font-weight:700;font-size:0.93rem;color:{TEXT_DARK};margin-top:0.3rem;'>
            No anomalies detected</div>
          <div style='color:{TEXT_MID};font-size:0.78rem;margin-top:0.12rem;'>
            All metrics within normal range.</div>
        </div>""", unsafe_allow_html=True)
    else:
        anom_df = rec_anomalies.copy()
        anom_df["date"]         = pd.to_datetime(anom_df["date"])
        anom_df["metric_label"] = anom_df["metric"].map(lambda m: METRIC_LABELS.get(m,m))
        anom_df["severity"]     = anom_df["deviation_pct"].apply(lambda d:"High" if d>=0.20 else "Medium")

        cb1,cb2 = st.columns([3,2])
        with cb1:
            st.markdown('<div class="section-header">Weekly Anomaly Frequency</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Red = high severity · Amber = medium severity</div>', unsafe_allow_html=True)
            anom_df["week"] = anom_df["date"].dt.to_period("W").apply(lambda p: p.start_time)
            weekly = anom_df.groupby(["week","severity"]).size().reset_index(name="count")
            fig_bar = go.Figure()
            for sev,color in [("High",RED_ACCENT),("Medium",AMBER)]:
                sub = weekly[weekly["severity"]==sev]
                if not sub.empty:
                    fig_bar.add_trace(go.Bar(x=sub["week"],y=sub["count"],
                        name=sev,marker_color=color,opacity=0.88))
            fig_bar.update_layout(**_base_layout(height=280,x_title="Week",y_title="Anomaly Count"),
                                   barmode="stack")
            st.plotly_chart(fig_bar, use_container_width=True)
            graph_ai_expander("anom_bar",
                f"Weekly anomaly frequency. Total: {len(anom_df)}. Most affected: {most_affected}.",
                ai_toggle)

        with cb2:
            st.markdown('<div class="section-header">Metric Stability</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Most volatile metrics first</div>', unsafe_allow_html=True)
            stability = (
                anom_df.groupby("metric_label")
                .agg(count=("deviation_pct","count"), worst=("deviation_pct","max"))
                .reset_index().sort_values("count", ascending=False)
            )
            for _, row in stability.iterrows():
                bb = RED_LIGHT   if row["worst"]>=0.20 else AMBER_LIGHT
                bc = RED_ACCENT  if row["worst"]>=0.20 else AMBER
                bl = "🔴 High"  if row["worst"]>=0.20 else "🟡 Medium"
                st.markdown(f"""
                <div style='display:flex;align-items:center;justify-content:space-between;
                     padding:0.48rem 0.72rem;background:{WHITE};border-radius:8px;
                     border:1px solid {BORDER};margin-bottom:0.32rem;'>
                  <div>
                    <div style='font-weight:600;font-size:0.8rem;'>{row['metric_label']}</div>
                    <div style='font-size:0.68rem;color:{TEXT_MID};'>
                      {int(row['count'])} anomalies · worst {row['worst']*100:.0f}%</div>
                  </div>
                  <div style='background:{bb};color:{bc};padding:2px 7px;
                       border-radius:20px;font-size:0.64rem;font-weight:700;'>{bl}</div>
                </div>""", unsafe_allow_html=True)

        cd1,cd2 = st.columns([2,3])
        with cd1:
            st.markdown('<div class="section-header">Share by Metric</div>', unsafe_allow_html=True)
            donut_data = anom_df["metric_label"].value_counts().reset_index()
            donut_data.columns = ["metric","count"]
            donut_colors = [
                ANOMALY_METRIC_COLORS.get(
                    {v:k for k,v in METRIC_LABELS.items()}.get(m,m), CRIMSON)
                for m in donut_data["metric"]
            ]
            fig_donut = go.Figure(go.Pie(
                labels=donut_data["metric"], values=donut_data["count"],
                hole=0.55,
                marker=dict(colors=donut_colors, line=dict(color=WHITE,width=2)),
                textinfo="label+percent", textfont=dict(size=11),
            ))
            fig_donut.update_layout(
                height=270, paper_bgcolor=WHITE,
                font=dict(family="DM Sans"),
                margin=dict(l=10,r=10,t=10,b=10), showlegend=False,
                annotations=[dict(text=f"<b>{len(anom_df)}</b><br>total",
                    x=0.5,y=0.5,font_size=13,font_color=TEXT_DARK,showarrow=False)],
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            graph_ai_expander("anom_donut",
                f"Anomaly share by metric. Most volatile: {most_affected}.", ai_toggle)

        with cd2:
            st.markdown('<div class="section-header">Anomaly Detail</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Click any row to expand and get an AI explanation.</div>', unsafe_allow_html=True)
            for _, row in anom_df.sort_values("date",ascending=False).head(12).iterrows():
                is_up     = "up" in str(row.get("direction",""))
                dev       = row.get("deviation_pct",0)
                sc        = RED_ACCENT if dev>=0.20 else AMBER
                sb        = RED_LIGHT  if dev>=0.20 else AMBER_LIGHT
                icon      = "↑" if is_up else "↓"
                date_str  = row["date"].date().strftime("%b %d, %Y") if hasattr(row["date"],"date") else str(row["date"])
                atype     = str(row.get("anomaly_type","")).replace("_"," ").title()
                with st.expander(f"{icon} {row['metric_label']}  ·  {dev*100:.0f}% deviation  ·  {date_str}",
                                  expanded=False):
                    c1,c2 = st.columns([1,2])
                    with c1:
                        st.markdown(f"""
                        <div style='background:{sb};border-radius:9px;padding:0.8rem;border:1px solid {sc};'>
                          <div style='font-weight:700;font-size:0.85rem;'>{row['metric_label']}</div>
                          <div style='font-size:1.4rem;font-weight:700;color:{sc};margin:0.25rem 0;'>
                            {icon} {dev*100:.1f}%</div>
                          <div style='font-size:0.68rem;color:{TEXT_MID};'>{atype}</div>
                          <div style='font-size:0.68rem;color:{TEXT_MID};margin-top:0.18rem;'>{date_str}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        if _has_key():
                            bk=f"anom_btn_{row.name}"; rk=f"anom_res_{row.name}"
                            if st.button("🤖 Explain this anomaly", key=bk):
                                with st.spinner("Analysing..."):
                                    result = call_gemini("anomaly",metric=row["metric"],
                                        direction=str(row.get("direction","")),
                                        deviation_pct=dev,
                                        anomaly_type=str(row.get("anomaly_type","")),
                                        period_summary=summary)
                                st.session_state[rk] = result or "No result."
                            if rk in st.session_state:
                                st.markdown(
                                    f'<div class="ai-box"><strong>🤖 Likely causes & actions:</strong>'
                                    f'<br><br>{st.session_state[rk].replace(chr(10),"<br>")}</div>',
                                    unsafe_allow_html=True)
                        else:
                            ai_placeholder("AI explanation available for this anomaly")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CHANNELS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Channel Performance</div>', unsafe_allow_html=True)

    CHART_H = 320
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-sub">Revenue attributed by channel</div>', unsafe_allow_html=True)
        colors_ch = [CHANNEL_COLORS.get(c,CRIMSON) for c in ch_summary["channel"]]
        fig_ch = go.Figure(go.Bar(
            x=ch_summary["channel"], y=ch_summary["revenue_attributed"],
            marker_color=colors_ch,
            text=[f"${v:,.0f}" for v in ch_summary["revenue_attributed"]],
            textposition="outside",
        ))
        fig_ch.update_layout(**_base_layout(height=CHART_H, x_title="Channel", y_title="Revenue Attributed (USD)"))
        fig_ch.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_ch, use_container_width=True)
        top_ch = ch_summary.sort_values("revenue_attributed",ascending=False).iloc[0]["channel"]
        graph_ai_expander("ch_revenue", f"Revenue by channel. Top: {top_ch}.", ai_toggle)

    with col2:
        ch_roas = ch_summary.dropna(subset=["roas"]).sort_values("roas")
        st.markdown('<div class="section-sub">roas by channel — target ≥ 2x</div>', unsafe_allow_html=True)
        if not ch_roas.empty:
            colors_roas = [CHANNEL_COLORS.get(c,CRIMSON) for c in ch_roas["channel"]]
            fig_rc = go.Figure(go.Bar(
                x=ch_roas["roas"], y=ch_roas["channel"],
                orientation="h", marker_color=colors_roas,
                text=[f"{v:.2f}x" for v in ch_roas["roas"]], textposition="outside",
            ))
            fig_rc.add_vline(x=2, line_dash="dot", line_color=RED_ACCENT,
                              annotation_text="Target 2x", annotation_font_color=RED_ACCENT)
            fig_rc.update_layout(**_base_layout(height=CHART_H, x_title="ROAS (x)", y_title="Channel"))
            fig_rc.update_xaxes(ticksuffix="x")
            st.plotly_chart(fig_rc, use_container_width=True)
            graph_ai_expander("ch_roas",
                f"ROAS by channel. Best: {ch_roas.iloc[-1]['channel']} at {ch_roas.iloc[-1]['roas']:.2f}x.",
                ai_toggle)
        else:
            st.info("Not enough spend data for ROAS in this period.")

    st.markdown('<div class="section-header">Channel Detail Table</div>', unsafe_allow_html=True)
    ch_tbl = ch_summary.copy()
    ch_tbl["spend"]              = ch_tbl["spend"].apply(lambda x: f"${x:,.0f}")
    ch_tbl["revenue_attributed"] = ch_tbl["revenue_attributed"].apply(lambda x: f"${x:,.0f}")
    ch_tbl["roas"]               = ch_tbl["roas"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    ch_tbl["conversion_rate"]    = ch_tbl["conversion_rate"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    ch_tbl.columns = ["Channel","Spend","Clicks","Conversions","Revenue Attributed","ROAS","Conversion Rate"]
    st.dataframe(ch_tbl, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Channel Trends Over Time</div>', unsafe_allow_html=True)
    tc1,tc2,tc3 = st.columns([1,1,2])
    with tc1:
        ch_ds = st.date_input("From##ch", value=d_start, min_value=DATA_MIN, max_value=DATA_MAX, key="ch_ds")
    with tc2:
        ch_de = st.date_input("To##ch",   value=d_end,   min_value=DATA_MIN, max_value=DATA_MAX, key="ch_de")
    with tc3:
        ch_opts = ch_full["channel"].unique().tolist()
        sel_chs = st.multiselect("Channels", ch_opts,
                                  default=[c for c in ["Social","Paid Search","Email"] if c in ch_opts])
    if ch_ds > ch_de:
        st.error("Start must be before end.")
    elif sel_chs:
        ch_tr = ch_full[
            (ch_full["channel"].isin(sel_chs)) &
            (ch_full["date"]>=pd.Timestamp(ch_ds)) &
            (ch_full["date"]<=pd.Timestamp(ch_de))
        ]
        fig_tr = go.Figure()
        for cn in sel_chs:
            cd = ch_tr[ch_tr["channel"]==cn]
            fig_tr.add_trace(go.Scatter(
                x=cd["date"], y=cd["revenue_attributed"],
                mode="lines", name=cn,
                line=dict(color=CHANNEL_COLORS.get(cn,CRIMSON), width=2.5),
            ))
        fig_tr.update_layout(**_base_layout(height=300, x_title="Date", y_title="Revenue Attributed (USD)"))
        fig_tr.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_tr, use_container_width=True)
        graph_ai_expander("ch_trend",
            f"Revenue over time for {', '.join(sel_chs)}. Period: {ch_ds} to {ch_de}.", ai_toggle)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Product Performance</div>', unsafe_allow_html=True)

    PROD_CHART_H = 320
    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown('<div class="section-sub">Revenue by category</div>', unsafe_allow_html=True)
        cat_rev = (prod.groupby("category")["revenue"].sum()
                   .reset_index().sort_values("revenue", ascending=False))
        cat_rev["icon"]  = cat_rev["category"].map(lambda c: CATEGORY_ICONS.get(c,"📦"))
        cat_rev["label"] = cat_rev.apply(lambda r: f"{r['icon']} {r['category']}", axis=1)
        colors_cat = [CATEGORY_COLORS.get(c, CRIMSON) for c in cat_rev["category"]]
        fig_cat = go.Figure(go.Bar(
            x=cat_rev["label"], y=cat_rev["revenue"],
            marker_color=colors_cat,
            text=[f"${v:,.0f}" for v in cat_rev["revenue"]],
            textposition="outside",
        ))
        fig_cat.update_layout(**_base_layout(height=PROD_CHART_H,
                              x_title="Category", y_title="Revenue (USD)"))
        fig_cat.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_cat, use_container_width=True)
        graph_ai_expander("cat_rev",
            f"Revenue by category. Top: {cat_rev.iloc[0]['category']} at ${cat_rev.iloc[0]['revenue']:,.0f}.",
            ai_toggle)

    with pc2:
        st.markdown('<div class="section-sub">Top products in selected category — sorted high to low</div>',
                    unsafe_allow_html=True)
        sel_cat = st.selectbox("Drill into category",
                                sorted(prod["category"].unique()), key="drill_cat")
        top_in_cat = (
            prod[prod["category"]==sel_cat]
            .groupby("product_label")["revenue"].sum()
            .reset_index()
            .sort_values("revenue", ascending=False)   # descending
            .head(8)
        )
        fig_top = go.Figure(go.Bar(
            x=top_in_cat["revenue"],
            y=top_in_cat["product_label"],
            orientation="h",
            marker_color=CATEGORY_COLORS.get(sel_cat, CRIMSON),
            text=[f"${v:,.0f}" for v in top_in_cat["revenue"]],
            textposition="outside",
        ))
        fig_top.update_layout(**_base_layout(height=PROD_CHART_H,
                              x_title="Revenue (USD)", y_title="Product"))
        fig_top.update_xaxes(tickprefix="$")
        fig_top.update_yaxes(autorange="reversed")   # highest at top
        st.plotly_chart(fig_top, use_container_width=True)
        graph_ai_expander(f"cat_{sel_cat}",
            f"Top products in {sel_cat}. Best: {top_in_cat.iloc[0]['product_label']} "
            f"at ${top_in_cat.iloc[0]['revenue']:,.0f}.", ai_toggle)

    # ── Top N Products ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Top Products</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Ranked by <strong>total revenue</strong> in the selected period. '
        'Bar shows each product\'s share of the #1 product\'s revenue.'
        '</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([1,2,2])
    with fc1:
        top_n = st.slider("Top N", min_value=5, max_value=20, value=10, step=1)
    with fc2:
        all_cats   = sorted(prod["category"].unique().tolist())
        sel_cats   = st.multiselect("Filter by category", all_cats,
                                     default=all_cats, key="top_n_cats")
    with fc3:
        st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
        st.caption(f"Showing top {top_n} across {len(sel_cats)} categories · {period_label}")

    if not sel_cats:
        st.warning("Select at least one category.")
    else:
        prod_agg = (
            prod[prod["category"].isin(sel_cats)]
            .groupby(["product_id","product_label","category"])
            .agg(revenue=("revenue","sum"), units_sold=("units_sold","sum"))
            .reset_index()
            .sort_values("revenue", ascending=False)
            .head(top_n)
        )
        if prod_agg.empty:
            st.info("No products found for the selected filters.")
        else:
            max_rev = prod_agg["revenue"].max()
            half    = len(prod_agg)//2 + len(prod_agg)%2
            left_df = prod_agg.iloc[:half]
            right_df= prod_agg.iloc[half:]

            tc1, tc2 = st.columns(2)

            def render_tiles(items, col, start_rank):
                with col:
                    for i, row in enumerate(items.itertuples(), start_rank):
                        bar_pct   = int((row.revenue / max_rev) * 100)
                        avg_price = row.revenue / row.units_sold if row.units_sold > 0 else 0
                        cat_icon  = CATEGORY_ICONS.get(row.category, "📦")
                        cat_color = CATEGORY_COLORS.get(row.category, CRIMSON)
                        st.markdown(f"""
                        <div class="prod-tile">
                          <div style="display:flex;align-items:center;gap:0.5rem;">
                            <div style="font-weight:700;font-size:0.88rem;
                                 color:{CRIMSON};min-width:1.25rem;">#{i}</div>
                            <div style="font-size:0.9rem;">{cat_icon}</div>
                            <div style="flex:1;min-width:0;">
                              <div style="font-weight:600;font-size:0.78rem;
                                   white-space:nowrap;overflow:hidden;
                                   text-overflow:ellipsis;">{row.product_label}</div>
                              <div style="font-size:0.66rem;color:{TEXT_MID};">
                                {row.category} &nbsp;·&nbsp;
                                {row.units_sold:.0f} units &nbsp;·&nbsp;
                                avg ${avg_price:.2f}
                              </div>
                              <div style="display:flex;align-items:center;
                                   gap:0.35rem;margin-top:0.22rem;">
                                <div style="flex:1;height:3px;background:{GRAY_LIGHT};
                                     border-radius:3px;overflow:hidden;">
                                  <div style="height:3px;width:{bar_pct}%;
                                       background:{cat_color};border-radius:3px;"></div>
                                </div>
                                <div style="font-size:0.63rem;color:{TEXT_LIGHT};
                                     min-width:36px;">{bar_pct}% of #1</div>
                              </div>
                            </div>
                            <div style="text-align:right;min-width:65px;">
                              <div style="font-weight:700;font-size:0.85rem;
                                   color:{CRIMSON};">${row.revenue:,.0f}</div>
                              <div style="font-size:0.61rem;color:{TEXT_LIGHT};">revenue</div>
                            </div>
                          </div>
                        </div>""", unsafe_allow_html=True)

            render_tiles(left_df,  tc1, 1)
            render_tiles(right_df, tc2, len(left_df)+1)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    if not _has_key():
        st.markdown(f"""
        <div class="callout">
          🔑 <strong>Add your Gemini API key in the sidebar</strong> to unlock all
          AI features. Free at
          <a href="https://aistudio.google.com" target="_blank"
             style="color:{CRIMSON};">aistudio.google.com</a>.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">AI Summary</div>', unsafe_allow_html=True)
    if _has_key():
        if st.button("Generate Summary", key="gen_narrative"):
            with st.spinner("Writing summary..."):
                r = call_gemini("narrative", insights=insights,
                                period_summary=summary, date_label=period_label)
            st.session_state["narrative_result"] = r or "No result."
        if "narrative_result" in st.session_state:
            st.markdown(
                f'<div class="ai-box">✨ <strong>Summary</strong><br><br>'
                f'{st.session_state["narrative_result"]}</div>', unsafe_allow_html=True)
        else:
            st.caption("Click Generate Summary to create your AI narrative.")
    else:
        ai_placeholder("AI will write a 3-5 sentence executive summary of your business performance")

    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    if not insights:
        st.success("No significant changes — metrics are stable ✅")
    else:
        for ins in insights:
            impact = ins.get("impact","low")
            badge  = {"high":"🔴 High","medium":"🟡 Medium","low":"🟢 Low"}.get(impact, impact)
            arrow  = "↑" if ins.get("direction")=="positive" else "↓"
            st.markdown(f"""
            <div class="insight-card insight-{impact}">
              <span class="badge badge-{impact}">{badge}</span>
              {arrow} {ins.get('finding','')}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
    if _has_key():
        if st.button("Generate Recommendations", key="gen_recs"):
            with st.spinner("Building recommendations..."):
                r = call_gemini("recommendations", insights=insights,
                                channel_summary=ch_summary, period_summary=summary)
            st.session_state["recs_result"] = r or "No result."
        if "recs_result" in st.session_state:
            st.markdown(
                f'<div class="ai-box">🎯 <strong>Recommended Actions</strong><br><br>'
                f'{st.session_state["recs_result"].replace(chr(10),"<br>")}</div>',
                unsafe_allow_html=True)
        else:
            st.caption("Click Generate Recommendations.")
    else:
        ai_placeholder("AI will generate 3-5 prioritised, specific action recommendations")

    st.markdown("---")
    st.markdown('<div class="section-header">💬 Ask Anything</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Ask a question about your data ({period_label}). '
                f'Also available in the sidebar on every tab.</div>', unsafe_allow_html=True)

    tab_q = st.text_area("Your question", height=95,
        placeholder="e.g. Why did revenue drop?\nWhich channel has best ROI?\nWhat should I focus on?",
        label_visibility="collapsed", key="tab5_question")
    if st.button("Ask Pulse 🤖", key="tab5_ask", disabled=not _has_key()):
        if tab_q.strip():
            st.session_state["sidebar_answer_pending"] = tab_q.strip()
            st.rerun()
        else:
            st.warning("Type a question first.")
    if not _has_key():
        ai_placeholder("Ask Pulse any question about your data and get a direct, data-grounded answer")
    if "sidebar_answer" in st.session_state:
        _, a = st.session_state["sidebar_answer"]
        st.markdown(
            f'<div class="ai-box">💬 <strong>Pulse says:</strong><br><br>{a}</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Generate Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Download a full written report ready to share with your team or clients.</div>',
                unsafe_allow_html=True)

    rc1, rc2 = st.columns([1,2])
    with rc1:
        report_mode = st.radio("Date range for report",
                                ["Use sidebar range","Custom range"], horizontal=True)
        if report_mode == "Use sidebar range":
            rep_start, rep_end = d_start, d_end
            st.markdown(
                f"<div style='font-size:0.76rem;color:{TEXT_MID};margin-top:0.2rem;'>"
                f"📅 {rep_start.strftime('%b %d, %Y')} → {rep_end.strftime('%b %d, %Y')}"
                f"</div>", unsafe_allow_html=True)
        else:
            rep_start = st.date_input("Report from", value=d_start,
                                       min_value=DATA_MIN, max_value=DATA_MAX, key="rep_start")
            rep_end   = st.date_input("Report to",   value=d_end,
                                       min_value=DATA_MIN, max_value=DATA_MAX, key="rep_end")
            if rep_start > rep_end:
                st.error("Start must be before end.")
                rep_start, rep_end = d_start, d_end
        include_ai = st.checkbox("Include AI narratives", value=_has_key())
        gen_btn    = st.button("📥 Generate Report", type="primary", use_container_width=True)

    with rc2:
        st.markdown(f"""
        <div style='background:{WHITE};border-radius:10px;padding:0.95rem;
             border:1px solid {BORDER};font-size:0.8rem;color:{TEXT_MID};'>
          <strong style='color:{TEXT_DARK};'>Report includes:</strong><br><br>
          ✅ &nbsp;Executive summary<br>
          ✅ &nbsp;KPIs with week-over-week comparison<br>
          ✅ &nbsp;Anomalies with deviation details<br>
          ✅ &nbsp;Channel performance breakdown<br>
          ✅ &nbsp;Top products ranked by revenue<br>
          ✅ &nbsp;Recommended next actions
        </div>""", unsafe_allow_html=True)

    if gen_btn:
        rep_df = df_full[
            (df_full["date"]>=pd.Timestamp(rep_start)) &
            (df_full["date"]<=pd.Timestamp(rep_end))
        ]
        if rep_df.empty:
            st.error(
                f"⚠️ No data found between **{rep_start}** and **{rep_end}**. "
                f"Please select dates within: "
                f"**{DATA_MIN.strftime('%b %d, %Y')}** → **{DATA_MAX.strftime('%b %d, %Y')}**."
            )
        else:
            rep_days   = max(1, (rep_end-rep_start).days+1)
            rep_cmp    = min(7, rep_days)
            rep_sum    = compute_period_summary(rep_df, days=rep_cmp)
            rep_ch     = ch_full[(ch_full["date"]>=pd.Timestamp(rep_start))&(ch_full["date"]<=pd.Timestamp(rep_end))]
            rep_prod   = prod_full[(prod_full["date"]>=pd.Timestamp(rep_start))&(prod_full["date"]<=pd.Timestamp(rep_end))]
            rep_anom   = get_recent_anomalies(all_anomalies, days=rep_days)
            rep_ch_sum = compute_channel_summary(rep_ch, days=rep_cmp)
            rep_prods  = compute_top_products(rep_prod, n=5, days=rep_cmp)
            rep_ins    = compile_all_insights(rep_sum, rep_anom, rep_ch_sum)

            with st.spinner("Assembling report..."):
                ai_nar=ai_recs=ai_anom_exp=None
                if include_ai and _has_key():
                    rl = f"{rep_start.strftime('%b %d')} – {rep_end.strftime('%b %d, %Y')}"
                    ai_nar  = call_gemini("narrative", insights=rep_ins, period_summary=rep_sum, date_label=rl)
                    ai_recs = call_gemini("recommendations", insights=rep_ins,
                                          channel_summary=rep_ch_sum, period_summary=rep_sum)
                    if not rep_anom.empty:
                        ta = rep_anom.iloc[0]
                        ai_anom_exp = call_gemini("anomaly", metric=ta["metric"],
                            direction=str(ta.get("direction","")),
                            deviation_pct=float(ta.get("deviation_pct",0)),
                            anomaly_type=str(ta.get("anomaly_type","")),
                            period_summary=rep_sum)
                report_text = generate_report(
                    period_summary=rep_sum, insights=rep_ins,
                    anomalies=rep_anom, channel_summary=rep_ch_sum,
                    top_products=rep_prods, report_date=rep_end,
                    ai_narrative=ai_nar, ai_anomaly_explanation=ai_anom_exp,
                    ai_recommendations=ai_recs, save=True,
                )
            st.session_state["last_report"] = report_text
            st.success(f"✅ Report ready — {len(rep_df):,} days of data.")
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=report_text,
                file_name=f"pulse_report_{rep_start}_{rep_end}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # Report preview / sample
    st.markdown("---")
    st.markdown('<div class="section-header">Report Preview</div>', unsafe_allow_html=True)
    if "last_report" in st.session_state:
        st.markdown('<div class="section-sub">Your most recently generated report.</div>',
                    unsafe_allow_html=True)
        with st.expander("📄 View last generated report", expanded=False):
            st.code(st.session_state["last_report"], language=None)
    else:
        st.markdown('<div class="section-sub">Generate a report above to see the full output. '
                    'Here is the report structure:</div>', unsafe_allow_html=True)
        sample = """
------------------------------------------------------------------------
  PULSE — DAILY INSIGHT REPORT
  Generated: YYYY-MM-DD HH:MM
  Reporting period: 7-day window ending YYYY-MM-DD
------------------------------------------------------------------------

EXECUTIVE SUMMARY
------------------------------------------------------------------------
[AI-generated 3-5 sentence plain-English summary of business performance]

KPI SNAPSHOT  (last 7 days vs prior 7 days)
------------------------------------------------------------------------
  Metric                        Recent        Prior       Change
  revenue (avg/day)             $X,XXX        $X,XXX      ^ +X.X%
  orders (avg/day)              XXX           XXX         ^ +X.X%
  aov                           $XX.XX        $XX.XX      v -X.X%
  conversion_rate               X.XX%         X.XX%       v -X.X%
  roas                          X.XXx         X.XXx       ^ +X.X%

ANOMALIES DETECTED
------------------------------------------------------------------------
  [UP]   Revenue — Single Day Spike (XX% deviation) on MMM DD, YYYY
  [DOWN] Conversion Rate — Sustained Deviation (XX%)

KEY FINDINGS
------------------------------------------------------------------------
  [HIGH]   ↑ Revenue surged XX% on Dec 23 (pre-holiday spike)
  [MEDIUM] ↓ conversion_rate -X.X% vs prior 7-day average

CHANNEL PERFORMANCE
------------------------------------------------------------------------
  Channel         Spend      Clicks  Conv    Rev Attr    ROAS
  Paid Search     $X,XXX     X,XXX   XXX     $XX,XXX     X.XXx

TOP PRODUCTS  (last 7 days by revenue)
------------------------------------------------------------------------
  1. Product Name  (Category)  Revenue: $XXX  Units: XX

RECOMMENDED ACTIONS
------------------------------------------------------------------------
  1. [AI-generated specific action]
  2. [AI-generated specific action]
------------------------------------------------------------------------
        """.strip()
        with st.expander("📄 View sample report structure", expanded=True):
            st.code(sample, language=None)
