"""
Pulse — AI-powered insight engine.
Streamlit dashboard: light mode, growth palette, per-graph AI insights,
revamped anomaly page, custom date ranges, axis labels throughout.
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pulse — AI Insight Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

PURPLE       = "#7C3AED"
PURPLE_LIGHT = "#EDE9FE"
PURPLE_MID   = "#A78BFA"
PINK         = "#EC4899"
PINK_LIGHT   = "#FCE7F3"
GREEN        = "#10B981"
GREEN_LIGHT  = "#D1FAE5"
AMBER        = "#F59E0B"
AMBER_LIGHT  = "#FEF3C7"
RED          = "#EF4444"
RED_LIGHT    = "#FEE2E2"
CYAN         = "#06B6D4"
BG           = "#F8FAFC"
WHITE        = "#FFFFFF"
TEXT_DARK    = "#1F2937"
TEXT_MID     = "#6B7280"
TEXT_LIGHT   = "#9CA3AF"
BORDER       = "#E5E7EB"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Sans:wght@400;500&display=swap');

  html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG};
    color: {TEXT_DARK};
  }}
  h1,h2,h3 {{ font-family: 'Plus Jakarta Sans', sans-serif; }}

  section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PURPLE} 0%, #5B21B6 100%);
  }}
  section[data-testid="stSidebar"] * {{ color: white !important; }}
  section[data-testid="stSidebar"] .stTextInput input,
  section[data-testid="stSidebar"] .stSelectbox > div,
  section[data-testid="stSidebar"] .stDateInput input {{
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 8px !important;
  }}
  section[data-testid="stSidebar"] label {{
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    opacity: 0.85;
  }}
  section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.2) !important; }}

  .kpi-card {{
    background: {WHITE}; border-radius: 16px;
    padding: 1.1rem 1.3rem; border: 1px solid {BORDER};
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 16px rgba(124,58,237,0.05);
    margin-bottom: 0.5rem;
  }}
  .kpi-icon  {{ font-size: 1.3rem; margin-bottom: 0.3rem; }}
  .kpi-label {{
    font-size: 0.7rem; color: {TEXT_MID}; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    font-family: 'Plus Jakarta Sans', sans-serif;
  }}
  .kpi-value {{
    font-size: 1.65rem; font-weight: 800; color: {TEXT_DARK};
    font-family: 'Plus Jakarta Sans', sans-serif; line-height: 1.1;
  }}
  .kpi-delta-up   {{ display:inline-block; margin-top:0.3rem; font-size:0.75rem; font-weight:600; color:{GREEN}; background:{GREEN_LIGHT}; padding:2px 8px; border-radius:20px; }}
  .kpi-delta-down {{ display:inline-block; margin-top:0.3rem; font-size:0.75rem; font-weight:600; color:{RED};   background:{RED_LIGHT};   padding:2px 8px; border-radius:20px; }}
  .kpi-delta-neu  {{ display:inline-block; margin-top:0.3rem; font-size:0.75rem; font-weight:600; color:{TEXT_MID}; background:#F3F4F6; padding:2px 8px; border-radius:20px; }}
  .kpi-hint {{ font-size:0.68rem; color:{TEXT_LIGHT}; margin-top:0.2rem; }}

  .section-header {{
    font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1rem;
    font-weight: 700; color: {TEXT_DARK}; margin: 1.4rem 0 0.3rem 0;
  }}
  .section-sub {{ font-size: 0.8rem; color: {TEXT_MID}; margin-bottom: 0.7rem; }}

  .ai-box {{
    background: linear-gradient(135deg, {PURPLE_LIGHT}, {PINK_LIGHT});
    border: 1px solid #DDD6FE; border-radius: 14px;
    padding: 1.1rem 1.4rem; line-height: 1.75;
    color: {TEXT_DARK}; font-size: 0.9rem; margin: 0.5rem 0 0.8rem 0;
  }}

  .insight-card {{
    background: {WHITE}; border-radius: 12px;
    padding: 0.85rem 1.1rem; border: 1px solid {BORDER};
    margin-bottom: 0.45rem; border-left: 4px solid {PURPLE};
  }}
  .insight-high   {{ border-left-color: {RED};   }}
  .insight-medium {{ border-left-color: {AMBER}; }}
  .insight-low    {{ border-left-color: {GREEN};  }}

  .badge {{ display:inline-block; font-size:0.63rem; font-weight:700; padding:2px 7px; border-radius:20px; text-transform:uppercase; letter-spacing:0.05em; margin-right:5px; }}
  .badge-high   {{ background:{RED_LIGHT};   color:{RED};   }}
  .badge-medium {{ background:{AMBER_LIGHT}; color:{AMBER}; }}
  .badge-low    {{ background:{GREEN_LIGHT}; color:{GREEN}; }}

  .anomaly-stat-card {{
    background: {WHITE}; border-radius: 14px;
    padding: 1.1rem 1.3rem; border: 1px solid {BORDER};
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    text-align: center;
  }}
  .anom-val {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2rem; font-weight: 800; color: {PURPLE};
  }}
  .anom-label {{ font-size: 0.78rem; color: {TEXT_MID}; margin-top: 0.2rem; }}

  .callout {{
    background: {PURPLE_LIGHT}; border-radius: 12px;
    padding: 0.85rem 1.1rem; font-size: 0.84rem;
    color: #4C1D95; margin-bottom: 0.9rem;
    border: 1px solid #DDD6FE;
  }}

  .hero {{
    background: linear-gradient(135deg, {PURPLE} 0%, {PINK} 100%);
    border-radius: 18px; padding: 1.6rem 2rem; color: white;
    margin-bottom: 1.4rem; position: relative; overflow: hidden;
  }}
  .hero::after {{
    content:''; position:absolute; top:-40px; right:-40px;
    width:160px; height:160px; border-radius:50%;
    background:rgba(255,255,255,0.08);
  }}
  .hero-title {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:1.4rem; font-weight:800; }}
  .hero-sub   {{ font-size:0.85rem; opacity:0.82; margin-top:0.2rem; }}
  .hero-stats {{ display:flex; gap:2rem; margin-top:1rem; }}
  .hero-stat-val   {{ font-family:'Plus Jakarta Sans',sans-serif; font-size:1.25rem; font-weight:800; }}
  .hero-stat-label {{ font-size:0.68rem; opacity:0.72; text-transform:uppercase; letter-spacing:0.05em; }}

  .stTabs [data-baseweb="tab-list"] {{
    gap:4px; background:{WHITE}; border-radius:12px;
    padding:4px; border:1px solid {BORDER}; margin-bottom:1rem;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius:8px; font-size:0.8rem; font-weight:600;
    font-family:'Plus Jakarta Sans',sans-serif; color:{TEXT_MID}; padding:6px 14px;
  }}
  .stTabs [aria-selected="true"] {{
    background:linear-gradient(135deg,{PURPLE},{PINK}) !important;
    color:white !important;
  }}

  .stButton > button {{
    background: linear-gradient(135deg,{PURPLE},{PINK});
    color:white; border:none; border-radius:10px;
    font-weight:600; font-family:'Plus Jakarta Sans',sans-serif;
    transition:opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity:0.88; }}

  .block-container {{ padding-top: 1.4rem; }}
  .stDataFrame {{ border-radius:12px; overflow:hidden; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KPI_META = {
    "revenue":         {"label":"revenue",         "icon":"💰", "hint":"Total sales per day",                                    "fmt":lambda v:f"${v:,.0f}",    "good":"up"},
    "orders":          {"label":"orders",           "icon":"🛍️", "hint":"Number of purchases made",                              "fmt":lambda v:f"{v:,.0f}",      "good":"up"},
    "aov":             {"label":"aov",              "icon":"🧾", "hint":"Average spend per order",                               "fmt":lambda v:f"${v:,.2f}",     "good":"up"},
    "conversion_rate": {"label":"conversion_rate",  "icon":"🎯", "hint":"% of clicks that resulted in a purchase",              "fmt":lambda v:f"{v*100:.2f}%",  "good":"up"},
    "roas":            {"label":"roas",             "icon":"📣", "hint":"Revenue earned per $1 of ad spend",                    "fmt":lambda v:f"{v:.2f}x",      "good":"up"},
    "spend":           {"label":"spend",            "icon":"💸", "hint":"Daily marketing spend",                                "fmt":lambda v:f"${v:,.0f}",     "good":"neutral"},
}

METRIC_LABELS = {
    "revenue":"Revenue", "orders":"Orders", "aov":"AOV",
    "conversion_rate":"Conversion Rate", "cac":"CAC",
    "roas":"ROAS", "spend":"Spend",
}

CHANNEL_COLORS = {
    "Paid Search":"#7C3AED","Social":"#EC4899","Email":"#06B6D4",
    "Affiliate":"#10B981","Display":"#F59E0B","Organic / Direct":"#8B5CF6",
}
CATEGORY_COLORS = {
    "Electronics":"#7C3AED","Fashion":"#EC4899","Sports":"#10B981",
    "Home":"#F59E0B","Beauty":"#EF4444","Grocery":"#06B6D4",
}
CATEGORY_ICONS = {
    "Electronics":"💻","Fashion":"👗","Sports":"🏃",
    "Home":"🏠","Beauty":"✨","Grocery":"🛒",
}
ANOMALY_METRIC_COLORS = {
    "revenue":"#7C3AED","orders":"#EC4899","aov":"#06B6D4",
    "conversion_rate":"#10B981","cac":"#F59E0B","roas":"#EF4444","spend":"#8B5CF6",
}

# ---------------------------------------------------------------------------
# Chart base layout
# ---------------------------------------------------------------------------

def _base_layout(title="", height=300, x_title="", y_title=""):
    layout = dict(
        title=dict(text=title, font=dict(family="Plus Jakarta Sans", size=13,
                   color=TEXT_DARK), x=0),
        height=height, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        font=dict(color=TEXT_DARK, family="DM Sans", size=11),
        xaxis=dict(title=x_title, gridcolor="#F3F4F6", showgrid=True,
                   zeroline=False, linecolor=BORDER,
                   title_font=dict(size=11, color=TEXT_MID)),
        yaxis=dict(title=y_title, gridcolor="#F3F4F6", showgrid=True,
                   zeroline=False, linecolor=BORDER,
                   title_font=dict(size=11, color=TEXT_MID)),
        margin=dict(l=50, r=20, t=44, b=44),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER,
                        font=dict(family="DM Sans")),
    )
    return layout

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

def _gemini_available():
    return bool(os.getenv("GEMINI_API_KEY") or
                st.secrets.get("GEMINI_API_KEY", ""))

def call_gemini(fn, *args, **kwargs):
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY","")
    if not api_key:
        return None
    os.environ["GEMINI_API_KEY"] = api_key
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

def call_gemini_freeform(question: str, period_summary: dict,
                          insights: list, channel_summary) -> str:
    """Answers a free-form question using current filtered data as context."""
    from analytics.insights import _format_insights_for_prompt  # reuse formatter
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY","")
    if not api_key:
        return None
    os.environ["GEMINI_API_KEY"] = api_key

    metrics_lines = []
    for metric, data in period_summary.items():
        if metric == "period_days" or not isinstance(data, dict):
            continue
        pct    = data.get("pct_change")
        recent = data.get("recent_avg")
        if pct is None or recent is None:
            continue
        metrics_lines.append(
            f"  {metric}: {recent} (WoW: {pct*100:+.1f}%)"
        )

    channel_lines = []
    try:
        for _, row in channel_summary.iterrows():
            roas = f"{row['roas']:.2f}x" if pd.notna(row.get("roas")) else "N/A"
            channel_lines.append(
                f"  {row['channel']}: spend=${row['spend']:.0f},"
                f" conversions={row['conversions']:.0f}, ROAS={roas}"
            )
    except Exception:
        pass

    insight_lines = [
        f"  [{i.get('impact','').upper()}] {i.get('finding','')}"
        for i in insights[:8]
    ]

    import textwrap, requests
    prompt = textwrap.dedent(f"""
        You are a data analytics advisor. Answer the following question
        using only the business data provided below.
        Be specific, concise, and actionable. 3-5 sentences maximum.

        USER QUESTION: {question}

        CURRENT PERIOD METRICS:
        {chr(10).join(metrics_lines)}

        CHANNEL PERFORMANCE:
        {chr(10).join(channel_lines)}

        KEY INSIGHTS:
        {chr(10).join(insight_lines)}
    """).strip()

    try:
        url     = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
        }
        r = requests.post(f"{url}?key={api_key}", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return f"API error {r.status_code}: {r.text}"
    except Exception as e:
        return f"Error: {e}"

def graph_ai_expander(graph_id: str, context: str, ai_on: bool):
    """
    Renders a collapsible AI Insight block below a graph.
    graph_id  - unique key for session state
    context   - plain-text description of what this graph shows
    ai_on     - whether the global AI toggle is on
    """
    if not ai_on:
        return
    with st.expander("🤖 AI Insight for this chart"):
        btn_key = f"ai_btn_{graph_id}"
        res_key = f"ai_res_{graph_id}"
        if st.button("Generate Insight", key=btn_key):
            with st.spinner("Thinking..."):
                api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY","")
                import textwrap, requests
                prompt = textwrap.dedent(f"""
                    You are a business analyst. Based on this chart description,
                    give 2-3 concise, actionable observations in plain English.
                    No bullet formatting — write in prose.
                    Do not start with 'I'.

                    Chart: {context}
                """).strip()
                try:
                    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
                    r   = requests.post(
                        f"{url}?key={api_key}",
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 300},
                        },
                        timeout=30,
                    )
                    if r.status_code == 200:
                        result = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    else:
                        result = f"API error {r.status_code}"
                except Exception as e:
                    result = f"Error: {e}"
                st.session_state[res_key] = result

        if res_key in st.session_state:
            st.markdown(
                f'<div class="ai-box">{st.session_state[res_key]}</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style='padding:0.4rem 0 1rem 0;'>
      <div style='font-family:Plus Jakarta Sans;font-size:1.25rem;
           font-weight:800;color:white;letter-spacing:-0.01em;'>⚡ Pulse</div>
      <div style='font-size:0.72rem;opacity:0.68;margin-top:1px;'>
        AI-powered insight engine
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    df_full, ch_full, prod_full = load_and_enrich()
    DATA_MIN = df_full["date"].min().date()
    DATA_MAX = df_full["date"].max().date()

    st.markdown("**📅 Date Range**")
    preset = st.selectbox("Quick select", [
        "Last 30 days","Last 90 days","Last 6 months",
        "This year","All time","Custom range",
    ], index=0)

    if preset == "Last 30 days":
        d_start, d_end = DATA_MAX - timedelta(days=29), DATA_MAX
    elif preset == "Last 90 days":
        d_start, d_end = DATA_MAX - timedelta(days=89), DATA_MAX
    elif preset == "Last 6 months":
        d_start, d_end = DATA_MAX - timedelta(days=179), DATA_MAX
    elif preset == "This year":
        d_start, d_end = date(DATA_MAX.year, 1, 1), DATA_MAX
    elif preset == "All time":
        d_start, d_end = DATA_MIN, DATA_MAX
    else:
        d_start = d_end = None

    if preset == "Custom range":
        d_start = st.date_input("From", value=DATA_MAX - timedelta(days=29),
                                 min_value=DATA_MIN, max_value=DATA_MAX)
        d_end   = st.date_input("To",   value=DATA_MAX,
                                 min_value=DATA_MIN, max_value=DATA_MAX)
        if d_start > d_end:
            st.error("Start must be before end date.")
            d_start, d_end = DATA_MAX - timedelta(days=29), DATA_MAX
    else:
        d_start = max(d_start, DATA_MIN)
        d_end   = min(d_end,   DATA_MAX)
        st.markdown(
            f"<div style='font-size:0.75rem;opacity:0.75;margin-top:-0.2rem;'>"
            f"{d_start.strftime('%b %d, %Y')} → {d_end.strftime('%b %d, %Y')}"
            f"</div>", unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**🤖 AI Insights**")
    gemini_key = st.text_input("Gemini API Key", type="password",
                                placeholder="Paste key to enable AI",
                                help="Free key at aistudio.google.com")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        st.markdown(
            "<div style='font-size:0.75rem;background:rgba(16,185,129,0.25);"
            "border-radius:8px;padding:3px 8px;margin-top:3px;'>"
            "✅ AI active</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='font-size:0.73rem;opacity:0.65;margin-top:3px;'>"
            "Add key to unlock AI features</div>", unsafe_allow_html=True)

    ai_toggle = st.toggle(
        "Show AI Insights on graphs",
        value=False,
        disabled=not bool(gemini_key),
        help="When ON, each chart gets an AI Insight button",
    )
    if not gemini_key:
        st.markdown(
            "<div style='font-size:0.7rem;opacity:0.55;margin-top:-0.4rem;'>"
            "Requires API key above</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        with st.spinner("Re-running ETL..."):
            from etl.pipeline import run as run_etl
            run_etl()
            st.cache_data.clear()
        st.success("Done — reload page.")

    st.markdown(
        f"<div style='font-size:0.67rem;opacity:0.45;margin-top:1rem;'>"
        f"Dataset: {DATA_MIN.strftime('%b %Y')} – {DATA_MAX.strftime('%b %Y')}"
        f"</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filtered data
# ---------------------------------------------------------------------------

ts_start = pd.Timestamp(d_start)
ts_end   = pd.Timestamp(d_end)
days_sel = (d_end - d_start).days + 1
cmp_days = min(7, days_sel)

df   = df_full[(df_full["date"] >= ts_start) & (df_full["date"] <= ts_end)].copy()
ch   = ch_full[(ch_full["date"] >= ts_start) & (ch_full["date"] <= ts_end)].copy()
prod = prod_full[(prod_full["date"] >= ts_start) & (prod_full["date"] <= ts_end)].copy()

summary      = compute_period_summary(df, days=cmp_days)
all_anomalies= get_all_anomalies(df_full)
rec_anomalies= get_recent_anomalies(all_anomalies, days=min(days_sel, 60))
ch_summary   = compute_channel_summary(ch, days=cmp_days)
top_products = compute_top_products(prod, n=5, days=cmp_days)
insights     = compile_all_insights(summary, rec_anomalies, ch_summary)

total_rev   = df["revenue"].sum()
total_orders= int(df["orders"].sum()) if "orders" in df.columns else 0
high_count  = sum(1 for i in insights if i.get("impact") == "high")
period_label= f"{d_start.strftime('%b %d')} – {d_end.strftime('%b %d, %Y')}"

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="hero">
  <div class="hero-title">⚡ Pulse — Insight Engine</div>
  <div class="hero-sub">{period_label} &nbsp;·&nbsp; {days_sel} days</div>
  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hero-stat-val">${total_rev:,.0f}</div>
      <div class="hero-stat-label">Total Revenue</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">{total_orders:,}</div>
      <div class="hero-stat-label">Total Orders</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">{len(insights)}</div>
      <div class="hero-stat-label">Insights</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">{high_count}</div>
      <div class="hero-stat-label">Need Attention</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview", "🚨 Anomalies", "📣 Channels",
    "📦 Products",  "🤖 AI Insights", "📄 Report",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">KPI Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Last {cmp_days} days vs prior {cmp_days} days</div>', unsafe_allow_html=True)

    kpi_cols = st.columns(len(KPI_META))
    for col, (metric, meta) in zip(kpi_cols, KPI_META.items()):
        data   = summary.get(metric, {})
        recent = data.get("recent_avg") if isinstance(data, dict) else None
        pct    = data.get("pct_change") if isinstance(data, dict) else None
        val_str= meta["fmt"](recent) if recent is not None else "—"
        good   = meta["good"]

        if pct is None:
            delta = '<span class="kpi-delta-neu">No comparison</span>'
        else:
            arrow  = "▲" if pct > 0 else "▼"
            label  = f"{arrow} {abs(pct)*100:.1f}% vs prior"
            is_good= (pct > 0 and good == "up") or (pct < 0 and good == "down")
            cls    = "kpi-delta-up" if is_good else ("kpi-delta-down" if good != "neutral" else "kpi-delta-neu")
            delta  = f'<span class="{cls}">{label}</span>'

        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-icon">{meta['icon']}</div>
              <div class="kpi-label">{meta['label']}</div>
              <div class="kpi-value">{val_str}</div>
              {delta}
              <div class="kpi-hint">{meta['hint']}</div>
            </div>""", unsafe_allow_html=True)

    # Revenue trend
    st.markdown('<div class="section-header">Revenue Trend</div>', unsafe_allow_html=True)
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=df["date"], y=df["revenue"], mode="lines", name="Daily Revenue",
        line=dict(color=PURPLE, width=2.5),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
    ))
    if "revenue_roll_mean" in df.columns:
        fig_rev.add_trace(go.Scatter(
            x=df["date"], y=df["revenue_roll_mean"], mode="lines",
            name="7-day avg", line=dict(color=PINK, width=1.8, dash="dot"),
        ))
    fig_rev.update_layout(**_base_layout(height=300, x_title="Date", y_title="Revenue (USD)"))
    fig_rev.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_rev, use_container_width=True)
    graph_ai_expander("rev_trend",
        f"Daily revenue trend from {d_start} to {d_end}. "
        f"Total revenue ${total_rev:,.0f} over {days_sel} days. "
        f"7-day average also shown.", ai_toggle)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">conversion_rate</div>', unsafe_allow_html=True)
        fig_cr = go.Figure()
        fig_cr.add_trace(go.Scatter(
            x=df["date"], y=df["conversion_rate"]*100, mode="lines",
            line=dict(color=GREEN, width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.08)", name="conversion_rate",
        ))
        fig_cr.update_layout(**_base_layout(height=240, x_title="Date", y_title="Conversion Rate (%)"))
        fig_cr.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_cr, use_container_width=True)
        cr_recent = summary.get("conversion_rate", {}).get("recent_avg", 0) or 0
        graph_ai_expander("conv_rate",
            f"Conversion rate trend. Current avg: {cr_recent*100:.2f}%. "
            f"Period: {d_start} to {d_end}.", ai_toggle)

    with col_b:
        st.markdown('<div class="section-header">New vs Returning Customers</div>', unsafe_allow_html=True)
        fig_cust = go.Figure()
        fig_cust.add_trace(go.Scatter(
            x=df["date"], y=df["new_customers"], mode="lines",
            name="new_customers", line=dict(color=PURPLE, width=2),
            stackgroup="one", fillcolor="rgba(124,58,237,0.3)",
        ))
        fig_cust.add_trace(go.Scatter(
            x=df["date"], y=df["returning_customers"], mode="lines",
            name="returning_customers", line=dict(color=PINK, width=2),
            stackgroup="one", fillcolor="rgba(236,72,153,0.3)",
        ))
        fig_cust.update_layout(**_base_layout(height=240, x_title="Date", y_title="Customer Count"))
        st.plotly_chart(fig_cust, use_container_width=True)
        graph_ai_expander("cust_mix",
            f"New vs returning customers stacked area chart from {d_start} to {d_end}.",
            ai_toggle)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-header">aov</div>', unsafe_allow_html=True)
        fig_aov = go.Figure()
        fig_aov.add_trace(go.Scatter(
            x=df["date"], y=df["aov"], mode="lines",
            line=dict(color=AMBER, width=2.5),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.08)", name="aov",
        ))
        fig_aov.update_layout(**_base_layout(height=240, x_title="Date", y_title="AOV (USD)"))
        fig_aov.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_aov, use_container_width=True)
        aov_val = summary.get("aov", {}).get("recent_avg", 0) or 0
        graph_ai_expander("aov",
            f"Average Order Value (AOV) trend. Current avg: ${aov_val:.2f}. "
            f"Period: {d_start} to {d_end}.", ai_toggle)

    with col_d:
        st.markdown('<div class="section-header">roas</div>', unsafe_allow_html=True)
        fig_roas = go.Figure()
        fig_roas.add_trace(go.Scatter(
            x=df["date"], y=df["roas"], mode="lines",
            line=dict(color=PINK, width=2.5),
            fill="tozeroy", fillcolor="rgba(236,72,153,0.08)", name="roas",
        ))
        fig_roas.add_hline(y=2, line_dash="dot", line_color=RED,
                           annotation_text="Min target 2x",
                           annotation_font_color=RED)
        fig_roas.update_layout(**_base_layout(height=240, x_title="Date", y_title="ROAS (x)"))
        fig_roas.update_yaxes(ticksuffix="x")
        st.plotly_chart(fig_roas, use_container_width=True)
        roas_val = summary.get("roas", {}).get("recent_avg", 0) or 0
        graph_ai_expander("roas",
            f"ROAS trend. Current avg: {roas_val:.2f}x. Target is 2x. "
            f"Period: {d_start} to {d_end}.", ai_toggle)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Summary stat cards ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Anomaly Summary</div>', unsafe_allow_html=True)

    most_affected = "—"
    worst_dev     = 0.0
    worst_date    = "—"

    if not rec_anomalies.empty:
        mc = rec_anomalies["metric"].value_counts()
        most_affected = METRIC_LABELS.get(mc.index[0], mc.index[0]) if len(mc) else "—"
        worst_row     = rec_anomalies.loc[rec_anomalies["deviation_pct"].idxmax()]
        worst_dev     = worst_row["deviation_pct"]
        wd = worst_row["date"]
        worst_date = wd.date().strftime("%b %d, %Y") if hasattr(wd, "date") else str(wd)

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, val, label, color in [
        (sc1, len(rec_anomalies),          "Total Anomalies",          PURPLE),
        (sc2, most_affected,               "Most Affected Metric",      PINK),
        (sc3, f"{worst_dev*100:.0f}%",     "Largest Single Deviation",  RED),
        (sc4, worst_date,                  "Date of Worst Anomaly",     AMBER),
    ]:
        with col:
            st.markdown(f"""
            <div class="anomaly-stat-card">
              <div class="anom-val" style="color:{color};">{val}</div>
              <div class="anom-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:0.8rem;"></div>', unsafe_allow_html=True)

    if rec_anomalies.empty:
        st.markdown("""
        <div style='text-align:center;padding:2.5rem;background:white;
             border-radius:16px;border:1px solid #E5E7EB;margin-top:0.5rem;'>
          <div style='font-size:2.2rem;'>✅</div>
          <div style='font-family:Plus Jakarta Sans;font-weight:700;
               font-size:1rem;color:#1F2937;margin-top:0.4rem;'>
            No anomalies in this period</div>
          <div style='color:#6B7280;font-size:0.82rem;margin-top:0.2rem;'>
            All metrics are within normal range.</div>
        </div>""", unsafe_allow_html=True)
    else:
        anom_df = rec_anomalies.copy()
        anom_df["date"] = pd.to_datetime(anom_df["date"])
        anom_df["metric_label"] = anom_df["metric"].map(
            lambda m: METRIC_LABELS.get(m, m))
        anom_df["color"] = anom_df["metric"].map(
            lambda m: ANOMALY_METRIC_COLORS.get(m, PURPLE))
        anom_df["direction_label"] = anom_df["direction"].map(
            lambda d: "Spike Up ↑" if "up" in str(d) else "Spike Down ↓")

        col_chart1, col_chart2 = st.columns([3, 2])

        # ── Scatter timeline ─────────────────────────────────────────────
        with col_chart1:
            st.markdown('<div class="section-header">Anomaly Timeline</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Each dot is an anomaly. Bigger dot = larger deviation. Hover for details.</div>', unsafe_allow_html=True)

            fig_scatter = go.Figure()
            for metric in anom_df["metric"].unique():
                sub = anom_df[anom_df["metric"] == metric]
                fig_scatter.add_trace(go.Scatter(
                    x=sub["date"],
                    y=sub["metric_label"],
                    mode="markers",
                    name=METRIC_LABELS.get(metric, metric),
                    marker=dict(
                        size=sub["deviation_pct"] * 120,
                        color=ANOMALY_METRIC_COLORS.get(metric, PURPLE),
                        opacity=0.75,
                        line=dict(width=1, color="white"),
                    ),
                    customdata=np.stack([
                        sub["deviation_pct"]*100,
                        sub["direction_label"],
                        sub["anomaly_type"].str.replace("_"," "),
                    ], axis=-1),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Date: %{x|%b %d, %Y}<br>"
                        "Deviation: %{customdata[0]:.1f}%<br>"
                        "Direction: %{customdata[1]}<br>"
                        "Type: %{customdata[2]}<extra></extra>"
                    ),
                ))
            fig_scatter.update_layout(**_base_layout(
                height=320, x_title="Date", y_title="Metric",
            ))
            st.plotly_chart(fig_scatter, use_container_width=True)
            graph_ai_expander("anom_timeline",
                f"Anomaly timeline showing {len(anom_df)} anomalies across "
                f"{anom_df['metric'].nunique()} metrics. "
                f"Most affected: {most_affected}.", ai_toggle)

        # ── Donut breakdown ──────────────────────────────────────────────
        with col_chart2:
            st.markdown('<div class="section-header">Anomalies by Metric</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Which metric has been most volatile?</div>', unsafe_allow_html=True)

            donut_data = anom_df["metric_label"].value_counts().reset_index()
            donut_data.columns = ["metric", "count"]
            donut_colors = [
                ANOMALY_METRIC_COLORS.get(
                    {v: k for k, v in METRIC_LABELS.items()}.get(m, m), PURPLE
                )
                for m in donut_data["metric"]
            ]
            fig_donut = go.Figure(go.Pie(
                labels=donut_data["metric"],
                values=donut_data["count"],
                hole=0.55,
                marker=dict(colors=donut_colors,
                            line=dict(color="white", width=2)),
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig_donut.update_layout(
                height=300, paper_bgcolor=WHITE,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{len(anom_df)}</b><br>total",
                    x=0.5, y=0.5, font_size=14,
                    font_color=TEXT_DARK, showarrow=False,
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            graph_ai_expander("anom_donut",
                f"Distribution of anomalies by metric. "
                f"Most volatile: {most_affected} with {donut_data['count'].max()} anomalies.",
                ai_toggle)

        # ── Heatmap ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Anomaly Heatmap</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Darker = more anomalies in that month for that metric. Use this to spot which periods were most unstable.</div>', unsafe_allow_html=True)

        anom_df["month"] = anom_df["date"].dt.to_period("M").astype(str)
        heatmap_data = (
            anom_df.groupby(["month","metric_label"]).size()
            .reset_index(name="count")
            .pivot(index="metric_label", columns="month", values="count")
            .fillna(0)
        )
        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale=[[0, "#F3F0FF"],[0.5, PURPLE_MID],[1, PURPLE]],
            text=heatmap_data.values.astype(int),
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="Month: %{x}<br>Metric: %{y}<br>Anomalies: %{z}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Count", thickness=12),
        ))
        fig_heat.update_layout(
            height=260, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
            font=dict(family="DM Sans", size=11, color=TEXT_DARK),
            margin=dict(l=100, r=20, t=20, b=60),
            xaxis=dict(title="Month", tickangle=-30,
                       title_font=dict(size=11, color=TEXT_MID)),
            yaxis=dict(title="Metric",
                       title_font=dict(size=11, color=TEXT_MID)),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        graph_ai_expander("anom_heatmap",
            f"Monthly anomaly frequency heatmap across all metrics. "
            f"Identifies which months had most instability.", ai_toggle)

        # ── Individual anomaly cards ──────────────────────────────────────
        st.markdown('<div class="section-header">Anomaly Detail</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Click any row to see what happened and get an AI explanation.</div>', unsafe_allow_html=True)

        for _, row in anom_df.sort_values("date", ascending=False).iterrows():
            is_up    = "up" in str(row.get("direction",""))
            dev      = row.get("deviation_pct", 0)
            severity = "🔴 High" if dev >= 0.20 else "🟡 Medium"
            sev_color= RED if dev >= 0.20 else AMBER
            sev_bg   = RED_LIGHT if dev >= 0.20 else AMBER_LIGHT
            icon     = "↑" if is_up else "↓"
            date_str = row["date"].date().strftime("%B %d, %Y") if hasattr(row["date"],"date") else str(row["date"])
            atype    = str(row.get("anomaly_type","")).replace("_"," ").title()
            metric_l = row["metric_label"]

            with st.expander(
                f"{icon} {metric_l}  ·  {dev*100:.0f}% deviation  ·  {date_str}",
                expanded=False,
            ):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div style='background:{sev_bg};border-radius:12px;
                         padding:1rem;border:1px solid {sev_color};'>
                      <div style='font-weight:700;font-family:Plus Jakarta Sans;
                           font-size:0.95rem;color:{TEXT_DARK};'>{metric_l}</div>
                      <div style='font-size:1.8rem;font-weight:800;
                           color:{sev_color};margin:0.4rem 0;'>
                        {icon} {dev*100:.1f}%</div>
                      <div style='font-size:0.75rem;color:{TEXT_MID};'>{atype}</div>
                      <div style='font-size:0.75rem;color:{TEXT_MID};
                           margin-top:0.3rem;'>{date_str}</div>
                      <div style='margin-top:0.5rem;font-size:0.72rem;
                           font-weight:700;color:{sev_color};'>{severity}</div>
                    </div>""", unsafe_allow_html=True)

                with c2:
                    if gemini_key:
                        btn_k = f"anom_btn_{row.name}"
                        res_k = f"anom_res_{row.name}"
                        if st.button("🤖 Explain this anomaly", key=btn_k):
                            with st.spinner("Analysing..."):
                                result = call_gemini(
                                    "anomaly",
                                    metric=row["metric"],
                                    direction=str(row.get("direction","")),
                                    deviation_pct=dev,
                                    anomaly_type=str(row.get("anomaly_type","")),
                                    period_summary=summary,
                                )
                            st.session_state[res_k] = result or "No explanation returned."
                        if res_k in st.session_state:
                            st.markdown(
                                f'<div class="ai-box">'
                                f'<strong>🤖 Likely causes & actions:</strong><br><br>'
                                f'{st.session_state[res_k].replace(chr(10),"<br>")}'
                                f'</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="callout">Add your Gemini API key in the sidebar '
                            'to get an AI explanation for this anomaly.</div>',
                            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CHANNELS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Channel Performance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-sub">Revenue attributed by channel</div>', unsafe_allow_html=True)
        colors_ch = [CHANNEL_COLORS.get(c, PURPLE) for c in ch_summary["channel"]]
        fig_ch = go.Figure(go.Bar(
            x=ch_summary["channel"], y=ch_summary["revenue_attributed"],
            marker_color=colors_ch,
            text=[f"${v:,.0f}" for v in ch_summary["revenue_attributed"]],
            textposition="outside",
        ))
        fig_ch.update_layout(**_base_layout(
            height=300, x_title="Channel", y_title="Revenue Attributed (USD)"))
        fig_ch.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_ch, use_container_width=True)
        graph_ai_expander("ch_revenue",
            f"Revenue by channel bar chart. "
            f"Top channel: {ch_summary.sort_values('revenue_attributed',ascending=False).iloc[0]['channel']}.",
            ai_toggle)

    with col2:
        ch_roas = ch_summary.dropna(subset=["roas"]).sort_values("roas")
        st.markdown('<div class="section-sub">roas by channel — target is 2x</div>', unsafe_allow_html=True)
        if not ch_roas.empty:
            colors_roas = [CHANNEL_COLORS.get(c, PURPLE) for c in ch_roas["channel"]]
            fig_roas_ch = go.Figure(go.Bar(
                x=ch_roas["roas"], y=ch_roas["channel"],
                orientation="h", marker_color=colors_roas,
                text=[f"{v:.2f}x" for v in ch_roas["roas"]],
                textposition="outside",
            ))
            fig_roas_ch.add_vline(x=2, line_dash="dot", line_color=RED,
                                   annotation_text="Target 2x",
                                   annotation_font_color=RED)
            fig_roas_ch.update_layout(**_base_layout(
                height=300, x_title="ROAS (x)", y_title="Channel"))
            fig_roas_ch.update_xaxes(ticksuffix="x")
            st.plotly_chart(fig_roas_ch, use_container_width=True)
            graph_ai_expander("ch_roas",
                f"ROAS by channel horizontal bar. Best: "
                f"{ch_roas.iloc[-1]['channel']} at {ch_roas.iloc[-1]['roas']:.2f}x.",
                ai_toggle)
        else:
            st.info("Not enough spend data for ROAS in this period.")

    # ── Channel detail table
    st.markdown('<div class="section-header">Channel Detail Table</div>', unsafe_allow_html=True)
    ch_tbl = ch_summary.copy()
    ch_tbl["spend"]              = ch_tbl["spend"].apply(lambda x: f"${x:,.0f}")
    ch_tbl["revenue_attributed"] = ch_tbl["revenue_attributed"].apply(lambda x: f"${x:,.0f}")
    ch_tbl["roas"]               = ch_tbl["roas"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    ch_tbl["conversion_rate"]    = ch_tbl["conversion_rate"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    ch_tbl.columns               = ["Channel","Spend","Clicks","Conversions","Revenue Attributed","ROAS","Conversion Rate"]
    st.dataframe(ch_tbl, use_container_width=True, hide_index=True)

    # ── Channel trends with its own date picker
    st.markdown('<div class="section-header">Channel Trends Over Time</div>', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns([1, 1, 2])
    with tc1:
        ch_date_start = st.date_input("From##ch", value=d_start,
                                       min_value=DATA_MIN, max_value=DATA_MAX,
                                       key="ch_date_start")
    with tc2:
        ch_date_end = st.date_input("To##ch", value=d_end,
                                     min_value=DATA_MIN, max_value=DATA_MAX,
                                     key="ch_date_end")
    with tc3:
        channel_opts   = ch_full["channel"].unique().tolist()
        default_sel    = [c for c in ["Social","Paid Search","Email"] if c in channel_opts]
        selected_chs   = st.multiselect("Channels", channel_opts, default=default_sel)

    if ch_date_start > ch_date_end:
        st.error("Start date must be before end date.")
    elif selected_chs:
        ch_trend = ch_full[
            (ch_full["channel"].isin(selected_chs)) &
            (ch_full["date"] >= pd.Timestamp(ch_date_start)) &
            (ch_full["date"] <= pd.Timestamp(ch_date_end))
        ]
        fig_trend = go.Figure()
        for ch_name in selected_chs:
            cdata = ch_trend[ch_trend["channel"] == ch_name]
            fig_trend.add_trace(go.Scatter(
                x=cdata["date"], y=cdata["revenue_attributed"],
                mode="lines", name=ch_name,
                line=dict(color=CHANNEL_COLORS.get(ch_name, PURPLE), width=2.5),
            ))
        fig_trend.update_layout(**_base_layout(
            height=300,
            x_title="Date",
            y_title="Revenue Attributed (USD)",
        ))
        fig_trend.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_trend, use_container_width=True)
        graph_ai_expander("ch_trend",
            f"Revenue over time for channels: {', '.join(selected_chs)}. "
            f"Period: {ch_date_start} to {ch_date_end}.", ai_toggle)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Product Performance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cat_rev = (prod.groupby("category")["revenue"].sum()
                   .reset_index().sort_values("revenue", ascending=False))
        cat_rev["icon"]  = cat_rev["category"].map(lambda c: CATEGORY_ICONS.get(c,"📦"))
        cat_rev["label"] = cat_rev.apply(lambda r: f"{r['icon']} {r['category']}", axis=1)
        colors_cat = [CATEGORY_COLORS.get(c, PURPLE) for c in cat_rev["category"]]
        fig_cat = go.Figure(go.Bar(
            x=cat_rev["label"], y=cat_rev["revenue"],
            marker_color=colors_cat,
            text=[f"${v:,.0f}" for v in cat_rev["revenue"]],
            textposition="outside",
        ))
        fig_cat.update_layout(**_base_layout(height=300,
                              x_title="Category", y_title="Revenue (USD)"))
        fig_cat.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_cat, use_container_width=True)
        graph_ai_expander("cat_rev",
            f"Revenue by product category bar chart for {d_start} to {d_end}. "
            f"Top: {cat_rev.iloc[0]['category']} at ${cat_rev.iloc[0]['revenue']:,.0f}.",
            ai_toggle)

    with col2:
        sel_cat = st.selectbox("Drill into category", sorted(prod["category"].unique()))
        top_in_cat = (
            prod[prod["category"] == sel_cat]
            .groupby("product_label")["revenue"].sum()
            .reset_index().sort_values("revenue", ascending=False).head(8)
        )
        fig_top = go.Figure(go.Bar(
            x=top_in_cat["revenue"], y=top_in_cat["product_label"],
            orientation="h",
            marker_color=CATEGORY_COLORS.get(sel_cat, PURPLE),
            text=[f"${v:,.0f}" for v in top_in_cat["revenue"]],
            textposition="outside",
        ))
        fig_top.update_layout(**_base_layout(
            height=300,
            x_title="Revenue (USD)",
            y_title="Product",
        ))
        fig_top.update_xaxes(tickprefix="$")
        st.plotly_chart(fig_top, use_container_width=True)
        graph_ai_expander(f"cat_{sel_cat}",
            f"Top products in {sel_cat} category by revenue. "
            f"Best seller: {top_in_cat.iloc[0]['product_label']} "
            f"at ${top_in_cat.iloc[0]['revenue']:,.0f}.", ai_toggle)

    st.markdown('<div class="section-header">Top 5 Products</div>', unsafe_allow_html=True)
    max_rev = top_products["revenue"].max() if not top_products.empty else 1
    for i, row in enumerate(top_products.itertuples(), 1):
        bar_pct = int((row.revenue / max_rev) * 100)
        st.markdown(f"""
        <div class="kpi-card" style="padding:0.85rem 1.1rem;margin-bottom:0.4rem;">
          <div style="display:flex;align-items:center;gap:0.7rem;">
            <div style="font-family:Plus Jakarta Sans;font-weight:800;
                 font-size:1.1rem;color:{PURPLE};min-width:1.4rem;">#{i}</div>
            <div style="font-size:1.1rem;">{CATEGORY_ICONS.get(row.category,'📦')}</div>
            <div style="flex:1;">
              <div style="font-weight:700;font-size:0.88rem;">{row.product_label}</div>
              <div style="font-size:0.72rem;color:{TEXT_MID};">
                {row.category} · {row.units_sold:.0f} units</div>
              <div style="margin-top:0.3rem;height:4px;background:#F3F4F6;
                   border-radius:4px;overflow:hidden;">
                <div style="height:4px;width:{bar_pct}%;
                     background:linear-gradient(90deg,{PURPLE},{PINK});
                     border-radius:4px;"></div>
              </div>
            </div>
            <div style="text-align:right;min-width:75px;">
              <div style="font-family:Plus Jakarta Sans;font-weight:800;
                   font-size:0.95rem;color:{PURPLE};">${row.revenue:,.0f}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    if not gemini_key:
        st.markdown("""
        <div class="callout" style="font-size:0.88rem;">
          🔑 <strong>Add your Gemini API key in the sidebar</strong> to unlock
          AI-generated summaries, explanations, and recommendations.
          Free at <a href="https://aistudio.google.com" target="_blank"
          style="color:#7C3AED;">aistudio.google.com</a>.
        </div>""", unsafe_allow_html=True)

    # Daily narrative
    st.markdown('<div class="section-header">AI Summary</div>', unsafe_allow_html=True)
    if gemini_key:
        if st.button("Generate Summary", key="gen_narrative"):
            with st.spinner("Writing your summary..."):
                result = call_gemini("narrative", insights=insights,
                                     period_summary=summary,
                                     date_label=period_label)
            st.session_state["narrative_result"] = result or "No result."
        if "narrative_result" in st.session_state:
            st.markdown(
                f'<div class="ai-box">✨ <strong>Summary</strong><br><br>'
                f'{st.session_state["narrative_result"]}</div>',
                unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-sub">Add API key to enable.</div>', unsafe_allow_html=True)

    # Key findings
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    if not insights:
        st.success("No significant changes — metrics are stable. ✅")
    else:
        for ins in insights:
            impact = ins.get("impact","low")
            badge  = {"high":"🔴 High","medium":"🟡 Medium","low":"🟢 Low"}.get(impact, impact)
            arrow  = "↑" if ins.get("direction") == "positive" else "↓"
            st.markdown(f"""
            <div class="insight-card insight-{impact}">
              <span class="badge badge-{impact}">{badge}</span>
              {arrow} {ins.get('finding','')}
            </div>""", unsafe_allow_html=True)

    # Recommendations
    st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
    if gemini_key:
        if st.button("Generate Recommendations", key="gen_recs"):
            with st.spinner("Building recommendations..."):
                result = call_gemini("recommendations", insights=insights,
                                     channel_summary=ch_summary,
                                     period_summary=summary)
            st.session_state["recs_result"] = result or "No result."
        if "recs_result" in st.session_state:
            st.markdown(
                f'<div class="ai-box">🎯 <strong>Recommended Actions</strong><br><br>'
                f'{st.session_state["recs_result"].replace(chr(10),"<br>")}</div>',
                unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-sub">Add API key to enable.</div>', unsafe_allow_html=True)

    # ── Free-form question box ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">💬 Ask Anything</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">Ask a specific question about your data '
        f'({period_label}). Pulse will answer using only your current filtered data.</div>',
        unsafe_allow_html=True)

    user_q = st.text_area(
        "Your question",
        placeholder=(
            "e.g. Why did revenue drop last week?\n"
            "Which channel has the best return on investment?\n"
            "What should I focus on to improve conversion rate?"
        ),
        height=110,
        label_visibility="collapsed",
    )
    if st.button("Ask Pulse 🤖", key="ask_freeform", disabled=not bool(gemini_key)):
        if user_q.strip():
            with st.spinner("Thinking..."):
                answer = call_gemini_freeform(
                    question=user_q,
                    period_summary=summary,
                    insights=insights,
                    channel_summary=ch_summary,
                )
            st.session_state["freeform_answer"] = answer or "No answer returned."
        else:
            st.warning("Please type a question first.")

    if not gemini_key:
        st.caption("Add your Gemini API key in the sidebar to enable this.")

    if "freeform_answer" in st.session_state:
        st.markdown(
            f'<div class="ai-box">💬 <strong>Pulse says:</strong><br><br>'
            f'{st.session_state["freeform_answer"]}</div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Generate Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Download a full written report — ready to share with your team, clients, or stakeholders.</div>', unsafe_allow_html=True)

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        report_mode = st.radio(
            "Date range for report",
            ["Use sidebar range", "Custom range"],
            horizontal=True,
        )

        if report_mode == "Use sidebar range":
            rep_start = d_start
            rep_end   = d_end
            st.markdown(
                f"<div style='font-size:0.8rem;color:{TEXT_MID};margin-top:0.3rem;'>"
                f"📅 {rep_start.strftime('%b %d, %Y')} → {rep_end.strftime('%b %d, %Y')}"
                f"</div>", unsafe_allow_html=True)
        else:
            rep_start = st.date_input("Report from", value=d_start,
                                       min_value=DATA_MIN, max_value=DATA_MAX,
                                       key="rep_start")
            rep_end   = st.date_input("Report to",   value=d_end,
                                       min_value=DATA_MIN, max_value=DATA_MAX,
                                       key="rep_end")
            if rep_start > rep_end:
                st.error("Start must be before end date.")
                rep_start, rep_end = d_start, d_end

        include_ai = st.checkbox("Include AI narratives", value=bool(gemini_key))
        gen_btn    = st.button("📥 Generate Report", type="primary",
                                use_container_width=True)

    with rc2:
        st.markdown(f"""
        <div style='background:white;border-radius:12px;padding:1.1rem;
             border:1px solid {BORDER};font-size:0.83rem;color:{TEXT_MID};'>
          <strong style='color:{TEXT_DARK};'>What's included:</strong><br><br>
          ✅ &nbsp;Executive summary<br>
          ✅ &nbsp;All KPIs with week-over-week comparison<br>
          ✅ &nbsp;Anomalies with deviation details<br>
          ✅ &nbsp;Channel performance breakdown<br>
          ✅ &nbsp;Top products<br>
          ✅ &nbsp;Recommended next actions
        </div>""", unsafe_allow_html=True)

    if gen_btn:
        # ── Data validation ───────────────────────────────────────────────
        rep_df = df_full[
            (df_full["date"] >= pd.Timestamp(rep_start)) &
            (df_full["date"] <= pd.Timestamp(rep_end))
        ]
        if rep_df.empty:
            st.error(
                f"⚠️ No data found between **{rep_start}** and **{rep_end}**. "
                f"Please select dates within the available range: "
                f"**{DATA_MIN.strftime('%b %d, %Y')}** → "
                f"**{DATA_MAX.strftime('%b %d, %Y')}**."
            )
        else:
            rep_days   = (rep_end - rep_start).days + 1
            rep_cmp    = min(7, rep_days)
            rep_sum    = compute_period_summary(rep_df, days=rep_cmp)
            rep_ch     = ch_full[
                (ch_full["date"] >= pd.Timestamp(rep_start)) &
                (ch_full["date"] <= pd.Timestamp(rep_end))
            ]
            rep_prod   = prod_full[
                (prod_full["date"] >= pd.Timestamp(rep_start)) &
                (prod_full["date"] <= pd.Timestamp(rep_end))
            ]
            rep_anom   = get_recent_anomalies(all_anomalies, days=rep_days)
            rep_ch_sum = compute_channel_summary(rep_ch, days=rep_cmp)
            rep_prods  = compute_top_products(rep_prod, n=5, days=rep_cmp)
            rep_ins    = compile_all_insights(rep_sum, rep_anom, rep_ch_sum)

            with st.spinner("Assembling report..."):
                ai_narrative = ai_recs = ai_anom_exp = None
                if include_ai and gemini_key:
                    rep_label = f"{rep_start.strftime('%b %d')} – {rep_end.strftime('%b %d, %Y')}"
                    ai_narrative = call_gemini("narrative", insights=rep_ins,
                                               period_summary=rep_sum,
                                               date_label=rep_label)
                    ai_recs      = call_gemini("recommendations", insights=rep_ins,
                                               channel_summary=rep_ch_sum,
                                               period_summary=rep_sum)
                    if not rep_anom.empty:
                        ta = rep_anom.iloc[0]
                        ai_anom_exp = call_gemini("anomaly",
                                                   metric=ta["metric"],
                                                   direction=str(ta.get("direction","")),
                                                   deviation_pct=float(ta.get("deviation_pct",0)),
                                                   anomaly_type=str(ta.get("anomaly_type","")),
                                                   period_summary=rep_sum)

                report_text = generate_report(
                    period_summary=rep_sum, insights=rep_ins,
                    anomalies=rep_anom, channel_summary=rep_ch_sum,
                    top_products=rep_prods, report_date=rep_end,
                    ai_narrative=ai_narrative,
                    ai_anomaly_explanation=ai_anom_exp,
                    ai_recommendations=ai_recs, save=True,
                )

            st.success(f"✅ Report ready — {len(rep_df):,} days of data included.")
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=report_text,
                file_name=f"pulse_report_{rep_start}_{rep_end}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            with st.expander("Preview report"):
                st.code(report_text, language=None)
