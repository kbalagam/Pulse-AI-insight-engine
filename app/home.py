"""
Pulse — AI Insight Engine
Entry point, shared configuration, and home page.
Run with: streamlit run app/Home.py
"""

import sys
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP  = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

st.set_page_config(
    page_title="Pulse — AI Insight Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import inject_styles
inject_styles()

# ── Home page content ─────────────────────────────────────────────────────────

st.markdown("""
<div class="home-hero-title" style="max-width:100%;">
An AI-powered revenue and marketing insight engine
</div>
<div class="home-hero-body" style="max-width:100%;">
Pulse connects raw e-commerce data to plain-English decisions. It tracks daily KPIs,
detects statistical anomalies, segments customers by behaviour, forecasts revenue, and
uses Gemini to explain what changed, why it changed, and what to do next —
without manual analysis.
</div>
""", unsafe_allow_html=True)

# ── Three questions ───────────────────────────────────────────────────────────
st.markdown('<div class="home-section-tag">Three questions this dashboard answers</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="small")
with c1:
    st.markdown("""
    <div class="q-card">
        <div class="q-num">Question 1</div>
        <div class="q-question">What changed in our key metrics?</div>
        <div class="q-answer">Daily KPIs tracked with week-over-week comparison,
        rolling averages, and automatic anomaly flags across revenue, CAC, ROAS,
        AOV, and conversion rate.</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="q-card">
        <div class="q-num">Question 2</div>
        <div class="q-question">Why did it change?</div>
        <div class="q-answer">Statistical root-cause detection across channels,
        products, and customer cohorts — with AI-generated explanations grounded
        in your actual numbers, not generic advice.</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="q-card">
        <div class="q-num">Question 3</div>
        <div class="q-question">What should we do next?</div>
        <div class="q-answer">Gemini-generated recommendations prioritised by
        business impact — specific, actionable, and tied directly to the metrics
        that triggered them.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Workflow ──────────────────────────────────────────────────────────────────
st.markdown('<div class="home-section-tag">How data flows through the system</div>',
            unsafe_allow_html=True)

steps = [
    ("Step 1", "Ingest",
     "Five raw CSV files are loaded — transactions, customers, events, campaigns, "
     "and products. Types are enforced and dates are parsed at this stage.",
     "loader.py"),
    ("Step 2", "Clean",
     "Refunded transactions are removed, nulls are handled with explicit rules, "
     "channel names are normalised, and organic traffic is labelled consistently.",
     "cleaner.py"),
    ("Step 3", "Transform",
     "Four analytical Parquet tables are built — daily metrics, product sales, "
     "marketing channel, and events — each with a clear grain and purpose.",
     "transformer.py"),
    ("Step 4", "Analyse",
     "Rolling statistics, week-over-week change, Z-score anomaly detection, RFM "
     "customer scoring, cohort retention, and a 90-day Holt-Winters forecast are computed.",
     "analytics/"),
    ("Step 5", "Narrate",
     "Rule-based insights are structured first, then passed to Gemini 2.0 Flash. "
     "The AI receives context and findings — not raw data — keeping output focused "
     "and consistent.",
     "gemini.py"),
]

cols = st.columns(5, gap="small")
for col, (num, title, desc, file) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="wf-card">
            <div class="wf-step-num">{num}</div>
            <div class="wf-step-title">{title}</div>
            <div class="wf-step-desc">{desc}</div>
            <span class="wf-file">{file}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Architecture ──────────────────────────────────────────────────────────────
st.markdown('<div class="home-section-tag">Architecture — what each layer does</div>',
            unsafe_allow_html=True)

arch = [
    ("🗄", "#E1F5EE", "ETL pipeline",
     "Loads raw CSVs, enforces business rules and data types, then writes four "
     "analytical Parquet tables to disk. Runs once locally; outputs are committed "
     "and read directly by the dashboard on Streamlit Cloud.",
     ["loader.py", "cleaner.py", "transformer.py", "pipeline.py"]),
    ("📊", "#EEEDFE", "Analytics engine",
     "Computes all metrics and statistical signals — rolling averages, WoW change, "
     "anomaly detection with configurable thresholds, RFM quartile scoring, cohort "
     "retention matrices, and time-series forecasting.",
     ["metrics.py", "anomaly.py", "segmentation.py", "forecasting.py", "cohort.py"]),
    ("🧠", "#FAEEDA", "AI layer",
     "Insights are structured by rule-based logic before reaching the AI. Gemini "
     "2.0 Flash is called via direct REST — no SDK dependency — with tightly scoped "
     "prompts for narrative, anomaly explanation, and recommendations.",
     ["gemini.py", "insights.py"]),
    ("📋", "#E6F1FB", "Dashboard",
     "Seven-page Streamlit app with Plotly charts, per-chart AI insights, an anomaly "
     "investigation view, RFM and cohort analysis, a 90-day forecast, and an "
     "exportable daily report with AI narrative.",
     ["Home.py", "daily_report.py"]),
]

a1, a2 = st.columns(2, gap="small")
for i, (icon, bg, title, desc, files) in enumerate(arch):
    col = a1 if i % 2 == 0 else a2
    with col:
        files_html = "".join(f'<span class="arch-file">{f}</span>' for f in files)
        st.markdown(f"""
        <div class="arch-card">
            <div class="arch-icon-wrap" style="background:{bg};">
                <span style="font-size:0.9rem;">{icon}</span>
            </div>
            <div class="arch-title">{title}</div>
            <div class="arch-desc">{desc}</div>
            <div class="arch-files">{files_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown('<div class="home-section-tag">Tech stack</div>', unsafe_allow_html=True)

stack = [
    ("Processing",  "pandas · NumPy"),
    ("Storage",     "Parquet · pyarrow"),
    ("Forecasting", "statsmodels"),
    ("AI",          "Gemini 2.0 Flash"),
    ("Frontend",    "Streamlit"),
    ("Charts",      "Plotly"),
    ("Language",    "Python 3.11"),
    ("Hosting",     "Streamlit Cloud"),
]

s_cols = st.columns(4, gap="small")
for i, (layer, tool) in enumerate(stack):
    with s_cols[i % 4]:
        st.markdown(f"""
        <div class="stack-cell" style="margin-bottom:0.5rem;">
            <div class="stack-layer">{layer}</div>
            <div class="stack-tool">{tool}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="home-note">
No API key is required to explore the dashboard — all analytics run locally on
the processed data. Add a free Gemini API key in the sidebar to unlock AI
summaries, anomaly explanations, and report generation.
</div>
""", unsafe_allow_html=True)
