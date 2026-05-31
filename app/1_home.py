"""
Pulse — Home
Project overview, workflow, architecture, and tech stack.
"""

import sys
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

st.markdown("""
<div class="page-title">Home</div>
<div class="page-sub">Project overview</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="home-hero-title">An AI-powered revenue and<br>marketing insight engine</div>
<div class="home-hero-body">
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
        <div class="q-answer">
            Daily KPIs tracked with week-over-week comparison, rolling averages,
            and automatic anomaly flags across revenue, CAC, ROAS, AOV,
            and conversion rate.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="q-card">
        <div class="q-num">Question 2</div>
        <div class="q-question">Why did it change?</div>
        <div class="q-answer">
            Statistical root-cause detection across channels, products, and customer
            cohorts — with AI-generated explanations grounded in your actual numbers,
            not generic advice.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="q-card">
        <div class="q-num">Question 3</div>
        <div class="q-question">What should we do next?</div>
        <div class="q-answer">
            Gemini-generated recommendations prioritised by business impact —
            specific, actionable, and tied directly to the metrics that triggered them.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Workflow ──────────────────────────────────────────────────────────────────
st.markdown('<div class="home-section-tag">How data flows through the system</div>',
            unsafe_allow_html=True)

steps = [
    {
        "num": "Step 1",
        "title": "Ingest",
        "desc": (
            "Five raw CSV files are loaded — transactions, customers, events, "
            "campaigns, and products. Types are enforced and dates are parsed "
            "at this stage."
        ),
        "file": "loader.py",
    },
    {
        "num": "Step 2",
        "title": "Clean",
        "desc": (
            "Refunded transactions are removed, nulls are handled with explicit "
            "rules, channel names are normalised, and organic traffic is "
            "labelled consistently."
        ),
        "file": "cleaner.py",
    },
    {
        "num": "Step 3",
        "title": "Transform",
        "desc": (
            "Four analytical Parquet tables are built — daily metrics, product "
            "sales, marketing channel, and events — each with a clear grain "
            "and purpose."
        ),
        "file": "transformer.py",
    },
    {
        "num": "Step 4",
        "title": "Analyse",
        "desc": (
            "Rolling statistics, week-over-week change, Z-score anomaly "
            "detection, RFM customer scoring, cohort retention, and a "
            "90-day Holt-Winters forecast are computed."
        ),
        "file": "analytics/",
    },
    {
        "num": "Step 5",
        "title": "Narrate",
        "desc": (
            "Rule-based insights are structured first, then passed to "
            "Gemini 2.0 Flash. The AI receives context and findings — "
            "not raw data — keeping output focused and consistent."
        ),
        "file": "gemini.py",
    },
]

cols = st.columns(5, gap="small")
for col, step in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="wf-card">
            <div class="wf-step-num">{step['num']}</div>
            <div class="wf-step-title">{step['title']}</div>
            <div class="wf-step-desc">{step['desc']}</div>
            <span class="wf-file">{step['file']}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Architecture ──────────────────────────────────────────────────────────────
st.markdown('<div class="home-section-tag">Architecture — what each layer does</div>',
            unsafe_allow_html=True)

arch = [
    {
        "icon": "🗄",
        "icon_bg": "#E1F5EE",
        "title": "ETL pipeline",
        "desc": (
            "Loads raw CSVs, enforces business rules and data types, then writes "
            "four analytical Parquet tables to disk. Runs once locally; outputs "
            "are committed and read directly by the dashboard on Streamlit Cloud."
        ),
        "files": ["loader.py", "cleaner.py", "transformer.py", "pipeline.py"],
    },
    {
        "icon": "📊",
        "icon_bg": "#EEEDFE",
        "title": "Analytics engine",
        "desc": (
            "Computes all metrics and statistical signals — rolling averages, "
            "WoW change, anomaly detection with configurable thresholds, RFM "
            "quartile scoring, cohort retention matrices, and time-series "
            "forecasting."
        ),
        "files": ["metrics.py", "anomaly.py", "segmentation.py",
                  "forecasting.py", "cohort.py"],
    },
    {
        "icon": "🧠",
        "icon_bg": "#FAEEDA",
        "title": "AI layer",
        "desc": (
            "Insights are structured by rule-based logic before reaching the AI. "
            "Gemini 2.0 Flash is called via direct REST — no SDK dependency — "
            "with tightly scoped prompts for narrative, anomaly explanation, "
            "and recommendations."
        ),
        "files": ["gemini.py", "insights.py"],
    },
    {
        "icon": "📋",
        "icon_bg": "#E6F1FB",
        "title": "Dashboard",
        "desc": (
            "Seven-page Streamlit app with Plotly charts, per-chart AI insights, "
            "an anomaly investigation view, RFM and cohort analysis, a 90-day "
            "forecast, and an exportable daily report with AI narrative."
        ),
        "files": ["dashboard.py", "daily_report.py"],
    },
]

a1, a2 = st.columns(2, gap="small")
for i, (col, card) in enumerate(zip([a1, a2, a1, a2], arch)):
    with col:
        files_html = "".join(
            f'<span class="arch-file">{f}</span>' for f in card["files"]
        )
        st.markdown(f"""
        <div class="arch-card" style="margin-bottom: 0.6rem;">
            <div class="arch-icon-wrap" style="background:{card['icon_bg']};">
                <span style="font-size:0.9rem;">{card['icon']}</span>
            </div>
            <div class="arch-title">{card['title']}</div>
            <div class="arch-desc">{card['desc']}</div>
            <div class="arch-files">{files_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown('<div class="home-section-tag">Tech stack</div>',
            unsafe_allow_html=True)

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
        <div class="stack-cell" style="margin-bottom: 0.5rem;">
            <div class="stack-layer">{layer}</div>
            <div class="stack-tool">{tool}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer note ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="home-note">
No API key is required to explore the dashboard — all analytics run locally on
the processed data. Add a free Gemini API key in the sidebar to unlock AI
summaries, anomaly explanations, and report generation.
</div>
""", unsafe_allow_html=True)
