"""
Pulse — AI Insight Engine
Entry point and shared configuration.
Run with: streamlit run app/dashboard.py
"""

import os
import sys
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Pulse — AI Insight Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] > div {
    background: #111827 !important;
}
section[data-testid="stSidebarNav"] a:first-child {
    display: none !important;
}
section[data-testid="stSidebar"] * {
    color: #D1D5DB !important;
}
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #F9FAFB !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] label {
    color: #9CA3AF !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.6rem 0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #1D4ED8 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    width: 100% !important;
    padding: 0.5rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1E40AF !important;
}

/* ── Sidebar nav ── */
section[data-testid="stSidebarNav"] {
    padding-top: 0 !important;
}
section[data-testid="stSidebarNav"] a {
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    color: #9CA3AF !important;
    padding: 0.4rem 0.8rem !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebarNav"] a:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #F9FAFB !important;
}
section[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: rgba(255,255,255,0.1) !important;
    color: #F9FAFB !important;
    font-weight: 500 !important;
}

/* ── KPI cards ── */
.kpi-card {
    background: #FFFFFF;
    border: 0.5px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.kpi-label {
    font-size: clamp(0.58rem, 0.9vw, 0.65rem);
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.kpi-value {
    font-size: clamp(1.1rem, 2vw, 1.45rem);
    font-weight: 600;
    color: #111827;
    line-height: 1.2;
}
.kpi-delta-up {
    display: inline-block;
    font-size: clamp(0.6rem, 0.9vw, 0.68rem);
    font-weight: 600;
    color: #15803D;
    background: #DCFCE7;
    padding: 2px 8px;
    border-radius: 20px;
}
.kpi-delta-down {
    display: inline-block;
    font-size: clamp(0.6rem, 0.9vw, 0.68rem);
    font-weight: 600;
    color: #B91C1C;
    background: #FEE2E2;
    padding: 2px 8px;
    border-radius: 20px;
}
.kpi-delta-neu {
    display: inline-block;
    font-size: clamp(0.6rem, 0.9vw, 0.68rem);
    font-weight: 600;
    color: #6B7280;
    background: #F3F4F6;
    padding: 2px 8px;
    border-radius: 20px;
}
.kpi-hint {
    font-size: clamp(0.58rem, 0.8vw, 0.63rem);
    color: #9CA3AF;
    margin-top: 2px;
}

/* ── Section headers ── */
.section-header {
    font-size: clamp(0.68rem, 1vw, 0.75rem);
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding-bottom: 0.5rem;
    border-bottom: 0.5px solid #E5E7EB;
    margin-bottom: 0.8rem;
    margin-top: 0.5rem;
}

/* ── Page title ── */
.page-title {
    font-size: clamp(1.1rem, 2.5vw, 1.3rem);
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.15rem;
}
.page-sub {
    font-size: clamp(0.7rem, 1.2vw, 0.78rem);
    color: #9CA3AF;
    margin-bottom: 1.2rem;
}

/* ── AI strip ── */
.ai-strip {
    background: #FFFFFF;
    border: 0.5px solid #E5E7EB;
    border-left: 3px solid #1D4ED8;
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem;
    font-size: clamp(0.75rem, 1.2vw, 0.83rem);
    color: #374151;
    line-height: 1.7;
    margin-bottom: 1.2rem;
}
.ai-strip-label {
    font-size: clamp(0.58rem, 0.9vw, 0.62rem);
    font-weight: 600;
    color: #1D4ED8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.ai-placeholder {
    background: #F9FAFB;
    border: 1px dashed #E5E7EB;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: clamp(0.7rem, 1.1vw, 0.78rem);
    color: #9CA3AF;
    margin-bottom: 1.2rem;
    text-align: center;
}

/* ── Chart card ── */
.chart-card {
    background: #FFFFFF;
    border: 0.5px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.1rem 0.5rem;
}
.chart-title {
    font-size: clamp(0.72rem, 1.1vw, 0.8rem);
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.15rem;
}
.chart-sub {
    font-size: clamp(0.65rem, 1vw, 0.72rem);
    color: #9CA3AF;
    margin-bottom: 0.6rem;
}

/* ── AI insight expander ── */
.ai-insight-box {
    background: #F8FAFF;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    font-size: clamp(0.7rem, 1.1vw, 0.78rem);
    color: #374151;
    line-height: 1.65;
    margin-top: 0.3rem;
}

/* ── Tables ── */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: clamp(0.7rem, 1.1vw, 0.78rem);
}
.data-table th {
    font-size: clamp(0.58rem, 0.9vw, 0.63rem);
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0 0 0.5rem;
    border-bottom: 0.5px solid #E5E7EB;
    text-align: left;
}
.data-table th.right, .data-table td.right {
    text-align: right;
}
.data-table td {
    padding: 0.5rem 0;
    color: #374151;
    border-bottom: 0.5px solid #F3F4F6;
}
.data-table tr:last-child td {
    border-bottom: none;
}
.td-good { color: #15803D; font-weight: 500; }
.td-bad  { color: #B91C1C; font-weight: 500; }
.td-bold { font-weight: 500; color: #111827; }

/* ── Badges ── */
.badge-up   { background:#DCFCE7; color:#15803D; font-size:clamp(0.58rem,0.9vw,0.65rem); font-weight:600; padding:2px 8px; border-radius:20px; white-space:nowrap; }
.badge-down { background:#FEE2E2; color:#B91C1C; font-size:clamp(0.58rem,0.9vw,0.65rem); font-weight:600; padding:2px 8px; border-radius:20px; white-space:nowrap; }
.badge-warn { background:#FEF3C7; color:#92400E; font-size:clamp(0.58rem,0.9vw,0.65rem); font-weight:600; padding:2px 8px; border-radius:20px; white-space:nowrap; }
.badge-neu  { background:#F3F4F6; color:#6B7280; font-size:clamp(0.58rem,0.9vw,0.65rem); font-weight:600; padding:2px 8px; border-radius:20px; white-space:nowrap; }

/* ── Home page specific ── */
.home-hero-title {
    font-size: clamp(1.4rem, 3vw, 2rem);
    font-weight: 600;
    color: #111827;
    line-height: 1.3;
    margin-bottom: 0.6rem;
}
.home-hero-body {
    font-size: clamp(0.8rem, 1.3vw, 0.92rem);
    color: #6B7280;
    line-height: 1.75;
    max-width: 680px;
}
.home-section-tag {
    font-size: clamp(0.6rem, 0.9vw, 0.68rem);
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    padding-bottom: 0.5rem;
    border-bottom: 0.5px solid #E5E7EB;
    margin-bottom: 1rem;
    margin-top: 2rem;
}
.q-card {
    background: #FFFFFF;
    border: 0.5px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.q-num {
    font-size: 0.62rem;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.q-question {
    font-size: clamp(0.78rem, 1.2vw, 0.88rem);
    font-weight: 600;
    color: #111827;
    line-height: 1.4;
    margin-bottom: 0.4rem;
}
.q-answer {
    font-size: clamp(0.7rem, 1.1vw, 0.78rem);
    color: #6B7280;
    line-height: 1.6;
}
.wf-card {
    background: #FFFFFF;
    border: 0.5px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.wf-step-num {
    font-size: 0.6rem;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.wf-step-title {
    font-size: clamp(0.78rem, 1.2vw, 0.86rem);
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.35rem;
}
.wf-step-desc {
    font-size: clamp(0.68rem, 1vw, 0.76rem);
    color: #6B7280;
    line-height: 1.6;
    margin-bottom: 0.6rem;
}
.wf-file {
    display: inline-block;
    font-size: 0.68rem;
    font-family: 'Courier New', monospace;
    padding: 2px 8px;
    border-radius: 6px;
    background: #F3F4F6;
    color: #374151;
    border: 0.5px solid #E5E7EB;
}
.arch-card {
    background: #FFFFFF;
    border: 0.5px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.arch-icon-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 7px;
    margin-bottom: 0.6rem;
}
.arch-title {
    font-size: clamp(0.78rem, 1.2vw, 0.86rem);
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.4rem;
}
.arch-desc {
    font-size: clamp(0.68rem, 1vw, 0.76rem);
    color: #6B7280;
    line-height: 1.65;
    margin-bottom: 0.6rem;
}
.arch-files {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
}
.arch-file {
    font-size: 0.65rem;
    font-family: 'Courier New', monospace;
    padding: 2px 7px;
    border-radius: 5px;
    background: #F3F4F6;
    color: #374151;
    border: 0.5px solid #E5E7EB;
}
.stack-cell {
    background: #FFFFFF;
    border: 0.5px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    text-align: center;
}
.stack-layer {
    font-size: 0.6rem;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}
.stack-tool {
    font-size: clamp(0.72rem, 1.1vw, 0.8rem);
    font-weight: 600;
    color: #111827;
}
.home-note {
    font-size: clamp(0.68rem, 1vw, 0.76rem);
    color: #9CA3AF;
    line-height: 1.65;
    font-style: italic;
    margin-top: 0.75rem;
}

/* ── Anomaly master-detail ── */
.anomaly-log-item {
    padding: 0.7rem 0.9rem;
    border-bottom: 0.5px solid #F3F4F6;
    cursor: pointer;
    border-radius: 8px;
    margin-bottom: 2px;
}
.anomaly-log-item:hover {
    background: #F9FAFB;
}
.anomaly-log-metric {
    font-size: clamp(0.72rem, 1.1vw, 0.8rem);
    font-weight: 600;
    color: #111827;
    margin-bottom: 2px;
}
.anomaly-log-date {
    font-size: clamp(0.62rem, 0.9vw, 0.68rem);
    color: #9CA3AF;
}
.anomaly-log-type {
    font-size: clamp(0.62rem, 0.9vw, 0.68rem);
    color: #6B7280;
    margin-top: 2px;
}

/* ── Responsive breakpoints ── */
@media (max-width: 1280px) {
    .kpi-card { height: 120px !important; padding: 0.85rem 1rem !important; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}
@media (max-width: 1024px) {
    .kpi-card { height: 115px !important; }
    .home-hero-title { font-size: 1.5rem !important; }
    .arch-desc { font-size: 0.72rem !important; }
}
@media (max-width: 900px) {
    .kpi-card { height: auto !important; min-height: 100px; }
    .kpi-value { font-size: 1.1rem !important; }
    .page-title { font-size: 1.1rem !important; }
    .home-hero-title { font-size: 1.3rem !important; }
    .ai-strip { font-size: 0.75rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.2rem 0 1rem;">
        <div style="font-size:1.05rem; font-weight:600; color:#F9FAFB; letter-spacing:-0.01em;">Pulse</div>
        <div style="font-size:0.72rem; color:#6B7280; margin-top:2px;">AI insight engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; color:#6B7280;
    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;">
    Controls
    </div>
    """, unsafe_allow_html=True)

    date_window = st.selectbox(
        "Date window",
        ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days"],
        index=0,
    )

    top_n = st.selectbox(
        "Top N products",
        [5, 10, 20],
        index=0,
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; color:#6B7280;
    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;">
    Gemini API key
    </div>
    """, unsafe_allow_html=True)

    gemini_key = st.text_input(
        "Gemini API key",
        type="password",
        placeholder="Paste key to unlock AI",
        label_visibility="collapsed",
    )

    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        st.markdown("""
        <div style="font-size:0.7rem; color:#34D399; margin-top:0.3rem;">
        AI insights enabled
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.7rem; color:#6B7280; margin-top:0.3rem;">
        Get a free key at aistudio.google.com
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    ai_on = st.toggle("Show AI insights on charts", value=False)

# ── Persist to session state ─────────────────────────────────────────────────
st.session_state["date_window"] = date_window
st.session_state["top_n"]       = int(top_n)
st.session_state["gemini_key"]  = gemini_key
st.session_state["ai_on"]       = ai_on

# ── Landing content ───────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">Welcome to Pulse</div>
<div class="page-sub">Select a page from the sidebar to get started.</div>
""", unsafe_allow_html=True)
