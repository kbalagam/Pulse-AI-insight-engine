import os
import datetime
import streamlit as st

DATA_MIN = datetime.date(2021, 1, 1)
DATA_MAX = datetime.date(2023, 12, 31)

def inject_styles():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { display: none !important; }

/* ── Sidebar base ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div,
section[data-testid="stSidebar"] > div > div > div {
    background: #FFFFFF !important;
}
section[data-testid="stSidebar"] * { color: #374151 !important; }

/* ── Kill ALL white/colored backgrounds on sidebar widgets ── */
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stRadio > div,
section[data-testid="stSidebar"] .stRadio > div > div,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] .stDateInput,
section[data-testid="stSidebar"] .stDateInput > div,
section[data-testid="stSidebar"] .stDateInput input,
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] .stTextInput > div > div,
section[data-testid="stSidebar"] .stToggle,
section[data-testid="stSidebar"] .stToggle > div,
section[data-testid="stSidebar"] .stToggle label,
section[data-testid="stSidebar"] [data-baseweb="radio"],
section[data-testid="stSidebar"] [data-baseweb="radio"] > div,
section[data-testid="stSidebar"] [role="radiogroup"],
section[data-testid="stSidebar"] [role="radiogroup"] > div {
    background: transparent !important;
    background-color: transparent !important;
}

/* ── Hide default radio circles ── */
section[data-testid="stSidebar"] input[type="radio"] {
    display: none !important;
}

/* ── Radio group — pill buttons ── */
section[data-testid="stSidebar"] [role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    padding: 0 !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 4px 10px !important;
    border-radius: 20px !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    background: #F3F4F6 !important;
    border: 1px solid #E5E7EB !important;
    color: #6B7280 !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #EFF6FF !important;
    border-color: #3B82F6 !important;
    color: #1D4ED8 !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] [role="radiogroup"] [aria-checked="true"] label,
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ── Input fields ── */
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stTextInput > div > div,
section[data-testid="stSidebar"] .stDateInput input {
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #9CA3AF !important;
}

/* ── Toggle ── */
section[data-testid="stSidebar"] .stToggle p {
    color: #374151 !important;
    font-size: 0.78rem !important;
}

/* ── Labels ── */
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #9CA3AF !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* ── Dividers ── */
section[data-testid="stSidebar"] hr {
    border-color: #E5E7EB !important;
    margin: 0.6rem 0 !important;
}

/* ── Nav — hide scroll gradients ── */
section[data-testid="stSidebarNav"] {
    padding-top: 0 !important;
    overflow: visible !important;
    max-height: none !important;
}
section[data-testid="stSidebarNav"]::before,
section[data-testid="stSidebarNav"]::after {
    display: none !important;
    background: none !important;
}
section[data-testid="stSidebarNav"] ul {
    overflow: visible !important;
    padding-bottom: 0 !important;
}
section[data-testid="stSidebarNav"] > ul::before,
section[data-testid="stSidebarNav"] > ul::after {
    display: none !important;
}
[data-testid="stSidebarNavSeparator"] { display: none !important; }

/* ── Nav links ── */
section[data-testid="stSidebarNav"] a {
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    color: #6B7280 !important;
    padding: 0.45rem 0.9rem !important;
    border-radius: 6px !important;
    border-left: 2px solid transparent !important;
    display: block !important;
}
section[data-testid="stSidebarNav"] a:hover {
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    border-left: 2px solid #3B82F6 !important;
}
section[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    font-weight: 600 !important;
    border-left: 2px solid #1D4ED8 !important;
}

/* ── KPI cards ── */
.kpi-card { background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:12px; padding:1rem 1.1rem; height:130px; display:flex; flex-direction:column; justify-content:space-between; }
.kpi-label { font-size:clamp(0.58rem,0.9vw,0.65rem); font-weight:600; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.08em; }
.kpi-value { font-size:clamp(1.1rem,2vw,1.45rem); font-weight:600; color:#111827; line-height:1.2; }
.kpi-delta-up { display:inline-block; font-size:clamp(0.6rem,0.9vw,0.68rem); font-weight:600; color:#15803D; background:#DCFCE7; padding:2px 8px; border-radius:20px; }
.kpi-delta-down { display:inline-block; font-size:clamp(0.6rem,0.9vw,0.68rem); font-weight:600; color:#B91C1C; background:#FEE2E2; padding:2px 8px; border-radius:20px; }
.kpi-delta-neu { display:inline-block; font-size:clamp(0.6rem,0.9vw,0.68rem); font-weight:600; color:#6B7280; background:#F3F4F6; padding:2px 8px; border-radius:20px; }
.kpi-hint { font-size:clamp(0.58rem,0.8vw,0.63rem); color:#9CA3AF; margin-top:2px; }
.section-header { font-size:clamp(0.68rem,1vw,0.75rem); font-weight:600; color:#374151; text-transform:uppercase; letter-spacing:0.07em; padding-bottom:0.5rem; border-bottom:0.5px solid #E5E7EB; margin-bottom:0.8rem; margin-top:0.5rem; }
.page-title { font-size:clamp(1.1rem,2.5vw,1.3rem); font-weight:600; color:#111827; margin-bottom:0.15rem; }
.page-sub { font-size:clamp(0.7rem,1.2vw,0.78rem); color:#9CA3AF; margin-bottom:1.2rem; }
.ai-strip { background:#FFFFFF; border:0.5px solid #E5E7EB; border-left:3px solid #1D4ED8; border-radius:0 10px 10px 0; padding:0.85rem 1.1rem; font-size:clamp(0.75rem,1.2vw,0.83rem); color:#374151; line-height:1.7; margin-bottom:1.2rem; }
.ai-strip-label { font-size:clamp(0.58rem,0.9vw,0.62rem); font-weight:600; color:#1D4ED8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem; }
.ai-placeholder { background:#F9FAFB; border:1px dashed #E5E7EB; border-radius:10px; padding:0.75rem 1rem; font-size:clamp(0.7rem,1.1vw,0.78rem); color:#9CA3AF; margin-bottom:1.2rem; text-align:center; }
.chart-title { font-size:clamp(0.72rem,1.1vw,0.8rem); font-weight:600; color:#111827; margin-bottom:0.15rem; }
.chart-sub { font-size:clamp(0.65rem,1vw,0.72rem); color:#9CA3AF; margin-bottom:0.6rem; }
.ai-insight-box { background:#F8FAFF; border-radius:8px; padding:0.7rem 0.9rem; font-size:clamp(0.7rem,1.1vw,0.78rem); color:#374151; line-height:1.65; margin-top:0.3rem; }
.data-table { width:100%; border-collapse:collapse; font-size:clamp(0.7rem,1.1vw,0.78rem); }
.data-table th { font-size:clamp(0.58rem,0.9vw,0.63rem); font-weight:600; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.07em; padding:0 0 0.5rem; border-bottom:0.5px solid #E5E7EB; text-align:left; }
.data-table th.right, .data-table td.right { text-align:right; }
.data-table td { padding:0.5rem 0; color:#374151; border-bottom:0.5px solid #F3F4F6; }
.data-table tr:last-child td { border-bottom:none; }
.td-good { color:#15803D; font-weight:500; } .td-bad { color:#B91C1C; font-weight:500; } .td-bold { font-weight:500; color:#111827; }
.badge-up { background:#DCFCE7; color:#15803D; font-size:clamp(0.58rem,0.9vw,0.65rem); font-weight:600; padding:2px 8px; border-radius:20px; white-space:nowrap; }
.badge-down { background:#FEE2E2; color:#B91C1C; font-size:clamp(0.58rem,0.9vw,0.65rem); font-weight:600; padding:2px 8px; border-radius:20px; white-space:nowrap; }
.badge-warn { background:#FEF3C7; color:#92400E; font-size:clamp(0.58rem,0.9vw,0.65rem); font-weight:600; padding:2px 8px; border-radius:20px; white-space:nowrap; }
.badge-neu { background:#F3F4F6; color:#6B7280; font-size:clamp(0.58rem,0.9vw,0.65rem); font-weight:600; padding:2px 8px; border-radius:20px; white-space:nowrap; }
.home-hero-title { font-size:clamp(1.4rem,3vw,2rem); font-weight:600; color:#111827; line-height:1.3; margin-bottom:0.6rem; }
.home-hero-body { font-size:clamp(0.8rem,1.3vw,0.92rem); color:#6B7280; line-height:1.75; max-width:680px; }
.home-section-tag { font-size:clamp(0.6rem,0.9vw,0.68rem); font-weight:600; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.09em; padding-bottom:0.5rem; border-bottom:0.5px solid #E5E7EB; margin-bottom:1rem; margin-top:2rem; }
.q-card { background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:12px; padding:1.1rem 1.2rem; height:180px; display:flex; flex-direction:column; justify-content:flex-start; }
.q-num { font-size:0.62rem; font-weight:600; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem; }
.q-question { font-size:clamp(0.78rem,1.2vw,0.88rem); font-weight:600; color:#111827; line-height:1.4; margin-bottom:0.4rem; }
.q-answer { font-size:clamp(0.7rem,1.1vw,0.78rem); color:#6B7280; line-height:1.6; }
.wf-card { background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:12px; padding:1.1rem 1.2rem; height:260px; display:flex; flex-direction:column; justify-content:flex-start; }
.wf-step-num { font-size:0.6rem; font-weight:600; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.35rem; }
.wf-step-title { font-size:clamp(0.78rem,1.2vw,0.86rem); font-weight:600; color:#111827; margin-bottom:0.35rem; }
.wf-step-desc { font-size:clamp(0.68rem,1vw,0.76rem); color:#6B7280; line-height:1.6; margin-bottom:0.6rem; flex:1; }
.wf-file { display:inline-block; font-size:0.68rem; font-family:'Courier New',monospace; padding:2px 8px; border-radius:6px; background:#F3F4F6; color:#374151; border:0.5px solid #E5E7EB; margin-top:auto; }
.arch-card { background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:12px; padding:1.1rem 1.2rem; height:220px; display:flex; flex-direction:column; justify-content:flex-start; margin-bottom:0.6rem; }
.arch-icon-wrap { display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:7px; margin-bottom:0.6rem; }
.arch-title { font-size:clamp(0.78rem,1.2vw,0.86rem); font-weight:600; color:#111827; margin-bottom:0.4rem; }
.arch-desc { font-size:clamp(0.68rem,1vw,0.76rem); color:#6B7280; line-height:1.65; margin-bottom:0.6rem; flex:1; }
.arch-files { display:flex; flex-wrap:wrap; gap:4px; margin-top:auto; }
.arch-file { font-size:0.65rem; font-family:'Courier New',monospace; padding:2px 7px; border-radius:5px; background:#F3F4F6; color:#374151; border:0.5px solid #E5E7EB; }
.stack-cell { background:#FFFFFF; border:0.5px solid #E5E7EB; border-radius:10px; padding:0.75rem 0.9rem; text-align:center; }
.stack-layer { font-size:0.6rem; font-weight:600; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.25rem; }
.stack-tool { font-size:clamp(0.72rem,1.1vw,0.8rem); font-weight:600; color:#111827; }
.home-note { font-size:clamp(0.68rem,1vw,0.76rem); color:#9CA3AF; line-height:1.65; font-style:italic; margin-top:0.75rem; }
.anomaly-log-item { padding:0.7rem 0.9rem; border-bottom:0.5px solid #F3F4F6; cursor:pointer; border-radius:8px; margin-bottom:2px; }
.anomaly-log-item:hover { background:#F9FAFB; }
.anomaly-log-metric { font-size:clamp(0.72rem,1.1vw,0.8rem); font-weight:600; color:#111827; margin-bottom:2px; }
.anomaly-log-date { font-size:clamp(0.62rem,0.9vw,0.68rem); color:#9CA3AF; }
.anomaly-log-type { font-size:clamp(0.62rem,0.9vw,0.68rem); color:#6B7280; margin-top:2px; }
@media (max-width:1280px) { .kpi-card { height:120px !important; padding:0.85rem 1rem !important; } .block-container { padding-left:1rem !important; padding-right:1rem !important; } }
@media (max-width:1024px) { .kpi-card { height:115px !important; } .home-hero-title { font-size:1.5rem !important; } .arch-desc { font-size:0.72rem !important; } .q-card { height:auto !important; } .wf-card { height:auto !important; } .arch-card { height:auto !important; } }
@media (max-width:900px) { .kpi-card { height:auto !important; min-height:100px; } .kpi-value { font-size:1.1rem !important; } .page-title { font-size:1.1rem !important; } .home-hero-title { font-size:1.3rem !important; } .ai-strip { font-size:0.75rem !important; } }
</style>
""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:

        st.markdown("""
        <div style="padding:0.6rem 0 1rem;">
            <div style="font-size:1.05rem; font-weight:600; color:#111827;
                        letter-spacing:-0.01em;">Pulse</div>
            <div style="font-size:0.7rem; color:#9CA3AF; margin-top:2px;">
                AI insight engine
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Date window ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:0.62rem; font-weight:600; color:#9CA3AF;
        text-transform:uppercase; letter-spacing:0.09em; margin-bottom:0.6rem;">
        Date window
        </div>
        """, unsafe_allow_html=True)

        preset = st.radio(
            "Date window",
            ["7d", "14d", "30d", "90d", "Custom"],
            index=["7d", "14d", "30d", "90d", "Custom"].index(
                st.session_state.get("sidebar_preset", "30d")
            ),
            horizontal=True,
            label_visibility="collapsed",
            key="sidebar_preset",
        )

        if preset != "Custom":
            days_map   = {"7d": 7, "14d": 14, "30d": 30, "90d": 90}
            end_date   = DATA_MAX
            start_date = end_date - datetime.timedelta(days=days_map[preset] - 1)
        else:
            date_range = st.date_input(
                "Select range",
                value=(
                    st.session_state.get("start_date", DATA_MAX - datetime.timedelta(days=29)),
                    st.session_state.get("end_date", DATA_MAX),
                ),
                min_value=DATA_MIN,
                max_value=DATA_MAX,
                label_visibility="collapsed",
                key="sidebar_daterange",
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = DATA_MAX - datetime.timedelta(days=29)
                end_date   = DATA_MAX

        days = (end_date - start_date).days + 1

        st.markdown(
            f"<div style='font-size:0.68rem; color:#6B7280; margin-top:0.4rem; "
            f"padding:0.4rem 0.6rem; background:#F9FAFB; "
            f"border-radius:6px; border:1px solid #E5E7EB;'>"
            f"{start_date.strftime('%b %d, %Y')} — {end_date.strftime('%b %d, %Y')} "
            f"<span style='color:#1D4ED8; font-weight:600;'>({days}d)</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Top N ─────────────────────────────────────────────────────────────
        top_n = st.selectbox(
            "Top N products",
            [5, 10, 20],
            index=0,
            key="sidebar_topn",
        )

        st.markdown("---")

        # ── Gemini API key ────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:0.62rem; font-weight:600; color:#9CA3AF;
        text-transform:uppercase; letter-spacing:0.09em; margin-bottom:0.5rem;">
        Gemini API key
        </div>
        """, unsafe_allow_html=True)

        gemini_key = st.text_input(
            "Gemini API key",
            type="password",
            placeholder="Paste key to unlock AI",
            label_visibility="collapsed",
            key="sidebar_gemini",
        )

        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            st.markdown("""
            <div style="font-size:0.7rem; color:#15803D; margin-top:0.3rem;
                        display:flex; align-items:center; gap:4px;">
                <span style="font-size:0.6rem;">●</span> AI insights enabled
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-size:0.68rem; color:#9CA3AF; margin-top:0.3rem;">
                Get a free key at aistudio.google.com
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        ai_on = st.toggle(
            "Show AI insights on charts",
            value=False,
            key="sidebar_ai_toggle",
        )

    # ── Persist to session state ──────────────────────────────────────────────
    st.session_state["start_date"] = start_date
    st.session_state["end_date"]   = end_date
    st.session_state["days"]       = days
    st.session_state["top_n"]      = int(top_n)
    st.session_state["gemini_key"] = gemini_key
    st.session_state["ai_on"]      = ai_on
