"""
Pulse — Customers
RFM segmentation and cohort retention analysis.
"""

import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analytics.segmentation import compute_rfm, segment_summary, rfm_summary_for_ai
from analytics.cohort import build_cohort_matrix, cohort_summary_for_ai

# ── Helpers ───────────────────────────────────────────────────────────────────

def _kpi_card(label, value, hint=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {"<div class='kpi-hint'>" + hint + "</div>" if hint else ""}
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
                        "You are a customer analytics expert. Give 2-3 concise, "
                        "specific, actionable observations about this data in plain "
                        "English. No bullet points. Do not start with 'I'.\n\n"
                        f"Context: {context}"
                    )
                    url = (
                        "https://generativelanguage.googleapis.com/v1/models/"
                        f"gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}"
                    )
                    r = requests.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.3,
                                "maxOutputTokens": 300,
                            },
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

@st.cache_data(show_spinner="Computing RFM segments...")
def _load_rfm():
    rfm     = compute_rfm()
    summary = segment_summary(rfm)
    ai_ctx  = rfm_summary_for_ai(rfm, summary)
    return rfm, summary, ai_ctx


@st.cache_data(show_spinner="Building cohort matrix...")
def _load_cohort():
    retention, sizes = build_cohort_matrix()
    ai_ctx           = cohort_summary_for_ai(retention, sizes)
    return retention, sizes, ai_ctx


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-title">Customers</div>
<div class="page-sub">RFM segmentation and cohort retention analysis</div>
""", unsafe_allow_html=True)

has_key = bool(st.session_state.get("gemini_key", ""))
ai_on   = st.session_state.get("ai_on", False)

rfm, rfm_sum, rfm_ai = _load_rfm()
retention, coh_sizes, coh_ai = _load_cohort()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_rfm, tab_cohort = st.tabs(["RFM segmentation", "Cohort retention"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — RFM
# ════════════════════════════════════════════════════════════════════════════════

with tab_rfm:

    # ── AI summary ────────────────────────────────────────────────────────────
    if has_key:
        rfm_sum_key = "rfm_summary"
        if rfm_sum_key not in st.session_state:
            with st.spinner("Generating AI summary..."):
                try:
                    import requests, os
                    prompt = (
                        "You are a customer analytics expert. Write 2-3 sentences "
                        "summarising the health of this customer base based on the "
                        "RFM segment data. Be specific. No bullet points. "
                        "Do not start with 'I'.\n\n"
                        f"Data: {rfm_ai}"
                    )
                    url = (
                        "https://generativelanguage.googleapis.com/v1/models/"
                        f"gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}"
                    )
                    r = requests.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.3,
                                "maxOutputTokens": 200,
                            },
                        },
                        timeout=30,
                    )
                    st.session_state[rfm_sum_key] = (
                        r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if r.status_code == 200 else None
                    )
                except Exception:
                    st.session_state[rfm_sum_key] = None

        narrative = st.session_state.get(rfm_sum_key)
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
            Add a Gemini API key in the sidebar to unlock the AI customer summary.
        </div>
        """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_cust = rfm_ai.get("total_customers", 0)
    champ_pct  = rfm_ai.get("champions_pct", 0)
    at_risk    = rfm_ai.get("at_risk_count", 0)
    lost_pct   = rfm_ai.get("lost_pct", 0)

    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        _kpi_card("Total customers", f"{total_cust:,}", hint="All-time")
    with k2:
        _kpi_card("Champions", f"{champ_pct:.1f}%", hint="High R + F scores")
    with k3:
        _kpi_card("At risk", f"{at_risk:,}", hint="Re-engagement needed")
    with k4:
        _kpi_card("Lost", f"{lost_pct:.1f}%", hint="Low recency + frequency")

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Segment breakdown chart + table ───────────────────────────────────────
    seg_l, seg_r = st.columns([1, 1.4], gap="small")

    SEGMENT_COLORS = {
        "Champions": "#1D4ED8",
        "Loyal":     "#3B82F6",
        "At Risk":   "#F59E0B",
        "New":       "#10B981",
        "Lost":      "#9CA3AF",
    }

    with seg_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-title">Customer distribution by segment</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="chart-sub">Share of total customer base</div>',
            unsafe_allow_html=True,
        )

        seg_order = ["Champions", "Loyal", "New", "At Risk", "Lost"]
        rfm_plot  = rfm_sum[rfm_sum["segment"].isin(seg_order)].copy()
        rfm_plot["segment"] = pd.Categorical(
            rfm_plot["segment"], categories=seg_order, ordered=True
        )
        rfm_plot = rfm_plot.sort_values("segment")

        fig = go.Figure(go.Pie(
            labels=rfm_plot["segment"],
            values=rfm_plot["customer_count"],
            hole=0.55,
            marker_colors=[
                SEGMENT_COLORS.get(s, "#E5E7EB") for s in rfm_plot["segment"]
            ],
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Customers: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=8, b=0),
            showlegend=False,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        _ai_chart_insight(
            key="customers_rfm_pie",
            context=(
                f"Donut chart of customer distribution by RFM segment. "
                f"Total customers: {total_cust:,}. "
                f"Champions: {champ_pct:.1f}%, At Risk: {at_risk:,} customers, "
                f"Lost: {lost_pct:.1f}%."
            ),
            ai_on=ai_on,
            has_key=has_key,
        )

    with seg_r:
        st.markdown(
            '<div class="section-header">Segment metrics</div>',
            unsafe_allow_html=True,
        )

        display = rfm_sum.copy()
        display["avg_recency"]   = display["avg_recency"].apply(lambda v: f"{v}d")
        display["avg_frequency"] = display["avg_frequency"].apply(lambda v: f"{v:.1f}")
        display["avg_monetary"]  = display["avg_monetary"].apply(lambda v: f"${v:,.2f}")
        display["total_revenue"] = display["total_revenue"].apply(lambda v: f"${v:,.0f}")
        display["pct_customers"] = display["pct_customers"].apply(lambda v: f"{v:.1f}%")
        display = display.rename(columns={
            "segment":       "Segment",
            "customer_count":"Customers",
            "pct_customers": "Share",
            "avg_recency":   "Avg recency",
            "avg_frequency": "Avg orders",
            "avg_monetary":  "Avg spend",
            "total_revenue": "Total revenue",
        })

        st.dataframe(
            display[["Segment", "Customers", "Share",
                      "Avg recency", "Avg orders",
                      "Avg spend", "Total revenue"]],
            use_container_width=True,
            hide_index=True,
        )

        # ── Avg spend by segment bar ──────────────────────────────────────────
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-title">Average spend per customer by segment</div>',
            unsafe_allow_html=True,
        )

        rfm_bar = rfm_sum[rfm_sum["segment"].isin(seg_order)].copy()
        rfm_bar["segment"] = pd.Categorical(
            rfm_bar["segment"], categories=seg_order, ordered=True
        )
        rfm_bar = rfm_bar.sort_values("avg_monetary", ascending=True)

        fig2 = go.Figure(go.Bar(
            x=rfm_bar["avg_monetary"],
            y=rfm_bar["segment"],
            orientation="h",
            marker_color=[
                SEGMENT_COLORS.get(s, "#E5E7EB") for s in rfm_bar["segment"]
            ],
            hovertemplate="<b>%{y}</b><br>Avg spend: $%{x:,.2f}<extra></extra>",
        ))
        fig2.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=4, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            xaxis=dict(
                showgrid=True, gridcolor="#F3F4F6",
                tickfont=dict(size=11), tickprefix="$",
            ),
            yaxis=dict(showgrid=False, tickfont=dict(size=11)),
            bargap=0.3,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        _ai_chart_insight(
            key="customers_rfm_spend",
            context=(
                "Horizontal bar chart of average spend per customer by RFM segment. "
                f"Top segment by spend: {rfm_bar.iloc[-1]['segment']} "
                f"at ${rfm_bar.iloc[-1]['avg_monetary']:,.2f}."
            ),
            ai_on=ai_on,
            has_key=has_key,
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Cohort retention
# ════════════════════════════════════════════════════════════════════════════════

with tab_cohort:

    # ── AI summary ────────────────────────────────────────────────────────────
    if has_key:
        coh_sum_key = "cohort_summary"
        if coh_sum_key not in st.session_state:
            with st.spinner("Generating AI summary..."):
                try:
                    import requests, os
                    prompt = (
                        "You are a customer retention expert. Write 2-3 sentences "
                        "summarising the cohort retention patterns. Be specific about "
                        "month-1 and month-3 retention rates. No bullet points. "
                        "Do not start with 'I'.\n\n"
                        f"Data: {coh_ai}"
                    )
                    url = (
                        "https://generativelanguage.googleapis.com/v1/models/"
                        f"gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}"
                    )
                    r = requests.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.3,
                                "maxOutputTokens": 200,
                            },
                        },
                        timeout=30,
                    )
                    st.session_state[coh_sum_key] = (
                        r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if r.status_code == 200 else None
                    )
                except Exception:
                    st.session_state[coh_sum_key] = None

        narrative = st.session_state.get(coh_sum_key)
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
            Add a Gemini API key in the sidebar to unlock AI cohort insights.
        </div>
        """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    m1  = coh_ai.get("avg_month1_retention")
    m3  = coh_ai.get("avg_month3_retention")
    m6  = coh_ai.get("avg_month6_retention")
    avg_size = coh_ai.get("avg_cohort_size", 0)

    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        _kpi_card(
            "Avg month-1 retention",
            f"{m1:.1f}%" if m1 else "N/A",
            hint="% returning after month 1",
        )
    with k2:
        _kpi_card(
            "Avg month-3 retention",
            f"{m3:.1f}%" if m3 else "N/A",
            hint="% returning after month 3",
        )
    with k3:
        _kpi_card(
            "Avg month-6 retention",
            f"{m6:.1f}%" if m6 else "N/A",
            hint="% returning after month 6",
        )
    with k4:
        _kpi_card(
            "Avg cohort size",
            f"{avg_size:,.0f}",
            hint="Customers per signup month",
        )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-title">Cohort retention heatmap</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chart-sub">'
        'Rows = signup month · Columns = months since signup · '
        'Values = % of cohort still active'
        '</div>',
        unsafe_allow_html=True,
    )

    if retention.empty:
        st.info("No cohort data available.")
    else:
        ret_pct  = (retention * 100).round(1)
        z_values = ret_pct.values.tolist()
        x_labels = [f"M+{int(c)}" for c in retention.columns]
        y_labels = list(retention.index)

        text_vals = [
            [
                f"{v:.0f}%" if not np.isnan(v) else ""
                for v in row
            ]
            for row in ret_pct.values
        ]

        fig3 = go.Figure(go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale=[
                [0.0,  "#EFF6FF"],
                [0.25, "#BFDBFE"],
                [0.5,  "#60A5FA"],
                [0.75, "#2563EB"],
                [1.0,  "#1E3A8A"],
            ],
            zmin=0,
            zmax=100,
            hovertemplate=(
                "<b>Cohort: %{y}</b><br>"
                "%{x}: %{z:.1f}%<extra></extra>"
            ),
            showscale=True,
            colorbar=dict(
                ticksuffix="%",
                tickfont=dict(size=10),
                thickness=12,
                len=0.8,
            ),
        ))
        fig3.update_layout(
            height=max(300, len(y_labels) * 26 + 60),
            margin=dict(l=0, r=40, t=8, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(tickfont=dict(size=10), side="top"),
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)

    _ai_chart_insight(
        key="customers_cohort_heatmap",
        context=(
            f"Cohort retention heatmap. {coh_ai.get('total_cohorts', 0)} cohorts. "
            f"Avg month-1 retention: {m1:.1f}% " if m1 else ""
            f"Avg month-3 retention: {m3:.1f}%. " if m3 else ""
            f"Best month-1 cohort: {coh_ai.get('best_month1_cohort', 'N/A')}. "
            f"Worst month-1 cohort: {coh_ai.get('worst_month1_cohort', 'N/A')}."
        ),
        ai_on=ai_on,
        has_key=has_key,
    )

    # ── Month-1 retention trend ───────────────────────────────────────────────
    if 1 in retention.columns:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-title">Month-1 retention by cohort</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="chart-sub">Trend across signup cohorts</div>',
            unsafe_allow_html=True,
        )

        m1_series = (retention[1].dropna() * 100).round(1)
        avg_line  = [m1_series.mean()] * len(m1_series)

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=m1_series.index,
            y=m1_series.values,
            name="Month-1 retention",
            marker_color="#BFDBFE",
            hovertemplate="<b>%{x}</b><br>M1 retention: %{y:.1f}%<extra></extra>",
        ))
        fig4.add_trace(go.Scatter(
            x=m1_series.index,
            y=avg_line,
            name=f"Average ({m1_series.mean():.1f}%)",
            line=dict(color="#1D4ED8", width=2, dash="dash"),
            hoverinfo="skip",
        ))
        fig4.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=8, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11),
            ),
            xaxis=dict(
                showgrid=False, tickfont=dict(size=10),
                tickangle=-45,
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#F3F4F6",
                tickfont=dict(size=11), ticksuffix="%",
                range=[0, 100],
            ),
            bargap=0.3,
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        _ai_chart_insight(
            key="customers_m1_trend",
            context=(
                f"Bar chart of month-1 retention rate across all cohorts. "
                f"Average M1 retention: {m1_series.mean():.1f}%. "
                f"Best cohort: {m1_series.idxmax()} at {m1_series.max():.1f}%. "
                f"Worst cohort: {m1_series.idxmin()} at {m1_series.min():.1f}%."
            ),
            ai_on=ai_on,
            has_key=has_key,
        )
