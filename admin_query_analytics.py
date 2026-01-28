# admin_query_analytics.py

import streamlit as st
import plotly.express as px
from services.query_analytics import (
    load_queries,
    compute_kpis,
    intent_distribution,
    confidence_distribution
)

def query_analytics_dashboard():
    st.subheader("🔍 Query Analytics")

    df = load_queries()

    if df.empty:
        st.info("No query data available yet.")
        return

    # --------------------------
    # KPIs (Top Cards)
    # --------------------------
    kpis = compute_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", kpis["total_queries"])
    c2.metric("Intents", kpis["unique_intents"])
    c3.metric("Low Confidence", kpis["low_confidence"])
    c4.metric("Today", kpis["today_queries"])

    st.divider()

    # --------------------------
    # Intent Distribution (Donut)
    # --------------------------
    intent_df = intent_distribution(df)

    intent_fig = px.pie(
        intent_df,
        names="intent",
        values="count",
        hole=0.45,
        title="Intent Distribution",
        template="plotly_dark"
    )

    # --------------------------
    # Confidence Distribution
    # --------------------------
    conf_df = confidence_distribution(df)

    conf_fig = px.histogram(
        conf_df,
        x="confidence_pct",
        nbins=10,
        title="Confidence Distribution",
        template="plotly_dark"
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(intent_fig, use_container_width=True)
    col2.plotly_chart(conf_fig, use_container_width=True)
