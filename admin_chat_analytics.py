# admin_chat_analytics.py
from database.bank_crud import load_accounts_admin
from database.bank_crud import load_transactions_admin
import streamlit as st
import plotly.express as px
from services.chat_analytics import (
    load_analytics_data,
    intent_distribution,
    top_queries
)

INTENT_META = {
    "transfer_money": ("💸 Transfer Money", "#ec4899"),
    "check_balance": ("💰 Check Balance", "#60a5fa"),
    "card_block": ("🚫 Card Block", "#f97316"),
    "find_atm": ("📍 Find ATM", "#22c55e"),
}

# ------------------------------------------------
def donut_pie(df, intent, title, color):
    intent_count = (df["detected_intent"] == intent).sum()
    other_count = len(df) - intent_count

    pie_df = {
        "Category": [intent.replace("_", " ").title(), "Other Intents"],
        "Count": [intent_count, other_count]
    }

    fig = px.pie(
        pie_df,
        names="Category",
        values="Count",
        hole=0.45,
        color_discrete_sequence=[color, "#1f2937"],
        template="plotly_dark"
    )

    fig.update_layout(title=title)
    return fig


def top_query_bar(df, intent):
    tq = top_queries(df, intent)
    if tq.empty:
        return None

    fig = px.bar(
        tq,
        x="count",
        y="query",
        orientation="h",
        template="plotly_dark"
    )
    fig.update_layout(title="Top Queries")
    return fig


# ------------------------------------------------
def overall_analytics(df):
    dist = intent_distribution(df)

    pie = px.pie(
        dist,
        names="intent",
        values="count",
        hole=0.45,
        template="plotly_dark"
    )
    pie.update_layout(title="All Intents Distribution")

    dist["usage_pct"] = (dist["count"] / dist["count"].sum()) * 100

    bar = px.bar(
        dist,
        x="intent",
        y="usage_pct",
        color="usage_pct",
        template="plotly_dark"
    )
    bar.update_layout(
        title="% of Questions by Intent",
        yaxis_title="Usage %"
    )

    st.plotly_chart(pie, use_container_width=True)
    st.plotly_chart(bar, use_container_width=True)


# ------------------------------------------------
def chat_analytics_dashboard():
    st.subheader("📊 Chat Analytics Dashboard")

    df = load_analytics_data()

    if df.empty:
        st.info("No analytics data yet")
        return

    tabs = st.tabs(
        ["📊 Overall Analytics"] +
        [v[0] for v in INTENT_META.values()]
    )

    # Overall
    with tabs[0]:
        overall_analytics(df)

    # Per Intent
    for i, (intent, (title, color)) in enumerate(INTENT_META.items(), start=1):
        with tabs[i]:
            st.plotly_chart(
            donut_pie(df, intent, title, color),
            use_container_width=True
        )

        bar = top_query_bar(df, intent)
        if bar:
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.info("No queries for this intent yet")

        # ==================================================
        # 🆕 EXTRA ANALYTICS FOR TRANSFER MONEY
        # ==================================================
        if intent == "transfer_money":
            st.divider()
            st.subheader("🏦 Banking Data (Admin View)")

            sub_tabs = st.tabs(["🏦 Accounts", "💸 Transactions"])

            # -----------------------------
            # Accounts
            # -----------------------------
            with sub_tabs[0]:
                acc_df = load_accounts_admin()

                if acc_df.empty:
                    st.info("No accounts found")
                else:
                    st.dataframe(acc_df, use_container_width=True)

                    st.caption(
                        f"Total Accounts: {len(acc_df)} | "
                        f"Total Balance: ₹{acc_df['Balance'].sum():,.2f}"
                    )

            # -----------------------------
            # Transactions
            # -----------------------------
            with sub_tabs[1]:
                tx_df = load_transactions_admin()

                if tx_df.empty:
                    st.info("No transactions recorded")
                else:
                    st.dataframe(tx_df, use_container_width=True)
                    st.caption(f"Showing latest {len(tx_df)} transactions")
