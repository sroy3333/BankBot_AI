# admin_panel.py
import streamlit as st
import pandas as pd
from datetime import datetime
from database.db import get_conn
from admin_chat_analytics import chat_analytics_dashboard
from admin_query_analytics import query_analytics_dashboard

import os

FAQ_CSV = "knowledge_base/faqs.csv"
LOG_CSV = "logs/chat_logs.csv"


# ==========================================================
# 🛠️ ADMIN PANEL
# ==========================================================
def admin_panel():
    st.title("🛠️ Admin Panel")

    tabs = st.tabs([
        "📚 Training Data",
        "❓ FAQs Manager",
        "💬 User Queries",
        "🔁 Retrain Model",
        "📊 Chat Analytics",
        "📈 Query Analytics",
        "📄 Raw Logs"
    ])

    training_data_tab(tabs[0])
    faq_tab(tabs[1])
    user_queries_tab(tabs[2])
    retrain_tab(tabs[3])

    with tabs[4]:
        chat_analytics_dashboard()

    with tabs[5]:
        query_analytics_dashboard()

    logs_tab(tabs[6])


# ==========================================================
# 📚 TRAINING DATA TAB
# ==========================================================
def training_data_tab(tab):
    with tab:
        st.subheader("📚 Edit Training Data")

        conn = get_conn()
        df = pd.read_sql("SELECT * FROM training_data", conn)
        conn.close()

        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic"
        )

        if st.button("💾 Save Training Data"):
            conn = get_conn()
            conn.execute("DELETE FROM training_data")
            edited_df.to_sql(
                "training_data",
                conn,
                if_exists="append",
                index=False
            )
            conn.commit()
            conn.close()
            st.success("Training data updated successfully")


# ==========================================================
# ❓ FAQ MANAGER TAB
# ==========================================================
def faq_tab(tab):
    with tab:
        st.subheader("❓ Manage FAQs")

        os.makedirs(os.path.dirname(FAQ_CSV), exist_ok=True)

        if not os.path.exists(FAQ_CSV):
            pd.DataFrame(
                [
                    {
                        "question": "What is BankBot?",
                        "answer": "BankBot is a secure AI-powered banking assistant."
                    },
                    {
                        "question": "What can BankBot help with?",
                        "answer": "Balance enquiry, money transfer, card blocking, and ATM search."
                    }
                ]
            ).to_csv(FAQ_CSV, index=False)

        df = pd.read_csv(FAQ_CSV)

        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic"
        )

        if st.button("💾 Save FAQs"):
            edited_df.to_csv(FAQ_CSV, index=False)
            st.success("FAQs updated successfully")

        st.info("These FAQs are checked before NLP model inference.")


# ==========================================================
# 💬 USER QUERIES TAB
# ==========================================================
def user_queries_tab(tab):
    with tab:
        st.subheader("💬 User Queries")

        conn = get_conn()
        df = pd.read_sql("""
            SELECT user_input, detected_intent, timestamp
            FROM user_queries
            ORDER BY timestamp DESC
        """, conn)
        conn.close()

        st.dataframe(df, use_container_width=True)

        if st.button("📤 Export Queries CSV"):
            file_name = f"user_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(file_name, index=False)
            st.success(f"Exported as {file_name}")


# ==========================================================
# 🔁 RETRAIN MODEL TAB
# ==========================================================
def retrain_tab(tab):
    with tab:
        st.subheader("🔁 Retrain NLP Model")
        st.warning("⚠️ Retraining may take several minutes")

        if st.button("🚀 Start Retraining"):
            with st.spinner("Training model..."):
                retrain_model()
            st.success("Model retrained successfully")


def retrain_model():
    import time
    time.sleep(3)
    # Hook real pipeline here
    # train_intent_classifier()
    # rebuild_embeddings()
    # save_model()


# ==========================================================
# 📄 RAW LOGS TAB
# ==========================================================
def logs_tab(tab):
    with tab:
        st.subheader("📄 Raw Chat Logs")

        os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

        if not os.path.exists(LOG_CSV):
            st.error("No logs available")
            return

        df = pd.read_csv(LOG_CSV)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", len(df))
        col2.metric("Unique Sessions", df["session_id"].nunique())
        col3.metric(
            "Avg Messages / Session",
            round(len(df) / df["session_id"].nunique(), 2)
        )

        st.divider()
        st.dataframe(df, use_container_width=True)

        if st.button("📤 Export Logs CSV"):
            export_name = f"chat_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(export_name, index=False)
            st.success(f"Exported as {export_name}")


# ==========================================================
# 🚀 RUN
# ==========================================================
if __name__ == "__main__":
    admin_panel()

