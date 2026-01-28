# services/query_analytics.py

import pandas as pd
from database.db import get_conn

# ------------------------------
# Load Queries
# ------------------------------
def load_queries(days=None):
    conn = get_conn()

    if days:
        df = pd.read_sql(f"""
            SELECT *
            FROM user_queries
            WHERE timestamp >= datetime('now', '-{days} day')
        """, conn)
    else:
        df = pd.read_sql("SELECT * FROM user_queries", conn)

    conn.close()
    return df


# ------------------------------
# KPI Metrics
# ------------------------------
def compute_kpis(df):
    today = pd.Timestamp.now().date()

    return {
        "total_queries": len(df),
        "unique_intents": df["detected_intent"].nunique(),
        "low_confidence": (df["confidence"] < 0.45).sum(),
        "today_queries": (pd.to_datetime(df["timestamp"]).dt.date == today).sum()
    }


# ------------------------------
# Intent Distribution
# ------------------------------
def intent_distribution(df):
    vc = df["detected_intent"].value_counts().reset_index()
    vc.columns = ["intent", "count"]
    return vc

# ------------------------------
# Confidence Distribution
# ------------------------------
def confidence_distribution(df):
    df["confidence_pct"] = df["confidence"] * 100
    return df
