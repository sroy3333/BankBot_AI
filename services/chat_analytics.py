# services/chat_analytics.py

import pandas as pd
from database.db import get_conn

BANKING_INTENTS = [
    "transfer_money",
    "check_balance",
    "card_block",
    "find_atm"
]

def load_analytics_data(limit_days=7):
    conn = get_conn()
    df = pd.read_sql(f"""
        SELECT user_input, detected_intent, timestamp
        FROM user_queries
        WHERE detected_intent IS NOT NULL
          AND timestamp >= datetime('now', '-{limit_days} day')
    """, conn)
    conn.close()
    return df


def intent_distribution(df):
    vc = df["detected_intent"].value_counts().reset_index()
    vc.columns = ["intent", "count"]
    return vc


def top_queries(df, intent, limit=5):
    vc = (
        df[df["detected_intent"] == intent]["user_input"]
        .value_counts()
        .head(limit)
        .reset_index(name="count")
    )
    vc.columns = ["query", "count"]
    return vc


