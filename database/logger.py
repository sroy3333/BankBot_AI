# database/logger.py

from database.db import get_conn
from datetime import datetime

def log_user_query(user_input, intent, confidence):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO user_queries(user_input, detected_intent, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (
        user_input,
        intent,
        confidence,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()