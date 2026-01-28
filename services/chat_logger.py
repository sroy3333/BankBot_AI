# services/chat_logger.py

import csv
import os
from datetime import datetime

LOG_FILE = "logs/chat_logs.csv"

def log_chat(session_id, role, message):
    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "session_id",
                "role",
                "message"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            session_id,
            role,
            message
        ])