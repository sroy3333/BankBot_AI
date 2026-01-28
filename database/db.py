# database/db.py

import sqlite3
import bcrypt
from datetime import datetime
from pathlib import Path

DB_NAME = Path(__file__).parent / "bankbot.db"

def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        account_number TEXT PRIMARY KEY,
        user_name TEXT,
        account_type TEXT,
        balance INTEGER,
        password_hash BLOB,
        FOREIGN KEY(user_name) REFERENCES users(name)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        from_account TEXT,
        to_account TEXT,
        amount INTEGER,
        timestamp TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        intent TEXT NOT NULL,
        example TEXT NOT NULL
    )
    """)

    # 🔹 User Queries
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        detected_intent TEXT,
        confidence REAL,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()
