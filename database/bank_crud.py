# database/bank_crud.py

from database.db import get_conn
from database.security import hash_password, verify_password
from datetime import datetime
import pandas as pd

def create_account(name, acc_no, acc_type, balance, password):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("INSERT OR IGNORE INTO users(name) VALUES (?)", (name,))
    pwd_hash = hash_password(password)

    cur.execute("""
    INSERT INTO accounts(account_number, user_name, account_type, balance, password_hash)
    VALUES (?, ?, ?, ?, ?)
    """, (acc_no, name, acc_type, balance, pwd_hash))

    conn.commit()
    conn.close()

def get_account(acc_no):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT account_number, user_name, account_type, balance, password_hash
    FROM accounts WHERE account_number=?
    """, (acc_no,))
    row = cur.fetchone()
    conn.close()
    return row


def get_account_owner_and_balance(acc_no):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT user_name, balance, password_hash
    FROM accounts
    WHERE account_number=?
    """, (acc_no,))

    row = cur.fetchone()
    conn.close()
    return row  


def list_accounts():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT account_number, user_name FROM accounts")
    rows = cur.fetchall()
    conn.close()
    return rows

def transfer_money(
    from_acc,
    from_pwd,
    to_acc,
    to_pwd,
    amount
):
    conn = get_conn()
    cur = conn.cursor()

    # --------------------------
    # Validate sender
    # --------------------------
    cur.execute(
        "SELECT balance, password_hash FROM accounts WHERE account_number=?",
        (from_acc,)
    )
    sender = cur.fetchone()
    if not sender:
        return "❌ Sender account not found"

    sender_balance, sender_hash = sender
    if not verify_password(from_pwd, sender_hash):
        return "❌ Incorrect sender password"

    if sender_balance < amount:
        return "❌ Insufficient balance"

    # --------------------------
    # Validate receiver
    # --------------------------
    cur.execute(
        "SELECT password_hash FROM accounts WHERE account_number=?",
        (to_acc,)
    )
    receiver = cur.fetchone()
    if not receiver:
        return "❌ Receiver account not found"

    receiver_hash = receiver[0]
    if not verify_password(to_pwd, receiver_hash):
        return "❌ Incorrect receiver password"


    # Transaction (ACID)
    cur.execute("UPDATE accounts SET balance = balance - ? WHERE account_number=?", (amount, from_acc))
    cur.execute("UPDATE accounts SET balance = balance + ? WHERE account_number=?", (amount, to_acc))

    cur.execute("""
    INSERT INTO transactions(from_account, to_account, amount, timestamp)
    VALUES (?, ?, ?, ?)
    """, (from_acc, to_acc, amount, datetime.now().isoformat()))

    conn.commit()
    conn.close()
    return "✅ Transfer Successful"


def delete_account(acc_no):
    conn = get_conn()
    cur = conn.cursor()

    # Check if account exists
    cur.execute(
        "SELECT 1 FROM accounts WHERE account_number = ?",
        (acc_no,)
    )
    exists = cur.fetchone()

    if not exists:
        conn.close()
        return False # ❌ Account not found

    # Delete account
    cur.execute(
        "DELETE FROM accounts WHERE account_number = ?",
        (acc_no,)
    )

    conn.commit()
    conn.close()
    return True # ✅ Deleted successfully

def load_accounts_admin():
    conn = get_conn()
    df = pd.read_sql("""
        SELECT
            account_number AS "Account No",
            user_name AS "Customer",
            account_type AS "Type",
            balance AS "Balance"
        FROM accounts
        ORDER BY user_name
    """, conn)
    conn.close()
    return df


def load_transactions_admin(limit=100):
    conn = get_conn()
    df = pd.read_sql(f"""
        SELECT
            from_account AS "From",
            to_account AS "To",
            amount AS "Amount",
            timestamp AS "Time"
        FROM transactions
        ORDER BY timestamp DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df

