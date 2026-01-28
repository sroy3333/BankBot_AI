# services/safety_guard.py

BANKING_KEYWORDS = [
    "account", "transfer", "balance", "atm",
    "card", "password", "money", "upi"
]

def looks_like_banking(text: str) -> bool:
    if text.strip().isdigit():
        return True
    return any(k in text.lower() for k in BANKING_KEYWORDS)


