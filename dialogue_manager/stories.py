# dialogue_manager/stories.py
from database.bank_crud import get_account
from database.bank_crud import transfer_money
from database.security import verify_password
from database.bank_crud import get_account_owner_and_balance
from services.atm_service import get_nearest_atm_distance

def card_block_response(slots):
    card_no = slots.get("CARD_NUMBER")
    acc_no = slots.get("ACCOUNT_NUMBER")

    if card_no:
        return f"🚫 Card ending with {card_no} has been blocked successfully."
    elif acc_no:
        return f"🚫 Card linked to account {acc_no} has been blocked successfully."
    else:
        return "🚫 Your card has been blocked successfully."


def find_atm_response(slots):
    location = slots.get("LOCATION", "your location")
    return f"🏧 Nearest ATM found near **{location}**."


def check_balance_response(slots):
    acc_no = slots["ACCOUNT_NUMBER"]
    password = slots["PASSWORD"]

    data = get_account_owner_and_balance(acc_no)

    if not data:
        return "❌ Account not found"

    owner, balance, pwd_hash = data

    if not verify_password(password, pwd_hash):
        return "❌ Incorrect password"

    # Detect ownership language
    original_query = slots.get("_original_query", "").lower()

    if "my" in original_query:
        return f"💰 Your current balance is ₹{balance}"
    else:
        return f"💰 Current balance of {owner}'s account is ₹{balance}"
    
def find_atm_response(slots):
    location = slots.get("LOCATION")
    distance = slots.get("DISTANCE")

    resolved_distance = get_nearest_atm_distance(
        location=location,
        user_distance=distance
    )

    return f"🏧 Nearest ATM is {resolved_distance} away."



INTENT_STORIES = {

    "check_balance": {
    "required_slots": ["ACCOUNT_NUMBER", "PASSWORD"],
    "confirm": False,
    "response": lambda slots: check_balance_response(slots)
    },

    "transfer_money": {
        "required_slots": ["FROM_ACCOUNT", "FROM_PASSWORD", "TO_ACCOUNT", "TO_PASSWORD", "AMOUNT"],
        "confirm": False,
        "response": lambda slots: transfer_money(
            from_acc=slots["FROM_ACCOUNT"],
            from_pwd=slots["FROM_PASSWORD"],
            to_acc=slots["TO_ACCOUNT"],
            to_pwd=slots["TO_PASSWORD"],
            amount=int(slots["AMOUNT"])
        )
    },

    "find_atm": {
        "required_slots": ["LOCATION"],
        "confirm": False,
        "response": lambda slots: find_atm_response(slots)
    },

    "card_block": {
        "required_slots": [],
        "confirm": True,
        "response": lambda slots: card_block_response(slots)
    },
}


