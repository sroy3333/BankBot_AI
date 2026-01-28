# dialogue_manager/dialogue_handler.py
from nlu_engine.infer_intent import IntentClassifier
from nlu_engine.entity_extractor import EntityExtractor
from dialogue_manager.stories import INTENT_STORIES


class DialogueManager:
    def __init__(self):
        self.classifier = IntentClassifier("models/intent_model")
        self.extractor = EntityExtractor()

    def handle(self, text, state):
        text = text.strip()
        state.slots["_original_query"] = text

        # ------------------------------------------------
        # 1️⃣ Escalation
        # ------------------------------------------------
        if state.escalated:
            return {"type": "human"}
        
        # ------------------------------------------------
        # 🔒 CONFIRMATION HANDLING
        # ------------------------------------------------
        if state.awaiting_confirmation:
            user_reply = text.lower().strip()

            if user_reply in ["yes", "y", "confirm", "sure"]:
                state.awaiting_confirmation = False
                state.confirmed = True   # ✅ MARK CONFIRMED
                return {"type": "confirmed"}

            elif user_reply in ["no", "n", "cancel", "stop"]:
                state.reset()
                return {
                    "type": "message",
                    "message": "✅ Card block request cancelled. Your card remains active."
                }

            else:
                return {
                    "type": "message",
                    "message": "⚠️ Please confirm with Yes or No."
                }

        # ------------------------------------------------
        # 2️⃣ Intent detection (ONLY once)
        # ------------------------------------------------
        if state.intent is None:
            pred = self.classifier.predict(text, top_k=1)[0]

            if pred["score"] < 0.45:
                return {"type": "fallback"}

            state.intent = pred["label"]

        story = INTENT_STORIES.get(state.intent)
        if not story:
            return {"type": "fallback"}

        # ------------------------------------------------
        # 3️⃣ Slot capture (ONLY if awaiting)
        # ------------------------------------------------
        if state.awaiting_slot is not None:
            slot = state.awaiting_slot

            if slot == "LOCATION":
                state.slots["LOCATION"] = text
                state.awaiting_slot = None

            elif slot in ["ACCOUNT_NUMBER", "FROM_ACCOUNT", "TO_ACCOUNT"]:
                if text.isdigit() and 6 <= len(text) <= 20:
                    state.slots[slot] = text
                    state.awaiting_slot = None
                else:
                    return {"type": "slot_request", "missing": [slot]}

            elif slot in ["PASSWORD", "FROM_PASSWORD", "TO_PASSWORD"]:
                state.slots[slot] = text
                state.awaiting_slot = None

            elif slot == "AMOUNT":
                try:
                    amount = int(float(text))
                    if amount <= 0:
                        raise ValueError
                    state.slots["AMOUNT"] = amount
                    state.awaiting_slot = None
                except ValueError:
                    return {"type": "slot_request", "missing": ["AMOUNT"]}

        
        # ------------------------------------------------
        # 4️⃣ Entity extraction (Fixed & MODE SAFE)
        # ------------------------------------------------
        entities = self.extractor.extract(text)

        for e in entities:
            ent = e["entity"]
            val = e["value"]

            # ❌ Never override awaited slot
            if state.awaiting_slot == ent:
                continue

            # ===============================
            # TRANSFER MONEY
            # ===============================
            if state.intent == "transfer_money":

                if ent == "ACCOUNT_NUMBER":
                    if "FROM_ACCOUNT" not in state.slots:
                        state.slots["FROM_ACCOUNT"] = val
                    elif "TO_ACCOUNT" not in state.slots:
                        state.slots["TO_ACCOUNT"] = val

                    elif ent == "AMOUNT":
                        state.slots.setdefault("AMOUNT", val)

            # ===============================
            # CHECK BALANCE
            # ===============================
            elif state.intent == "check_balance":

                if ent == "ACCOUNT_NUMBER":
                    state.slots.setdefault("ACCOUNT_NUMBER", val)

                elif ent == "PASSWORD":
                    state.slots.setdefault("PASSWORD", val)

            # ===============================
            # FIND ATM
            # ===============================
            elif state.intent == "find_atm":
                if ent == "LOCATION":
                    state.slots.setdefault("LOCATION", val)
                elif ent == "DISTANCE":
                    state.slots.setdefault("DISTANCE", val)


            # ===============================
            # FALLBACK
            # ===============================
            else:
                state.slots.setdefault(ent, val)

        print("DEBUG slots:", state.slots)

        # ------------------------------------------------
        # DEFAULT LOCATION FOR FIND ATM
        # ------------------------------------------------
        if state.intent == "find_atm" and "LOCATION" not in state.slots:
            state.slots["LOCATION"] = "near me"

        # ------------------------------------------------
        # 5️⃣ Find missing slots
        # ------------------------------------------------
        missing_slots = [
            s for s in story["required_slots"]
            if s not in state.slots
        ]

        # ------------------------------------------------
        # 6️⃣ Ask for next slot
        # ------------------------------------------------
        if missing_slots:
            state.awaiting_slot = missing_slots[0]
            return {
                "type": "slot_request",
                "missing": missing_slots
            }

        # ------------------------------------------------
        # 7️⃣ Confirmation
        # ------------------------------------------------
        if story.get("confirm") and not state.awaiting_confirmation and not state.confirmed:
            state.awaiting_confirmation = True
            return {"type": "confirm"}
        
        # ------------------------------------------------
        # Hard safety check (bank-grade)
        # ------------------------------------------------
        if state.intent == "transfer_money":
            if "AMOUNT" not in state.slots:
                state.awaiting_slot = "AMOUNT"
                return {
                    "type": "slot_request",
                    "missing": ["AMOUNT"]
                }

        
        #------------------------------------------------
        # 8️⃣ Success
        # ------------------------------------------------
        message = story["response"](state.slots)

        state.confirmed = False
        state.reset()

        return {
            "type": "success",
            "message": message
        }


