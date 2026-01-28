# nlu_engine/entity_extractor.py
import re

class EntityExtractor:
    def __init__(self):

        self.txn_patterns = [
            r'\b(?:txn|transaction|utr|ref(?:\.?|erence)?(?:\s*no)?|reference number|transaction id|txn id)[\s:]*([A-Za-z0-9\-_]+)\b'
        ]

        self.account_context_patterns = [
            r'\b(?:account number|acct no|account no|a/c|to account)\s*([0-9]{4,20})\b',
            r'\baccount\s+ending\s*([0-9]{3,10})\b'
        ]

        self.amount_patterns = [
            r'(₹|Rs\.?|INR|\$|usd|rupees?|dollars?)\s*([0-9,]+(?:\.[0-9]+)?)',
            r'\b([0-9,]+(?:\.[0-9]+)?)\s*(?:rupees?|dollars?|INR|Rs\.?)\b',
            r'\b([0-9]+(?:\.[0-9]+)?)\s*[kK]\b',
            r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:thousand)\b',
            r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:lakh|lac)\b',
            r'\b([1-9][0-9]{0,4})\b'
        ]

        self.account_type_patterns = [
            r'\b(savings account|saving account|savings|saving)\b',
            r'\b(current account|current)\b',
            r'\b(checking account|checking)\b',
            r'\b(salary account|salary)\b'
        ]

        self.card_patterns = [
            r'\b(?:card|card number)\s+(?:ending|ending with)\s*([0-9]{3,6})\b'
        ]

        self.password_patterns = [
            r'\bpassword\s*[:=]?\s*([A-Za-z0-9@#$_!]{4,20})\b'
        ]

        self.sender_patterns = [
            r'\bfrom account\s*([0-9]{6,20})\b'
        ]

        self.receiver_patterns = [
            r'\bto account\s*([0-9]{6,20})\b'
        ]

        self.location_patterns = [
            r'\b(?:in|near|around|at|on)\s+([A-Za-z][A-Za-z\s]{2,30})\b'
            
        ]

        self.distance_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(km|kilometers|kilometres|meters|m)\b'
        ]



    
    # ------------------------
    # Helper function
    # ------------------------
    def _looks_like_account(self, value: str) -> bool:
        return value.isdigit() and len(value) >= 6

    # -------------------------
    # Utilities
    # -------------------------
    def _overlaps(self, reserved, start, end):
        return any(not (end <= a or start >= b) for a, b in reserved)

    def _reserve(self, reserved, start, end):
        reserved.append((start, end))

    def _normalize_amount(self, raw):
        raw = raw.replace(",", "").lower().strip()

        if raw.endswith("k"):
            return float(raw[:-1]) * 1000
        if "lakh" in raw or "lac" in raw:
            return float(re.sub(r"[^\d.]", "", raw)) * 100000
        if "thousand" in raw:
            return float(re.sub(r"[^\d.]", "", raw)) * 1000

        return float(raw)

    # -------------------------
    # Extraction
    # -------------------------
    def extract(self, text):
        raw_entities = []
        reserved = []

        # CARD
        for p in self.card_patterns:
            for m in re.finditer(p, text, re.I):
                s, e = m.span(1)
                if not self._overlaps(reserved, s, e):
                    raw_entities.append(("CARD_NUMBER", m.group(1), s, e))
                    self._reserve(reserved, s, e)

        # TXN
        for p in self.txn_patterns:
            for m in re.finditer(p, text, re.I):
                s, e = m.span(1)
                if not self._overlaps(reserved, s, e):
                    raw_entities.append(("TXN_ID", m.group(1), s, e))
                    self._reserve(reserved, s, e)

        # ACCOUNT
        for p in self.account_context_patterns:
            for m in re.finditer(p, text, re.I):
                s, e = m.span(1)
                if not self._overlaps(reserved, s, e):
                    raw_entities.append(("ACCOUNT_NUMBER", m.group(1), s, e))
                    self._reserve(reserved, s, e)

        # ACCOUNT TYPE
        for p in self.account_type_patterns:
            for m in re.finditer(p, text, re.I):
                s, e = m.span(1)
                if not self._overlaps(reserved, s, e):
                    raw_entities.append(("ACCOUNT_TYPE", m.group(1).lower(), s, e))
                    self._reserve(reserved, s, e)

        # AMOUNT (SAFE)
        for p in self.amount_patterns:
            for m in re.finditer(p, text, re.I):
                raw_value = m.group(m.lastindex)
                s, e = m.span(m.lastindex)

                # ❌ Prevent account numbers becoming amount
                if self._looks_like_account(raw_value):
                    continue

                if not self._overlaps(reserved, s, e):
                    raw_entities.append((
                        "AMOUNT",
                        self._normalize_amount(raw_value),
                        s,
                        e
                    ))
                    self._reserve(reserved, s, e)


        # PASSWORD
        for p in self.password_patterns:
            for m in re.finditer(p, text, re.I):
                s, e = m.span(1)
                if not self._overlaps(reserved, s, e):
                    raw_entities.append(("PASSWORD", m.group(1), s, e))
                    self._reserve(reserved, s, e)

        # LOCATION
        for p in self.location_patterns:
            for m in re.finditer(p, text, re.I):
                s, e = m.span(1)
                if not self._overlaps(reserved, s, e):
                    raw_entities.append(("LOCATION", m.group(1).strip(), s, e))
                    self._reserve(reserved, s, e)

        # DISTANCE
        for p in self.distance_patterns:
            for m in re.finditer(p, text, re.I):
                raw_entities.append(("DISTANCE", f"{m.group(1)} {m.group(2)}", m.start(), m.end()))



        # -------------------------
        # ✅ FINAL DEDUPLICATION
        # -------------------------
        seen = set()
        entities = []

        for ent, val, s, e in raw_entities:
            key = (ent, str(val))
            if key not in seen:
                seen.add(key)
                entities.append({
                    "entity": ent,
                    "value": val,
                    "start": s,
                    "end": e
                })

        return entities

def extract(text):
    return EntityExtractor().extract(text)

