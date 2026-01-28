# router/query_router.py

BANKING_INTENTS = {
    "check_balance",
    "transfer_money",
    "card_block",
    "find_atm"
}

class QueryRouter:
    def __init__(self, classifier, threshold=0.45):
        self.classifier = classifier
        self.threshold = threshold

    def route(self, text):
        preds = self.classifier.predict(text, top_k=1)
        top = preds[0]

        if (
            top["score"] >= self.threshold
            and top["label"] in BANKING_INTENTS
        ):
            return "BANKING", top

        return "GENERAL", top
