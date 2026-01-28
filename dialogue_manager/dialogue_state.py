# dialogue_manager/dialogue_state.py
class DialogueState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.awaiting_slot = None
        self.intent = None
        self.slots = {}
        self.awaiting_confirmation = False
        self.confirmed = False
        self.fallback_count = 0
        self.escalated = False


