# nlu_engine/infer_intent.py
import os
import json
from typing import List, Dict
import joblib


class IntentClassifier:
    def __init__(self, model_dir="models/intent_model"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "model.pkl")
        self.labels_path = os.path.join(model_dir, "labels.json")

        self.model = None
        self.labels = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            return
        if not os.path.exists(self.labels_path):
            return

        self.model = joblib.load(self.model_path)

        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

    def predict(self, text: str, top_k: int = 4) -> List[Dict]:
        if self.model is None or self.labels is None:
            raise RuntimeError("Model not trained.")

        # Get probability distribution
        probs = self.model.predict_proba([text])[0]

        class_indices = list(self.model.classes_)  # numeric indices

        mapped = []
        for cls_idx, score in zip(class_indices, probs):
            label = self.labels[int(cls_idx)]
            mapped.append({"label": label, "score": float(score)})

        mapped = sorted(mapped, key=lambda x: x["score"], reverse=True)

        return mapped[:top_k]


