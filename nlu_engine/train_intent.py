# nlu_engine/train_intent.py
import os
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


def load_intents(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else data["intents"]


def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=5000,
                C=3.0,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto"
            )
        )
    ])



def train(args):
    intents = load_intents(args.data_path)

    texts, labels = [], []
    for intent in intents:
        for ex in intent["examples"]:
            texts.append(ex)
            labels.append(intent["tag"])

    label_list = sorted(list(set(labels)))
    label_to_idx = {lab: i for i, lab in enumerate(label_list)}

    y = [label_to_idx[l] for l in labels]

    model = build_model()
    model.fit(texts, y)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.pkl"))

    with open(os.path.join(args.model_dir, "labels.json"), "w") as f:
        json.dump(label_list, f, indent=2)

    print("Training Complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_dir", default="models/intent_model")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    args = p.parse_args()
    train(args)
