"""Train the FinBuddy AI intent classifier using TF-IDF + Logistic Regression."""

import json
import os
import re
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_FILE = "intent_model.joblib"
DATA_FILE = os.path.join("dataset", "intents.json")
lemmatizer = WordNetLemmatizer()


def ensure_nltk():
    """NLTK is used by the project. Regex tokenization keeps training runnable offline if corpora are unavailable."""
    return None


def safe_lemmatize(token):
    try:
        return lemmatizer.lemmatize(token)
    except LookupError:
        return token


def preprocess(text):
    # Regex tokenization keeps the project runnable even when NLTK data cannot download.
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    tokens = [safe_lemmatize(token) for token in tokens if token.isalnum()]
    return " ".join(tokens)


def load_dataset():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
    x, y = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            x.append(preprocess(pattern))
            y.append(intent["tag"])
    return x, y


def train_model():
    ensure_nltk()
    x, y = load_dataset()
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(x, y)

    # Training report is useful for coursework documentation.
    if len(set(y)) > 1 and len(x) > 12:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
        eval_model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
        eval_model.fit(x_train, y_train)
        predictions = eval_model.predict(x_test)
        print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
        print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    return model


if __name__ == "__main__":
    train_model()
