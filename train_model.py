"""
train_model.py
Trains an intent classifier for the customer service chatbot.

Pipeline:
  raw text -> preprocessing (lowercase, tokenize, lemmatize)
           -> TF-IDF vectorization
           -> Logistic Regression classifier (predicts intent tag)

Run this whenever you update intents.json.
"""

import json
import pickle
import string

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data (only needs to run once, cached after)
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, lemmatize."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)


def load_dataset(path="intents.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(preprocess(pattern))
            labels.append(intent["tag"])
    return texts, labels, data


def main():
    texts, labels, data = load_dataset()
    print(f"Loaded {len(texts)} training examples across "
          f"{len(set(labels))} intents.")

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = labels

    # Small dataset -> keep test split small just to sanity check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nTest accuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:\n", classification_report(y_test, preds, zero_division=0))

    # Retrain on FULL data for the actual deployed model (small dataset,
    # so we want every example contributing to the final model)
    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("responses.json", "w", encoding="utf-8") as f:
        json.dump(
            {intent["tag"]: intent["responses"] for intent in data["intents"]},
            f,
            indent=2,
        )

    print("\nSaved model.pkl, vectorizer.pkl, responses.json")


if __name__ == "__main__":
    main()
