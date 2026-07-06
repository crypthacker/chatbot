"""
chatbot.py
Core chatbot logic: takes user text, predicts intent, returns a response.

Uses the model/vectorizer trained by train_model.py.
Includes a confidence threshold so the bot admits when it's unsure
instead of confidently returning a wrong answer.
"""

import json
import pickle
import random
import string

import nltk
from nltk.stem import WordNetLemmatizer

CONFIDENCE_THRESHOLD = 0.18  # below this, use the fallback intent

lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)


class ChatBot:
    def __init__(self,
                 model_path="model.pkl",
                 vectorizer_path="vectorizer.pkl",
                 responses_path="responses.json"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(responses_path, "r", encoding="utf-8") as f:
            self.responses = json.load(f)

    def predict_intent(self, text: str):
        """Returns (intent_tag, confidence)."""
        cleaned = preprocess(text)
        X = self.vectorizer.transform([cleaned])

        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        best_idx = probs.argmax()
        return classes[best_idx], probs[best_idx]

    def get_response(self, text: str) -> str:
        if not text.strip():
            return "Please type a message so I can help!"

        intent, confidence = self.predict_intent(text)

        if confidence < CONFIDENCE_THRESHOLD or intent not in self.responses:
            intent = "fallback"

        return random.choice(self.responses[intent])


if __name__ == "__main__":
    # quick manual test in the terminal
    bot = ChatBot()
    print("Chatbot ready! Type 'quit' to exit.\n")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        print("Bot:", bot.get_response(msg))
