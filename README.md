# AI Chatbot for Customer Service

An intent-classification chatbot built for small-business customer support.
It understands common customer queries (order status, refunds, shipping,
payments, business hours, complaints, human handoff) and responds
appropriately, using classic NLP + machine learning rather than a
pre-trained LLM.

## How it works

1. **Dataset** (`intents.json`) — example phrases grouped by intent (e.g.
   `order_status`, `refund_policy`), each with a set of possible bot
   responses.
2. **Preprocessing** — user input is lowercased, punctuation is stripped,
   tokenized, and lemmatized using **NLTK**.
3. **Vectorization** — cleaned text is converted into TF-IDF feature
   vectors with **scikit-learn**.
4. **Classification** — a **Logistic Regression** model predicts which
   intent the message belongs to.
5. **Confidence threshold** — if the model isn't confident enough about
   its prediction, the bot falls back to a "not sure" response instead of
   guessing wrong.
6. **UI** — a simple chat interface built with **Streamlit**.

## Project structure

```
chatbot/
├── intents.json        # training data: patterns + responses per intent
├── train_model.py       # preprocessing + training pipeline
├── chatbot.py            # inference logic (predict intent, get response)
├── app.py                 # Streamlit chat UI
├── model.pkl               # trained classifier (generated)
├── vectorizer.pkl          # fitted TF-IDF vectorizer (generated)
├── responses.json          # intent -> responses lookup (generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/crypthacker/chatbot.git
cd chatbot
pip install -r requirements.txt
```

## Train the model

Run this any time you change `intents.json`:

```bash
python train_model.py
```

This prints a train/test accuracy report and saves `model.pkl`,
`vectorizer.pkl`, and `responses.json`.

## Run the chatbot

**Web UI (recommended):**
```bash
streamlit run app.py
```

**Command line (quick test):**
```bash
python chatbot.py
```

## Adding more intents

Open `intents.json` and add a new object to the `"intents"` list with a
`tag`, a list of example `patterns`, and a list of possible `responses`.
Then re-run `python train_model.py`. More example patterns per intent =
better accuracy — aim for at least 8-10 varied phrasings per intent.

## Known limitations

- The dataset is small and hand-written, so the model can be confused by
  phrasings very different from the training examples.
- It has no memory of previous turns — each message is classified
  independently.
- It's intent-classification based, not a generative model — it can only
  return responses you've written, not compose new ones.

## Possible improvements

- Expand `intents.json` with more real customer message examples
- Add conversation memory / multi-turn context (e.g. remembering an order
  ID mentioned earlier)
- Swap Logistic Regression for a small neural network for comparison
- Log unmatched (fallback) queries to see what intents are missing
- Deploy publicly (Streamlit Community Cloud is free and simple)
