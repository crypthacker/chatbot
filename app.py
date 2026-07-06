"""
app.py
Streamlit chat interface for the customer service chatbot,
styled as a support ticket log rather than a generic chat window.

Run with:
    streamlit run app.py
"""

import json
import time
from datetime import datetime

import streamlit as st

from chatbot import ChatBot

DEFAULT_CONFIG = {
    "business_name": "Your Business",
    "tagline": "Ask me about orders, refunds, shipping, or hours.",
}


def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    except FileNotFoundError:
        return DEFAULT_CONFIG


config = load_config()

st.set_page_config(page_title=f"{config['business_name']} Support", page_icon="🎫", layout="centered")

# ---------------------------------------------------------------------------
# Custom styling — support-ticket-log aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #1B2B2B;
    --teal: #1F3D3D;
    --sage: #7FA69B;
    --sage-bg: #E9F1EE;
    --amber: #E8A33D;
    --clay: #C1554B;
    --paper: #F2F4F1;
    --paper-line: #D8DFDA;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: var(--paper);
}

/* Hide default streamlit chrome we don't want */
#MainMenu, footer, header {visibility: hidden;}

/* Ticket header */
.ticket-header {
    background: var(--teal);
    color: var(--paper);
    padding: 20px 24px;
    border-radius: 6px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ticket-header .ticket-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    opacity: 0.7;
    letter-spacing: 0.05em;
}
.ticket-header h1 {
    font-size: 20px;
    font-weight: 600;
    margin: 4px 0 0 0;
    color: var(--paper);
}
.status-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    background: rgba(232, 163, 61, 0.15);
    color: var(--amber);
    border: 1px solid rgba(232, 163, 61, 0.4);
    padding: 4px 10px;
    border-radius: 20px;
    white-space: nowrap;
}
.status-pill::before {
    content: "●";
    margin-right: 6px;
    color: var(--amber);
}

.ticket-subline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--teal);
    opacity: 0.65;
    margin-bottom: 18px;
    padding-left: 4px;
}

/* Chat message log styling */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 4px 0;
}

[data-testid="stChatMessageContent"] {
    font-size: 15px;
}

/* User messages */
div[data-testid="stChatMessage"]:has(div[data-testid*="user"]) {
    border-left: 3px solid var(--teal);
}

/* Bot bubble look via markdown wrapper */
.log-line {
    display: flex;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 4px;
    margin-bottom: 2px;
    border-left: 3px solid var(--paper-line);
}
.log-line.bot {
    background: var(--sage-bg);
    border-left-color: var(--sage);
}
.log-line.user {
    background: #FFFFFF;
    border-left-color: var(--teal);
}
.log-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #7A8783;
    min-width: 58px;
    padding-top: 2px;
}
.log-text {
    color: var(--ink);
    line-height: 1.5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--teal);
}
[data-testid="stSidebar"] * {
    color: var(--paper) !important;
}
[data-testid="stSidebar"] button {
    background: var(--amber) !important;
    color: var(--ink) !important;
    border: none !important;
    font-weight: 600;
}

/* Chat input box */
[data-testid="stChatInput"] textarea {
    font-family: 'IBM Plex Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bot():
    return ChatBot()


bot = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! How can I help you today?",
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    ]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="ticket-header">
    <div>
        <div class="ticket-id">TICKET #0001 · OPENED TODAY</div>
        <h1>{config['business_name']} Support</h1>
    </div>
    <div class="status-pill">Bot online</div>
</div>
<div class="ticket-subline">{config['tagline']}</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Render ticket log (custom, not default st.chat_message bubbles)
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    role_class = "bot" if msg["role"] == "assistant" else "user"
    speaker = "SUPPORT" if msg["role"] == "assistant" else "YOU"
    st.markdown(f"""
    <div class="log-line {role_class}">
        <div class="log-meta">{msg['time']}<br>{speaker}</div>
        <div class="log-text">{msg['content']}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Ticket Options")
    if st.button("Close & start new ticket"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! How can I help you today?",
                "time": datetime.now().strftime("%H:%M:%S"),
            }
        ]
        st.rerun()
    st.markdown("---")
    st.markdown(
        "**How this works**\n\n"
        "Messages are matched against known request types "
        "(orders, refunds, shipping, payments, hours) using "
        "TF-IDF + logistic regression. Unmatched messages get "
        "routed to a fallback response."
    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "time": datetime.now().strftime("%H:%M:%S"),
    })

    with st.spinner("Support is typing..."):
        time.sleep(0.4)
        response = bot.get_response(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "time": datetime.now().strftime("%H:%M:%S"),
    })
    st.rerun()

