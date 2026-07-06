"""
app.py
Streamlit chat interface for the customer service chatbot.

Run with:
    streamlit run app.py
"""

import time

import streamlit as st

from chatbot import ChatBot

st.set_page_config(page_title="Customer Support Bot", page_icon="💬")

st.title("💬 Customer Support Assistant")
st.caption("Ask me about orders, refunds, shipping, payments, or business hours.")


@st.cache_resource
def load_bot():
    return ChatBot()


bot = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# clear chat button in sidebar
with st.sidebar:
    st.header("Options")
    if st.button("Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How can I help you today?"}
        ]
        st.rerun()
    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "This is a rule-trained intent classification chatbot "
        "(TF-IDF + Logistic Regression) built for a small business "
        "customer service use case."
    )

# chat input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            time.sleep(0.4)  # small delay so it doesn't feel instant/robotic
            response = bot.get_response(user_input)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
