import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_NAME = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are a helpful, concise assistant.
You explain things clearly and avoid unnecessary verbosity.
If you do not know something, you say so honestly.
"""

# --------------------------------------------------
# INITIALIZE CHATBOT (ONCE PER SESSION)
# --------------------------------------------------

if "chatbot" not in st.session_state:
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.7
    )

    memory = ConversationBufferWindowMemory(
        k=5,
        return_messages=False
    )

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=f"""
System: {SYSTEM_PROMPT}

Conversation so far:
{{history}}

User: {{input}}
Assistant:
"""
    )

    st.session_state.chatbot = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt
    )

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("Tony's Awesome Chatbot")

user_input = st.text_input("You:")

if user_input:
    response = st.session_state.chatbot.predict(input=user_input)
    st.write("Bot:", response)
