"""
capstone_streamlit.py — Course Assistant Agent
Run: streamlit run capstone_streamlit.py
"""
import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from agent import load_agent, DOCUMENTS

load_dotenv()

DOMAIN_NAME = "Agentic AI Course Assistant"
DOMAIN_DESCRIPTION = "A LangGraph-powered teaching assistant for 4th-year B.Tech students studying Agentic AI."
KB_TOPICS = [d["topic"] for d in DOCUMENTS]

st.set_page_config(page_title=DOMAIN_NAME, page_icon="🤖", layout="centered")
st.title(f"🤖 {DOMAIN_NAME}")
st.caption(DOMAIN_DESCRIPTION)

# ── Load models and KB (cached) ───────────────────────────
@st.cache_resource
def get_cached_agent():
    return load_agent()

try:
    agent_app, embedder, collection = get_cached_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write(DOMAIN_DESCRIPTION)
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics covered:**")
    for t in KB_TOPICS:
        st.write(f"• {t}")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Display history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ────────────────────────────────────────────
if prompt := st.chat_input("Ask something about Agentic AI..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")
        st.write(answer)
        faith = result.get("faithfulness", 0.0)
        if faith > 0:
            st.caption(f"Faithfulness: {faith:.2f} | Sources: {result.get('sources', [])}") 

    st.session_state.messages.append({"role":"assistant","content":answer})
