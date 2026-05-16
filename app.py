# app.py
import os
import uuid
import tempfile
import streamlit as st
from backend.ingest import load_and_embed_pdf
from backend.chain import build_llm, ask_question, save_user

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Support Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ── Session state ─────────────────────────────────────────
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_registered" not in st.session_state:
    st.session_state.user_registered = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# ── Screen 1 — User Registration ─────────────────────────
if not st.session_state.user_registered:
    st.title("🤖 AI Customer Support Chatbot")
    st.markdown("#### Welcome! Please introduce yourself to get started.")
    st.divider()

    with st.form("user_form"):
        name = st.text_input("👤 Your Name", placeholder="Sai Naveen")
        email = st.text_input("📧 Your Email", placeholder="sai@example.com")
        submitted = st.form_submit_button("Start Chatting →", use_container_width=True)

        if submitted:
            if not name or not email:
                st.error("Please fill in both name and email.")
            elif "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                with st.spinner("Setting up your session..."):
                    user_id = save_user(
                        session_id=st.session_state.session_id,
                        name=name,
                        email=email
                    )
                    st.session_state.user_id = user_id
                    st.session_state.user_name = name
                    st.session_state.user_registered = True
                    st.session_state.llm = build_llm()
                st.rerun()

# ── Screen 2 — Main Chatbot ───────────────────────────────
else:
    st.title("🤖 AI Customer Support Chatbot")
    st.caption(f"Welcome, {st.session_state.user_name}! 👋 Upload a PDF and ask questions about it.")

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file and not st.session_state.pdf_loaded:
            with st.spinner("Reading and embedding PDF... this may take a minute"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                load_and_embed_pdf(tmp_path)
                st.session_state.pdf_loaded = True
                os.unlink(tmp_path)

            st.success(f"✅ '{uploaded_file.name}' loaded!")
            st.info("Now ask questions in the chat →")

        if st.session_state.pdf_loaded:
            if st.button("🔄 Load a different PDF"):
                st.session_state.pdf_loaded = False
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.markdown(f"**Session ID:**")
        st.caption(st.session_state.session_id)
        st.divider()
        st.markdown("**How it works:**")
        st.markdown("1. Upload any company PDF")
        st.markdown("2. Ask questions in plain English")
        st.markdown("3. Get instant answers from the doc")
        st.markdown("4. Chat history saved to database")

        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Chat UI ───────────────────────────────────────────
    if not st.session_state.pdf_loaded:
        st.info("👈 Upload a PDF from the sidebar to get started")
    else:
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    st.caption(f"📄 Source pages: {msg['sources']}")

        # Chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = ask_question(
                        llm=st.session_state.llm,
                        question=prompt,
                        session_id=st.session_state.session_id,
                        user_id=st.session_state.user_id
                    )
                    answer = result["answer"]
                    sources = result["sources"]

                st.markdown(answer)
                if sources:
                    st.caption(f"📄 Source pages: {sources}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })