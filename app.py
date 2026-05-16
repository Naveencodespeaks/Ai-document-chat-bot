# app.py
import os
import tempfile
import streamlit as st
from backend.ingest import load_and_embed_pdf, load_existing_vectorstore
from backend.chain import build_chain, ask_question

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Support Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Customer Support Chatbot")
st.caption("Upload a PDF and ask questions about it")

# ── Session state ─────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "llm" not in st.session_state:
    st.session_state.llm = None

# ── Sidebar — PDF Upload ──────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and not st.session_state.pdf_loaded:
        with st.spinner("Reading and embedding PDF... this may take a minute"):
            # Save uploaded file to a temp path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Process PDF
            vectorstore = load_and_embed_pdf(tmp_path)
            st.session_state.chain, st.session_state.llm = build_chain(vectorstore)
            st.session_state.pdf_loaded = True
            os.unlink(tmp_path)  # clean up temp file

        st.success(f"✅ '{uploaded_file.name}' loaded!")
        st.info("Now ask questions in the chat →")

    if st.session_state.pdf_loaded:
        if st.button("🔄 Load a different PDF"):
            st.session_state.chain = None
            st.session_state.messages = []
            st.session_state.pdf_loaded = False
            st.rerun()

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload any company PDF")
    st.markdown("2. Ask questions in plain English")
    st.markdown("3. Get instant answers from the doc")

# ── Chat UI ───────────────────────────────────────────────
if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF from the sidebar to get started")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                st.caption(f"📄 Source pages: {msg['sources']}")

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(st.session_state.chain, st.session_state.llm, prompt)                
                answer = result["answer"]
                sources = result["sources"]

            st.markdown(answer)
            st.caption(f"📄 Source pages: {sources}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })