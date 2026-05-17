# backend/chain.py
import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from supabase import create_client
from backend.ingest import search_documents

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


def build_llm():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
    return llm


def invoke_with_retry(llm, prompt, retries=3):
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
                continue
    return None


def save_message(session_id: str, role: str, message: str, user_id: int = None):
    supabase.table("chat_history").insert({
        "session_id": session_id,
        "role": role,
        "message": message,
        "user_id": user_id
    }).execute()


def get_chat_history(session_id: str):
    result = supabase.table("chat_history") \
        .select("role, message") \
        .eq("session_id", session_id) \
        .order("created_at") \
        .execute()
    return result.data


def save_user(session_id: str, name: str, email: str) -> int:
    existing = supabase.table("users") \
        .select("id") \
        .eq("session_id", session_id) \
        .execute()

    if existing.data:
        supabase.table("users") \
            .update({"last_seen": "now()"}) \
            .eq("session_id", session_id) \
            .execute()
        return existing.data[0]["id"]

    result = supabase.table("users").insert({
        "session_id": session_id,
        "name": name,
        "email": email
    }).execute()
    return result.data[0]["id"]


def ask_question(llm, question: str, session_id: str, user_id: int) -> dict:
    # Step 1 — classify message
    classifier_prompt = f"""You are a classifier. Decide if the user message is:
A) Small talk / greeting / general conversation (NOT about any document)
B) A question that needs to be answered from a document

User message: "{question}"

Reply with only one letter: A or B"""

    classification = invoke_with_retry(llm, classifier_prompt)
    print(f"DEBUG classification: {classification}")

    if not classification:
        return {"answer": "Connection error. Please try again in a moment.", "sources": []}

    classification = classification.upper()

    # Step 2 — small talk
    if classification == "A":
        chat_prompt = f"""You are a friendly AI assistant.
Reply naturally and helpfully to this message in 1-2 sentences.
User: {question}"""

        reply = invoke_with_retry(llm, chat_prompt)
        if not reply:
            reply = "Connection error. Please try again."

        save_message(session_id, "user", question, user_id)
        save_message(session_id, "assistant", reply, user_id)
        return {"answer": reply, "sources": []}

    # Step 3 — document question
    docs = search_documents(question, top_k=3)
    print(f"DEBUG docs: {docs}")

    if not docs:
        reply = "I couldn't find relevant information in the document. Please try rephrasing your question."
        save_message(session_id, "user", question, user_id)
        save_message(session_id, "assistant", reply, user_id)
        return {"answer": reply, "sources": []}

    # Build context
    context = "\n\n".join([doc["content"] for doc in docs])

    # Fix metadata parsing — handle both dict and string
    sources = []
    for doc in docs:
        meta = doc["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        sources.append(meta.get("page", "unknown"))

    # Get chat history
    history = get_chat_history(session_id)
    history_text = "\n".join([
        f"{h['role'].capitalize()}: {h['message']}"
        for h in history[-6:]
    ])

    # Build prompt
    prompt = f"""You are a helpful AI customer support assistant.
Use the following document context to answer the user's question.
If the answer is not in the context, say you don't know.

Chat History:
{history_text}

Document Context:
{context}

User Question: {question}

Answer:"""

    reply = invoke_with_retry(llm, prompt)
    if not reply:
        reply = "Connection error. Please try again."

    save_message(session_id, "user", question, user_id)
    save_message(session_id, "assistant", reply, user_id)

    return {"answer": reply, "sources": sources}