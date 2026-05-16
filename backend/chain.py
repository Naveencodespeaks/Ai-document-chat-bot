# backend/chain.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()

def build_chain(vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    return chain, llm


def ask_question(chain, llm, question: str) -> dict:
    # Step 1 — Ask LLM if this is small talk or a document question
    classifier_prompt = f"""You are a classifier. Decide if the user message is:
A) Small talk / greeting / general conversation (NOT about any document)
B) A question that needs to be answered from a document

User message: "{question}"

Reply with only one letter: A or B"""

    classification = llm.invoke(classifier_prompt).content.strip().upper()

    # Step 2 — If small talk, reply naturally without searching PDF
    if classification == "A":
        chat_prompt = f"""You are a friendly AI assistant. 
Reply naturally and helpfully to this message in 1-2 sentences.
User: {question}"""
        reply = llm.invoke(chat_prompt).content.strip()
        return {
            "answer": reply,
            "sources": []
        }

    # Step 3 — If document question, search the PDF
    response = chain.invoke({"question": question})
    return {
        "answer": response["answer"],
        "sources": [
            doc.metadata.get("page", "unknown")
            for doc in response["source_documents"]
        ]
    }