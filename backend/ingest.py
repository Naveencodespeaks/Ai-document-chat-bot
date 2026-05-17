# backend/ingest.py
import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

embeddings_model = get_embeddings_model()


def load_and_embed_pdf(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    supabase.table("documents").delete().neq("id", 0).execute()
    print("Cleared old documents ✓")

    print("Embedding and storing in Supabase...")
    for i, chunk in enumerate(chunks):
        embedding = [float(x) for x in embeddings_model.embed_query(chunk.page_content)]
        supabase.table("documents").insert({
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": embedding
        }).execute()
        if i % 10 == 0:
            print(f"  Stored {i+1}/{len(chunks)} chunks...")

    print(f"✓ Stored {len(chunks)} chunks in Supabase")
    return True


def search_documents(query: str, top_k: int = 3):
    query_embedding = [float(x) for x in embeddings_model.embed_query(query)]

    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": top_k
    }).execute()

    print(f"DEBUG search: {len(result.data)} results found")
    return result.data