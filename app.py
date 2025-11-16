# app.py

import streamlit as st
import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read HF token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Show token loaded (for debugging only — delete later)
# st.write("Token starts with:", HUGGINGFACE_TOKEN[:8])

# -----------------------------
# Load Chunks
# -----------------------------
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -----------------------------
# Load embeddings
# -----------------------------
embeddings = np.load("embeddings.npy")
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# -----------------------------
# Create FAISS index
# -----------------------------
d = embeddings_norm.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings_norm)

# -----------------------------
# Load embedding model
# -----------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Load HF Chat Model
# -----------------------------
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=HUGGINGFACE_TOKEN
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 RAG Demo — ML Knowledge Base")

query = st.text_input("Ask a question:")


# -----------------------------
# 1️⃣ Retrieve Relevant Chunks
# -----------------------------
def retrieve_chunks(query, k=3):
    q_emb = embedder.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores, ids = index.search(q_emb.astype(np.float32), k)

    results = []
    for i, idx in enumerate(ids[0]):
        results.append({
            "chunk_id": int(idx),
            "score": float(scores[0][i]),
            "text": chunks[idx]["text"]
        })

    return results


# -----------------------------
# 2️⃣ Build Prompt for LLM
# -----------------------------
def build_prompt(query, retrieved):
    ctx = ""
    for i, c in enumerate(retrieved):
        ctx += f"[{i}] {c['text']}\n\n"

    return f"""
You are an AI assistant. Use ONLY the context below to answer.

Context:
{ctx}

Question: {query}

Answer with citation numbers like [0], [1].
If answer not found in context, say "I don't know".
"""


# -----------------------------
# 3️⃣ Final RAG Answer
# -----------------------------
if query:
    retrieved = retrieve_chunks(query)
    prompt = build_prompt(query, retrieved)

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )

    st.write("### 📘 Response:")
    st.write(response.choices[0].message["content"])

    st.write("---")
    st.write("### 📄 Retrieved Chunks (Context Used):")
    st.json(retrieved)
