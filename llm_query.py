import os
import faiss
import pickle
import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from llama_cpp import Llama
from PyPDF2 import PdfReader
from docx import Document
import requests


@st.cache_resource
def load_faiss_index(index_folder):
    index = faiss.read_index(os.path.join(index_folder, "faiss_index.index"))
    with open(os.path.join(index_folder, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def embed_text(texts, tokenizer, device, embed_model):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# --- LLaMA Answer Generation ---
def answer_with_llama(context, question, max_tokens=1024):
    prompt = f"""Use the context below to answer the question clearly and precisely.

            Context:
            {context}

            Question:
            {question}

            Answer:"""
    # output = llama_model(prompt, max_tokens=max_tokens)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    # return output["choices"][0]["text"].strip()
    return response.json()["response"].strip()


# --- Query Document ---
def query_document(question, index_folder,tokenizer, device, embed_model, top_k=5):
    index, metadata = load_faiss_index(index_folder)
    query_vec = embed_text([question], tokenizer, device, embed_model)
    D, I = index.search(query_vec, k=top_k)
    context_chunks = [metadata[i] for i in I[0] if i < len(metadata)]
    context = "\n".join(context_chunks)
   # context = context[:1500]  # Truncate to fit LLaMA's context window
    answer = answer_with_llama(context, question)
    return answer, context_chunks
      
  
