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
from indexing import extract_text, split_text, embed_text, store_faiss_index
from llm_query import query_document

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Load Embedding Model ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL).to(device)

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Streamlit Interface ---
st.set_page_config(page_title="Document Q&A with LLaMA", layout="wide")
st.title("📚 Document Q&A with LLaMA 3.2 3B")
st.markdown("Upload a document, index it, and ask questions.")

index_folder = "vector_index"

# Check if the index folder exists, and create it if not
if not os.path.exists(index_folder):
    os.makedirs(index_folder)
    st.success(f"Created index folder at: {index_folder}")
else:
    st.info(f"Index folder already exists at: {index_folder}")

temp_dir = "temp_dir"
# Check if the index folder exists, and create it if not
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
if uploaded_file:
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    if st.button("Process and Index Document"):
        with st.spinner("Processing document..."):
            text = extract_text(file_path)
            chunks = split_text(text)
            vectors = embed_text(chunks, tokenizer, device, embed_model)
            store_faiss_index(vectors, chunks, index_folder)
        st.success("Document processed and indexed successfully!")

# --- Handle User Input ---
question = st.text_input("❓ Ask a question:")
if question:
    with st.spinner("Generating answer..."):
        answer, context = query_document(question, index_folder, tokenizer, device, embed_model)
        # Store user question and assistant answer in chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# --- Display Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Display Context Used ---
if question:
    with st.expander("📚 Context Used"):
        st.text("\n".join(context))
