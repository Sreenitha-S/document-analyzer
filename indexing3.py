# indexing.py
import os
import faiss
import pickle
import torch
import numpy as np
from PyPDF2 import PdfReader
from docx import Document

# --- Text Extraction ---
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return "\n".join([page.extract_text() or "" for page in PdfReader(file_path).pages])
    elif ext == ".docx":
        return "\n".join([p.text for p in Document(file_path).paragraphs])
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# --- Text Chunking ---
def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# --- Embedding Text ---
def embed_text(texts, tokenizer, device, embed_model):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# --- FAISS Indexing ---
def store_faiss_index(vectors, metadata, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, os.path.join(output_folder, "faiss_index.index"))
    with open(os.path.join(output_folder, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
