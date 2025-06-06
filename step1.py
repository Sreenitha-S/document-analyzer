import os
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from transformers import AutoTokenizer, AutoModel
import torch

# Load embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
model = AutoModel.from_pretrained(EMBED_MODEL).to(device)

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

def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def embed_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def store_faiss_index(vectors, metadata, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, os.path.join(output_folder, "faiss_index.index"))
    with open(os.path.join(output_folder, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

def process_document(file_path, output_folder):
    print(f"Reading: {file_path}")
    text = extract_text(file_path)
    chunks = split_text(text)
    print(f"Total chunks: {len(chunks)}")

    all_vectors, metadata = [], []
    for i in range(0, len(chunks), 8):
        batch = chunks[i:i + 8]
        vectors = embed_text(batch)
        all_vectors.append(vectors)
        metadata.extend(batch)

    store_faiss_index(np.vstack(all_vectors), metadata, output_folder)
    print(f"Stored {len(metadata)} vectors in: {output_folder}")

if __name__ == "__main__":
    process_document("C:\Users\riset\Downloads\Feeder-vehicle-policy_KMRL.pdf", "vector_index")