import os
import torch
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from docx import Document

# --- Text Extraction ---
def extract_text(file_path):
    """Extracts text from PDF, DOCX, or TXT files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            return "\n".join([page.extract_text() or "" for page in PdfReader(file_path).pages])
        except Exception as e:
            raise ValueError(f"Could not read PDF file: {file_path}. It may be scanned or corrupted.") from e
    elif ext == ".docx":
        return "\n".join([p.text for p in Document(file_path).paragraphs])
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# --- Text Chunking ---
def split_text(text, chunk_size=500, overlap=50):
    """Splits text into chunks of words."""
    words = text.split()
    if len(words) < chunk_size:
        return [" ".join(words)]
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# --- Embedding Text ---
def embed_text(texts, tokenizer, device, embed_model):
    """Creates vector embeddings for a list of text chunks."""
    if not texts:
        return []
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # Use mean pooling for a robust sentence representation
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()