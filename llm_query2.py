import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # FIX: To prevent OpenMP runtime error

import faiss
import pickle
import requests
import numpy as np

# Import from your new shared utility file
import core_utils


def load_faiss_index(index_folder):
    """Loads a FAISS index and its metadata from a directory."""
    index_path = os.path.join(index_folder, "faiss_index.index")
    metadata_path = os.path.join(index_folder, "metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None

    print("Loading FAISS index and metadata...")
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def answer_with_llama(context, question, model_tag="llama3.2:3b"):
    """Generates an answer using a local Ollama model."""
    prompt = f"""Use the context below to answer the question clearly and precisely.

            Context:
            {context}

            Question:
            {question}

            Answer:"""

    print(f"Querying Ollama with model: {model_tag}")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_tag,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def query_document(question, index_folder, tokenizer, device, embed_model, top_k=5):
    """Orchestrates the process of querying the document."""
    index, metadata = load_faiss_index(index_folder)
    if index is None:
        raise FileNotFoundError("FAISS index not found. Please index a document first.")

    print("Embedding user query...")
    query_vec = core_utils.embed_text([question], tokenizer, device, embed_model)

    print(f"Searching index for top {top_k} relevant chunks...")
    distances, indices = index.search(query_vec.astype(np.float32), k=top_k)

    context_chunks = [metadata[i] for i in indices[0] if i < len(metadata)]
    if not context_chunks:
        return "Could not find any relevant context in the document.", []

    context = "\n".join(context_chunks)

    print("Generating answer with local LLM...")
    answer = answer_with_llama(context, question)
    return answer, context_chunks