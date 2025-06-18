import os
import faiss
import pickle
import requests
import numpy as np

# Import from your new shared utility file (assuming core_utils.py exists and works)
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


def answer_with_llama(context, question, model_tag="llama3.2:3b", ollama_url="http://localhost:11434"):
    """Generates an answer using a local Ollama model."""
    prompt = f"""Use the context below to answer the question clearly and precisely.

            Context:
            {context}

            Question:
            {question}

            Answer:"""

    print(f"Querying Ollama at {ollama_url} with model: {model_tag}")
    try:
        response = requests.post(
            f"{ollama_url}/api/generate", # Uses the provided ollama_url
            json={
                "model": model_tag,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return f"Error: Could not retrieve answer from Ollama. {e}"


def query_document(question, index_folder, tokenizer, device, embed_model, top_k=5, ollama_url="http://localhost:11434"):
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
    answer = answer_with_llama(context, question, ollama_url=ollama_url)  # Passes the ollama_url received
    return answer, context_chunks