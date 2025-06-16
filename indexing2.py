import os
import faiss
import pickle
import numpy as np

# Import from your new shared utility file
import core_utils


def store_faiss_index(vectors, metadata, output_folder):
    """Creates and saves a FAISS index and its metadata."""
    os.makedirs(output_folder, exist_ok=True)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors.astype(np.float32))

    faiss.write_index(index, os.path.join(output_folder, "faiss_index.index"))
    with open(os.path.join(output_folder, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print("FAISS index and metadata stored successfully.")


def process_and_index_document(file_path, tokenizer, device, embed_model, output_folder):
    """
    The main orchestration function for the indexing process.
    This calls functions from core_utils to perform the steps.
    """
    print("Step 1: Extracting text...")
    text = core_utils.extract_text(file_path)

    print("Step 2: Splitting text into chunks...")
    chunks = core_utils.split_text(text)

    if not chunks:
        raise ValueError("Document is empty or could not be chunked.")

    print(f"Step 3: Embedding {len(chunks)} chunks...")
    vectors = core_utils.embed_text(chunks, tokenizer, device, embed_model)

    print("Step 4: Storing vectors in FAISS index...")
    store_faiss_index(vectors, chunks, output_folder)