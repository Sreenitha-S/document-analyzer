mport os
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


# --- Environment Setup ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Load Embedding Model ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL).to(device)

# --- Load LLaMA Model ---
MODEL_PATH = r"C:\Users\riset\Downloads\Llama-3.2-3B-Instruct-IQ3_M.gguf"
llama_model = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=16,
    n_gpu_layers=-1  # Adjust based on your GPU capabilities
)

# --- Streamlit Caching ---
@st.cache_resource
def load_faiss_index(index_folder):
    index = faiss.read_index(os.path.join(index_folder, "faiss_index.index"))
    with open(os.path.join(index_folder, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

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
def embed_text(texts):
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
def query_document(question, index_folder, top_k=5):
    index, metadata = load_faiss_index(index_folder)
    query_vec = embed_text([question])
    D, I = index.search(query_vec, k=top_k)
    context_chunks = [metadata[i] for i in I[0] if i < len(metadata)]
    context = "\n".join(context_chunks)
    context = context[:1500]  # Truncate to fit LLaMA's context window
    answer = answer_with_llama(context, question)
    return answer, context_chunks

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Streamlit Interface ---
st.set_page_config(page_title="Document Q&A with LLaMA", layout="wide")
st.title("📚 Document Q&A with LLaMA 3.2 3B")
st.markdown("Upload a document, index it, and ask questions.")

index_folder = r"C:\Users\riset\PycharmProjects\doc_analyzer3\vector_index2"

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
if uploaded_file:
    file_path = os.path.join("tempDir", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    if st.button("Process and Index Document"):
        with st.spinner("Processing document..."):
            text = extract_text(file_path)
            chunks = split_text(text)
            vectors = embed_text(chunks)
            store_faiss_index(vectors, chunks, index_folder)
        st.success("Document processed and indexed successfully!")

# --- Handle User Input ---
question = st.text_input("❓ Ask a question:")
if question:
    with st.spinner("Generating answer..."):
        answer, context = query_document(question, index_folder)
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