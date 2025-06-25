# main_app.py
import os
import uuid
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

from indexing import extract_text, split_text, embed_text, store_faiss_index
from llm_query import query_document, summarize_documents

# --- Settings ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL).to(device)

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "user_index_folder" not in st.session_state:
    st.session_state.user_index_folder = os.path.join("vector_index", st.session_state.session_id)

if "summaries" not in st.session_state:
    st.session_state.summaries = {}

# --- Setup Folders ---
os.makedirs(st.session_state.user_index_folder, exist_ok=True)
os.makedirs("temp_dir", exist_ok=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Document Q&A with LLaMA", layout="wide")
st.title("📚 Document Q&A with LLaMA 3.2 3B")
st.markdown("Upload documents, index them, and ask questions or request summaries.")

uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.file_uploaded = True
    st.session_state.uploaded_file_paths = []
    st.session_state.source_map = []

    for uploaded_file in uploaded_files:
        temp_path = os.path.join("temp_dir", f"{st.session_state.session_id}_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_paths.append(temp_path)

    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

if st.session_state.get("file_uploaded") and st.button("📄 Process and Index Document(s)"):
    all_chunks = []
    all_sources = []
    st.session_state.all_raw_text = {}
    with st.spinner("Processing and indexing all documents..."):
        for path in st.session_state.uploaded_file_paths:
            text = extract_text(path)
            filename = os.path.basename(path)
            st.session_state.all_raw_text[filename] = text
            chunks = split_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks))

        vectors = embed_text(all_chunks, tokenizer, device, embed_model)
        store_faiss_index(vectors, list(zip(all_chunks, all_sources)), st.session_state.user_index_folder)
    st.success("All documents indexed successfully!")

# --- Q&A Interface ---
question = st.text_input("❓ Ask a question:")

if question:
    if os.path.exists(os.path.join(st.session_state.user_index_folder, "faiss_index.index")):
        with st.spinner("Generating answer..."):
            answer, context = query_document(
                question,
                st.session_state.user_index_folder,
                tokenizer,
                device,
                embed_model
            )
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.error("Please upload and index document(s) first.")

# --- Display Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Display Used Context ---
if question:
    with st.expander("📚 Context Used"):
        for ctx, source in context:
            st.markdown(f"**Source: {source}**\n\n{ctx}")

# --- Summary Button ---
if st.session_state.get("file_uploaded") and st.button("📝 Generate Summary for Each Document"):
    st.session_state.summaries = {}
    with st.spinner("Generating summaries..."):
        for path in st.session_state.uploaded_file_paths:
            filename = os.path.basename(path)
            text = st.session_state.all_raw_text.get(filename)
            if text:
                summary = summarize_documents(text)
                st.session_state.summaries[filename] = summary

if st.session_state.get("summaries"):
    st.subheader("📝 Individual Document Summaries")
    for filename, summary in st.session_state.summaries.items():
        st.markdown(f"### 📄 {filename}")
        st.markdown(summary)
