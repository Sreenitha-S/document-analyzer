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

# --- Session State Initialization ---
st.set_page_config(page_title="Document Q&A with LLaMA", layout="wide")
st.title("📚 Document Q&A with LLaMA 3.2 3B")
st.markdown("Upload documents, index them, and ask questions or request summaries.")

# --- Session Variables ---
defaults = {
    "chat_history": [],
    "indexed": False,
    "session_id": str(uuid.uuid4()),
    "summaries": {},
    "all_raw_text": {},
    "uploaded_file_paths": [],
    "original_filenames": [],
    "last_question": "",
    "last_context": []
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.session_state.user_index_folder = os.path.join("vector_index", st.session_state.session_id)
os.makedirs(st.session_state.user_index_folder, exist_ok=True)
os.makedirs("temp_dir", exist_ok=True)

# --- File Upload ---
uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
new_files_uploaded = False

if uploaded_files:
    if not st.session_state.indexed:
        st.session_state.uploaded_file_paths = []
        st.session_state.original_filenames = []
        st.session_state.all_raw_text = {}
        new_files_uploaded = True

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            temp_path = os.path.join("temp_dir", f"{st.session_state.session_id}_{filename}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_file_paths.append(temp_path)
            st.session_state.original_filenames.append(filename)

        st.success(f"{len(uploaded_files)} file(s) uploaded: " + ", ".join(st.session_state.original_filenames))

# --- Process and Index ---
process_button = st.button("📄 Process and Index Document(s)", disabled=not uploaded_files or st.session_state.indexed)

if process_button:
    all_chunks = []
    all_sources = []
    with st.spinner("Processing and indexing all documents..."):
        for i, path in enumerate(st.session_state.uploaded_file_paths):
            text = extract_text(path)
            filename = st.session_state.original_filenames[i]
            st.session_state.all_raw_text[filename] = text
            chunks = split_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks))

        vectors = embed_text(all_chunks, tokenizer, device, embed_model)
        store_faiss_index(vectors, list(zip(all_chunks, all_sources)), st.session_state.user_index_folder)

    st.session_state.indexed = True
    st.success("✅ All documents indexed successfully!")

# --- Q&A Interface ---
question = st.text_input("❓ Ask a question:")

if question and question != st.session_state.last_question:
    index_path = os.path.join(st.session_state.user_index_folder, "faiss_index.index")
    if st.session_state.indexed and os.path.exists(index_path):
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
            st.session_state.last_question = question
            st.session_state.last_context = context
    else:
        st.error("❌ Please process and index the uploaded documents first.")

# --- Display Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Show Context Used ---
if st.session_state.last_context:
    with st.expander("📚 Context Used"):
        for ctx, source in st.session_state.last_context:
            st.markdown(f"**Source: {source}**\n\n{ctx}")

# --- Summarization ---
summary_button = st.button("📝 Generate Summary for Each Document", disabled=not st.session_state.indexed)

if summary_button:
    st.session_state.summaries = {}
    with st.spinner("Generating summaries..."):
        for filename in st.session_state.original_filenames:
            text = st.session_state.all_raw_text.get(filename)
            if text:
                summary = summarize_documents(text)
                st.session_state.summaries[filename] = summary

# --- Show Summaries ---
if st.session_state.summaries:
    st.subheader("📝 Individual Document Summaries")
    for filename, summary in st.session_state.summaries.items():
        st.markdown(f"### 📄 {filename}")
        st.markdown(summary)
