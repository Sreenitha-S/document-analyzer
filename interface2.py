import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # FIX: To prevent OpenMP runtime error

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
from queue import Queue
import torch
from transformers import AutoTokenizer, AutoModel

# --- Import your refactored backend modules with new names ---
import indexing2
import llm_query2

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_OUTPUT_FOLDER = "vector_index"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

class DocQAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Q&A (Offline)")
        self.root.geometry("1100x800")
        self.root.configure(bg="black")

        self.style = self.setup_styles()
        self.queue = Queue()

        self.file_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.embed_model = self.load_embedding_model()

        main_frame = ttk.Frame(root, padding="10", style="App.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_upload_section(main_frame)
        self.create_processing_section(main_frame)
        self.create_ollama_url_section(main_frame)
        self.create_qa_section(main_frame)

        self.root.after(100, self.process_queue)

    def load_embedding_model(self):
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on device '{self.device}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(self.device)
            print("Embedding model loaded successfully.")
            return tokenizer, model
        except Exception as e:
            messagebox.showerror("Model Error",
                                 f"Failed to load embedding model: {e}\n\nPlease check your internet connection for the first download and ensure 'transformers' and 'torch' are installed.")
            self.root.quit()
            return None, None

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("App.TFrame", background="black")
        style.configure("Section.TFrame", background="#1c1c1c")
        style.configure("TLabel", background="black", foreground="white", font=('Segoe UI', 10))
        style.configure("Header.TLabel", font=('Segoe UI', 14, 'bold'), background="#1c1c1c")
        style.configure("SubHeader.TLabel", font=('Segoe UI', 10), background="#1c1c1c")
        style.configure("Status.TLabel", font=('Segoe UI', 10), background="#1c1c1c")
        style.configure("TButton", background="#333333", foreground="white", font=('Segoe UI', 10, 'bold'),
                        borderwidth=1, relief="solid")
        style.map("TButton", background=[('active', '#555555')])
        self.root.option_add('*Text*background', '#101010')
        self.root.option_add('*Text*foreground', '#d4d4d4')
        self.root.option_add('*Text*font', ('Consolas', 11))
        self.root.option_add('*Text*selectBackground', '#007acc')
        self.root.option_add('*Text*insertBackground', 'white')
        return style

    def create_upload_section(self, parent):
        frame = ttk.Frame(parent, padding=10, style="Section.TFrame")
        frame.pack(fill='x', pady=5)
        ttk.Label(frame, text="Document Analyzer App", style="Header.TLabel").pack()
        ttk.Label(frame, text="Upload a document, index it, and ask questions.", style="SubHeader.TLabel").pack(
            pady=(0, 10))
        ttk.Button(frame, text="Browse Files", command=self.browse_file).pack(pady=5)
        self.upload_status_label = ttk.Label(frame, text="No file selected.", style="Status.TLabel")
        self.upload_status_label.pack(pady=5)

    def create_processing_section(self, parent):
        frame = ttk.Frame(parent, padding=10, style="Section.TFrame")
        frame.pack(fill='x', pady=5)
        self.process_button = ttk.Button(frame, text="Process and Index Document", command=self.start_indexing_thread,
                                         state="disabled")
        self.process_button.pack(pady=10)
        self.processing_status_label = ttk.Label(frame, text=f"Index will be stored in: '{INDEX_OUTPUT_FOLDER}'",
                                                 style="Status.TLabel")
        self.processing_status_label.pack(pady=5)

    def create_ollama_url_section(self, parent):
            frame = ttk.Frame(parent, padding=10, style="Section.TFrame")
            frame.pack(fill='x', pady=5)
            ttk.Label(frame, text="Ollama API URL:", style="SubHeader.TLabel").pack(anchor='w')
            self.ollama_url_entry = ttk.Entry(frame, width=50)
            self.ollama_url_entry.insert(0, DEFAULT_OLLAMA_URL)  # Set default value
            self.ollama_url_entry.pack(fill='x', pady=5)
            ttk.Label(frame, text="e.g., http://localhost:11434 or http://<your_server_ip>:11434",  style="Status.TLabel").pack(anchor='w')

    def create_qa_section(self, parent):
        frame = ttk.Frame(parent, padding=10, style="Section.TFrame")
        frame.pack(fill='both', expand=True, pady=5)
        ttk.Label(frame, text="Ask a question:", style="SubHeader.TLabel", font=('Segoe UI', 12, 'bold')).pack(
            anchor='w')
        self.chat_display = scrolledtext.ScrolledText(frame, wrap=tk.WORD, state='disabled', height=15, relief="flat")
        self.chat_display.pack(fill='both', expand=True, pady=10)
        input_frame = ttk.Frame(frame, style="Section.TFrame")
        input_frame.pack(fill='x')
        self.question_input = tk.Text(input_frame, height=3, relief="flat")
        self.question_input.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.question_input.bind("<Return>", self.start_query_thread)
        self.ask_button = ttk.Button(input_frame, text="Ask", command=self.start_query_thread, state="disabled")
        self.ask_button.pack(side='right')
        self.chat_display.tag_config('user', foreground='#60a5fa', font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_config('bot', foreground='#a7f3d0', font=('Segoe UI', 11))
        self.chat_display.tag_config('context', foreground='#777777', font=('Consolas', 9, 'italic'))

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Supported Files", "*.pdf;*.docx;*.txt"), ("All files", "*.*")])
        if path:
            self.file_path = path
            self.upload_status_label.config(text=f"File '{os.path.basename(path)}' uploaded successfully!",
                                            foreground="#34d399")
            self.process_button.config(state="normal")
            self.processing_status_label.config(text=f"Ready to process '{os.path.basename(path)}'.",
                                                foreground="white")

    def start_indexing_thread(self):
        if not self.file_path:
            messagebox.showwarning("No File", "Please select a document file first.")
            return
        self.processing_status_label.config(text="Processing and indexing... Please wait.", foreground="orange")
        self.process_button.config(state="disabled")
        thread = threading.Thread(target=self._index_document_worker, daemon=True)
        thread.start()

    def _index_document_worker(self):
        try:
            indexing2.process_and_index_document(self.file_path, self.tokenizer, self.device, self.embed_model,
                                                 INDEX_OUTPUT_FOLDER)
            self.queue.put(("indexing_success", "Document processed and indexed successfully!"))
        except Exception as e:
            self.queue.put(("error", f"Indexing Error: {e}"))
        finally:
            self.queue.put(("enable_process_button", True))

    def start_query_thread(self, event=None):
        question = self.question_input.get("1.0", tk.END).strip()
        if not question:
            return 'break'
        if not os.path.exists(os.path.join(INDEX_OUTPUT_FOLDER, "faiss_index.index")):
            messagebox.showwarning("No Index",
                                   "The document has not been indexed yet. Please process the document first.")
            return 'break'
        ollama_url = self.ollama_url_entry.get().strip()
        if not ollama_url:
            messagebox.showwarning("Ollama URL Missing", "Please enter the Ollama API URL.")
            return 'break'
        thread = threading.Thread(target=self._query_document_worker, args=(question, ollama_url,), daemon=True)

        self.update_chat_display("user", question)
        self.question_input.delete("1.0", tk.END)
        self.ask_button.config(state="disabled")
       # thread = threading.Thread(target=self._query_document_worker, args=(question,), daemon=True)
        thread.start()
        return 'break'

    def _query_document_worker(self, question, ollama_url):
        try:
            answer, context_chunks = llm_query2.query_document(question, INDEX_OUTPUT_FOLDER, self.tokenizer,
                                                               self.device, self.embed_model, ollama_url=ollama_url)
            self.queue.put(("new_answer", (answer, context_chunks)))
        except Exception as e:
            self.queue.put(("error", f"Query Error: {e}"))
        finally:
            self.queue.put(("enable_ask_button", True))

    def update_chat_display(self, sender, message, context=None):
        self.chat_display.config(state='normal')
        if sender == "user":
            self.chat_display.insert(tk.END, f"You: {message}\n", 'user')
        elif sender == "bot":
            self.chat_display.insert(tk.END, f"Analyzer: {message}\n", 'bot')
            if context:
                context_str = "\n".join([f"  - {chunk[:80]}..." for chunk in context])
                self.chat_display.insert(tk.END, f"\n[Context Used]:\n{context_str}\n\n", 'context')
        self.chat_display.config(state='disabled')
        self.chat_display.yview(tk.END)

    def process_queue(self):
        try:
            msg_type, data = self.queue.get_nowait()
            if msg_type == "indexing_success":
                self.processing_status_label.config(text=data, foreground="#34d399")
                self.ask_button.config(state="normal")
            elif msg_type == "new_answer":
                answer, context = data
                self.update_chat_display("bot", answer, context)
            elif msg_type == "error":
                self.processing_status_label.config(text="An error occurred. See message box.", foreground="red")
                messagebox.showerror("Application Error", data)
            elif msg_type == "enable_process_button":
                self.process_button.config(state="normal")
            elif msg_type == "enable_ask_button":
                self.ask_button.config(state="normal")
        except:
            pass
        finally:
            self.root.after(100, self.process_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = DocQAApp(root)
    root.mainloop()
