"""
RAG Chatbot using Ollama (Phi3) + LangChain + ChromaDB
--------------------------------------------------------
Supports: PDF, DOCX, TXT documents
UI: Gradio web interface
"""

import os
import gradio as gr
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────
# CONFIGURATION  — edit these as needed
# ─────────────────────────────────────────────
OLLAMA_MODEL       = "phi3"          # must be pulled in Ollama
EMBED_MODEL        = "phi3"          # same model used for embeddings
CHROMA_DB_DIR      = "./chroma_db"   # where vector DB is stored on disk
DOCS_FOLDER        = "./documents"   # drop your files here
CHUNK_SIZE         = 500             # characters per chunk
CHUNK_OVERLAP      = 50              # overlap between chunks
TOP_K_RESULTS      = 3               # how many chunks to retrieve per query
# ─────────────────────────────────────────────


# ── Prompt template ──────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer
the question. If the answer is not in the context, say "I don't have enough information
in the provided documents to answer that."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_documents(folder: str) -> list:
    """Load all supported documents from *folder*."""
    docs = []
    folder_path = Path(folder)

    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        print(f"📁 Created '{folder}' — add your documents there and restart.")
        return docs

    for file_path in folder_path.iterdir():
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                print(f"⚠️  Skipping unsupported file: {file_path.name}")
                continue

            loaded = loader.load()
            docs.extend(loaded)
            print(f"✅ Loaded: {file_path.name}  ({len(loaded)} page(s)/section(s))")
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")

    return docs


def build_vector_store(docs: list) -> Chroma:
    """Split documents and store embeddings in ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"🔪 Split into {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"💾 Vector store saved to '{CHROMA_DB_DIR}'")
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )


def create_qa_chain(vector_store: Chroma) -> RetrievalQA:
    """Create a RetrievalQA chain backed by Phi3 via Ollama."""
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# ── Initialise ────────────────────────────────────────────────────────────────

def initialise() -> RetrievalQA:
    """
    Build or load the vector store, then return a QA chain.
    Re-ingests every time the script runs.  
    Tip: wrap the build step with a flag file check to skip re-ingestion.
    """
    print("\n🚀 Initialising RAG Chatbot...")

    docs = load_documents(DOCS_FOLDER)

    if not docs:
        print("⚠️  No documents found. The bot will run without context.")
        # Create an empty store so ChromaDB doesn't crash
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
        )
    else:
        vector_store = build_vector_store(docs)

    qa_chain = create_qa_chain(vector_store)
    print("✅ Chatbot ready!\n")
    return qa_chain


qa_chain = initialise()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    """Process one user message and return the assistant reply."""
    if not message.strip():
        return "Please enter a question."

    try:
        result = qa_chain.invoke({"query": message})
        answer = result["result"]

        # Append source filenames (optional — remove if noisy)
        sources = result.get("source_documents", [])
        if sources:
            filenames = sorted({
                Path(doc.metadata.get("source", "unknown")).name
                for doc in sources
            })
            answer += f"\n\n📄 *Sources: {', '.join(filenames)}*"

        return answer
    except Exception as e:
        return f"❌ Error: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="🤖 RAG Chatbot (Phi3 + Ollama)",
    description=(
        "Ask questions about your documents.  \n"
        f"Documents folder: `{DOCS_FOLDER}`  |  "
        f"Model: `{OLLAMA_MODEL}`"
    ),
    examples=[
        "What is this document about?",
        "Summarise the key points.",
        "What are the main topics covered?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
