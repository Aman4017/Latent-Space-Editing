"""
ingest.py  —  Run this standalone to (re)build the vector database.
Use when you add/update documents without wanting to launch the full chatbot.

Usage:
    python ingest.py
"""

import shutil
from pathlib import Path
from chatbot import (
    CHROMA_DB_DIR,
    DOCS_FOLDER,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL,
    load_documents,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


def ingest(force_rebuild: bool = True):
    if force_rebuild and Path(CHROMA_DB_DIR).exists():
        print(f"🗑️  Removing old vector store at '{CHROMA_DB_DIR}'...")
        shutil.rmtree(CHROMA_DB_DIR)

    docs = load_documents(DOCS_FOLDER)
    if not docs:
        print("❌ No documents to ingest. Add files to the 'documents' folder.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"🔪 Total chunks: {len(chunks)}")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"✅ Ingestion complete! Vector store saved to '{CHROMA_DB_DIR}'")


if __name__ == "__main__":
    ingest()
