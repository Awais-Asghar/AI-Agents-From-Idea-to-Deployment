"""Utility script to build the FAISS vector store backing the local RAG tool."""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.rag_tool import DEFAULT_EMBEDDING_MODEL

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
DEFAULT_DOC = DOCUMENTS_DIR / "sample_docs.txt"


def build_vector_store(doc_path: Path = DEFAULT_DOC, *, chunk_size: int = 600, chunk_overlap: int = 50) -> None:
    """Build a FAISS index from the supplied document."""
    if not doc_path.exists():
        raise FileNotFoundError(f"Document source not found at {doc_path}")

    text = doc_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTORSTORE_DIR))
    print(f"Vector store saved to {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vector_store()
