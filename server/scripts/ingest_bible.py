"""
Script to ingest Bible PDF into ChromaDB vector store.

Usage:
    python scripts/ingest_bible.py path/to/bible.pdf
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.vector_store import get_vector_store


def ingest_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Ingest a PDF file into the vector store."""
    print(f"Loading PDF from: {pdf_path}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Add to vector store
    print("Adding to vector store (this may take a while)...")
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    print("Ingestion complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_bible.py <path_to_bible_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    ingest_pdf(pdf_path)
