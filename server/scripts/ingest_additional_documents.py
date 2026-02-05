"""
Ingest Additional Documents into ChromaDB Collections.

This script ingests various document types (commentaries, study notes,
theological texts) into separate ChromaDB collections for multi-source RAG.

Supported formats:
- PDF files
- Text files (.txt)
- Markdown files (.md)
- JSON files (structured data)

Usage:
    # Ingest commentary PDFs
    python scripts/ingest_additional_documents.py --collection commentary --input data/commentaries/

    # Ingest study notes from text files
    python scripts/ingest_additional_documents.py --collection study_notes --input data/notes/ --format txt

    # Ingest from JSON
    python scripts/ingest_additional_documents.py --collection theological --input data/theology.json --format json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.multi_collection_store import multi_store


def load_pdf(file_path: str) -> List[Document]:
    """Load documents from PDF file."""
    print(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages")
    return documents


def load_text(file_path: str) -> List[Document]:
    """Load documents from text file."""
    print(f"Loading text file: {file_path}")
    loader = TextLoader(file_path)
    documents = loader.load()
    print(f"  Loaded {len(documents)} documents")
    return documents


def load_directory(dir_path: str, file_pattern: str = "**/*.pdf") -> List[Document]:
    """Load all documents from directory."""
    print(f"Loading directory: {dir_path}")
    print(f"  Pattern: {file_pattern}")

    # Determine loader based on pattern
    if "*.pdf" in file_pattern:
        loader = DirectoryLoader(
            dir_path,
            glob=file_pattern,
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
    elif "*.txt" in file_pattern:
        loader = DirectoryLoader(
            dir_path,
            glob=file_pattern,
            loader_cls=TextLoader,
            show_progress=True,
        )
    else:
        print(f"Unsupported file pattern: {file_pattern}")
        return []

    documents = loader.load()
    print(f"  Loaded {len(documents)} documents")
    return documents


def load_json(file_path: str) -> List[Document]:
    """
    Load documents from JSON file.

    Expected format:
    [
        {
            "content": "text content",
            "metadata": {"key": "value"}
        },
        ...
    ]
    """
    print(f"Loading JSON: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                content = item.get("content", item.get("text", ""))
                metadata = item.get("metadata", {})

                if content:
                    doc = Document(
                        page_content=content,
                        metadata=metadata,
                    )
                    documents.append(doc)

    print(f"  Loaded {len(documents)} documents")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into chunks for better retrieval.

    Args:
        documents: List of documents to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """
    print(f"\nChunking documents...")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Chunk overlap: {chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunked = splitter.split_documents(documents)
    print(f"  Created {len(chunked)} chunks")

    return chunked


def ingest_to_collection(
    collection_name: str,
    documents: List[Document],
) -> None:
    """
    Ingest documents into a ChromaDB collection.

    Args:
        collection_name: Name of the collection
        documents: List of documents to ingest
    """
    print(f"\nIngesting into collection: {collection_name}")

    try:
        multi_store.add_documents(collection_name, documents)
        print(f"  Successfully ingested {len(documents)} documents")

        # Verify
        count = multi_store.get_collection_count(collection_name)
        print(f"  Total documents in collection: {count}")

    except Exception as e:
        print(f"  Error ingesting documents: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Ingest additional documents into ChromaDB collections"
    )
    parser.add_argument(
        "--collection",
        required=True,
        choices=["bible", "commentary", "study_notes", "theological", "historical"],
        help="Target collection name",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file or directory path",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "txt", "md", "json", "directory"],
        default="pdf",
        help="Input format (default: pdf)",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.pdf",
        help="File pattern for directory mode (default: **/*.pdf)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for text splitting (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap (default: 200)",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Skip chunking (use for pre-chunked data)",
    )

    args = parser.parse_args()

    print("="*80)
    print("DOCUMENT INGESTION")
    print("="*80)
    print(f"Collection: {args.collection}")
    print(f"Input: {args.input}")
    print(f"Format: {args.format}")
    print("="*80)

    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\nError: Input path does not exist: {args.input}")
        sys.exit(1)

    # Load documents based on format
    documents = []

    if args.format == "pdf":
        if input_path.is_file():
            documents = load_pdf(str(input_path))
        else:
            documents = load_directory(str(input_path), "**/*.pdf")

    elif args.format == "txt" or args.format == "md":
        if input_path.is_file():
            documents = load_text(str(input_path))
        else:
            pattern = f"**/*.{args.format}"
            documents = load_directory(str(input_path), pattern)

    elif args.format == "json":
        documents = load_json(str(input_path))

    elif args.format == "directory":
        documents = load_directory(str(input_path), args.pattern)

    if not documents:
        print("\nNo documents loaded. Exiting.")
        sys.exit(1)

    # Chunk documents (unless disabled)
    if not args.no_chunk:
        documents = chunk_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    # Ingest into collection
    ingest_to_collection(args.collection, documents)

    print("\n" + "="*80)
    print("INGESTION COMPLETE")
    print("="*80)
    print(f"\nYou can now query the '{args.collection}' collection")
    print(f"using the multi-source RAG endpoints.")


if __name__ == "__main__":
    main()
