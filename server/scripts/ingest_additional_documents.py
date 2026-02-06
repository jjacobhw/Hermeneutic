"""
Ingest Additional Documents into ChromaDB Collections with Advanced Preprocessing.

This script ingests various document types (commentaries, study notes,
theological texts) into separate ChromaDB collections for multi-source RAG.

Features:
- Intelligent text cleaning and normalization
- Metadata enrichment for better retrieval
- Semantic structure detection (principles, methods, instructions)
- Context-aware chunking that preserves meaning
- Document classification and tagging

Supported formats:
- PDF files
- Text files (.txt)
- Markdown files (.md)
- JSON files (structured data)

Usage:
    # Ingest commentary PDFs with preprocessing
    python scripts/ingest_additional_documents.py --collection commentary --input data/Commentary/

    # Ingest with custom preprocessing
    python scripts/ingest_additional_documents.py --collection commentary --input data/Commentary/ --preprocess

    # Ingest study notes from text files
    python scripts/ingest_additional_documents.py --collection study_notes --input data/notes/ --format txt
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import Counter

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


# ============================================================================
# PREPROCESSING FUNCTIONS FOR PROMPT ENGINEERING & RAG OPTIMIZATION
# ============================================================================


def clean_text(text: str) -> str:
    """
    Clean and normalize text extracted from PDFs.

    - Removes excessive whitespace
    - Normalizes line breaks
    - Fixes common PDF extraction artifacts
    - Preserves intentional formatting (lists, paragraphs)
    """
    # Remove null bytes and other control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # Fix common PDF artifacts
    text = text.replace('\uf0b7', '•')  # Bullet points
    text = text.replace('\u2022', '•')
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '—')  # Em dash
    text = text.replace('\u201c', '"')  # Smart quotes
    text = text.replace('\u201d', '"')
    text = text.replace('\u2018', "'")
    text = text.replace('\u2019', "'")

    # Normalize whitespace while preserving structure
    # Keep single line breaks, collapse multiple
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Remove trailing/leading whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Collapse multiple spaces (but not at line start for indentation)
    text = re.sub(r'(?<=\S)  +', ' ', text)

    return text.strip()


def extract_document_metadata(text: str, file_path: str) -> Dict[str, any]:
    """
    Extract rich metadata from document content for better RAG retrieval.

    Returns metadata including:
    - Document type (principle, method, guide, study)
    - Subject matter (hermeneutics, Bible study, theology)
    - Key topics and themes
    - Intended use (instruction, reference, study guide)
    """
    filename = Path(file_path).stem
    metadata = {
        "source": file_path,
        "filename": filename,
    }

    # Classify document type based on filename and content
    doc_types = []
    if any(word in filename.lower() for word in ['principle', 'hermeneutic']):
        doc_types.append('principles')
    if any(word in filename.lower() for word in ['tips', 'guide', 'study']):
        doc_types.append('study_guide')
    if any(word in filename.lower() for word in ['method', 'inductive']):
        doc_types.append('methodology')
    if any(word in filename.lower() for word in ['theology', 'covenant', 'dispensation']):
        doc_types.append('theological_framework')

    metadata['document_type'] = doc_types if doc_types else ['general']

    # Extract subject matter
    subjects = []
    content_lower = text.lower()

    if any(word in content_lower for word in ['hermeneutic', 'interpret', 'exegesis']):
        subjects.append('hermeneutics')
    if any(word in content_lower for word in ['inductive', 'observation', 'meditation']):
        subjects.append('bible_study_methods')
    if any(word in content_lower for word in ['covenant', 'dispensation', 'theology']):
        subjects.append('systematic_theology')
    if any(word in content_lower for word in ['proverb', 'psalm', 'revelation', 'gospel']):
        subjects.append('biblical_books')

    metadata['subjects'] = subjects if subjects else ['general']

    # Identify key topics (words that appear frequently and are meaningful)
    # Extract capitalized phrases and important theological terms
    theological_terms = re.findall(
        r'\b(?:Scripture|Biblical|Gospel|Covenant|Testament|Theological?|Hermeneutic|'
        r'Exegesis|Doctrine|Principle|Author|Context|Interpretation)\b',
        text,
        re.IGNORECASE
    )

    # Count frequency
    term_counts = Counter([term.lower() for term in theological_terms])
    top_terms = [term for term, count in term_counts.most_common(5) if count > 1]
    metadata['key_topics'] = top_terms

    # Determine intended use
    if 'task:' in content_lower or 'step' in content_lower:
        metadata['intended_use'] = 'instructional'
    elif 'principle' in content_lower:
        metadata['intended_use'] = 'reference'
    else:
        metadata['intended_use'] = 'study'

    return metadata


def detect_content_structure(text: str) -> Dict[str, List[str]]:
    """
    Detect semantic structure within the document.

    Identifies:
    - Principles and rules
    - Instructions and methods
    - Examples and applications
    - Questions and prompts
    """
    structure = {
        'principles': [],
        'instructions': [],
        'examples': [],
        'questions': [],
        'definitions': []
    }

    lines = text.split('\n')

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Detect principles (numbered statements, definitive rules)
        if re.match(r'^\d+\..*(?:principle|rule|goal|responsibility)', line_stripped, re.IGNORECASE):
            structure['principles'].append(line_stripped)

        # Detect instructions (imperative verbs, task markers)
        if any(marker in line_stripped.lower() for marker in ['task:', 'step', 'how to', 'seek to']):
            structure['instructions'].append(line_stripped)

        # Detect examples
        if re.match(r'^(?:example|e\.g\.|for instance)', line_stripped, re.IGNORECASE):
            structure['examples'].append(line_stripped)

        # Detect questions
        if '?' in line_stripped and len(line_stripped) < 200:
            structure['questions'].append(line_stripped)

        # Detect definitions
        if ':' in line_stripped and len(line_stripped.split(':')[0]) < 50:
            structure['definitions'].append(line_stripped)

    return structure


def add_contextual_tags(document: Document) -> Document:
    """
    Add contextual tags to documents for enhanced RAG retrieval.

    Tags help the LLM understand how to use the information:
    - Content type (principle, instruction, example)
    - Authority level (fundamental, supplementary)
    - Application context (when to apply this knowledge)
    """
    text = document.page_content
    metadata = document.metadata

    # Tag content based on patterns
    tags = set()

    # Content type tags
    if re.search(r'(?:principle|fundamental|rule|must|should)', text, re.IGNORECASE):
        tags.add('foundational_principle')

    if re.search(r'(?:task|step|how to|method|process)', text, re.IGNORECASE):
        tags.add('practical_instruction')

    if re.search(r'(?:example|for instance|such as|like)', text, re.IGNORECASE):
        tags.add('illustrative_example')

    # Authority level
    if 'scripture' in text.lower() or 'biblical' in text.lower():
        tags.add('scripture_based')

    # Application context
    if any(word in text.lower() for word in ['interpret', 'understand', 'read']):
        tags.add('interpretation_guidance')

    if 'context' in text.lower():
        tags.add('contextual_analysis')

    metadata['content_tags'] = list(tags)

    # Add a summary prefix for better retrieval
    doc_type = metadata.get('document_type', ['general'])[0]
    summary_prefix = f"[{doc_type.upper()}] "

    return Document(
        page_content=summary_prefix + text,
        metadata=metadata
    )


def serialize_metadata(metadata: Dict) -> Dict:
    """
    Serialize complex metadata for ChromaDB compatibility.

    ChromaDB only accepts str, int, float, bool values.
    This function converts lists and dicts to strings.
    """
    serialized = {}

    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            serialized[key] = value
        elif isinstance(value, list):
            # Convert list to comma-separated string
            serialized[key] = ', '.join(str(v) for v in value)
        elif isinstance(value, dict):
            # Convert dict to JSON string
            serialized[key] = json.dumps(value)
        else:
            # Convert anything else to string
            serialized[key] = str(value)

    return serialized


def preprocess_document(document: Document, file_path: str = None) -> Document:
    """
    Complete preprocessing pipeline for a single document.

    Steps:
    1. Clean text (normalize formatting)
    2. Extract metadata (classify and tag)
    3. Detect structure (identify semantic elements)
    4. Add contextual tags (enhance retrieval)
    5. Serialize metadata for ChromaDB
    """
    # Clean text
    cleaned_text = clean_text(document.page_content)

    # Extract metadata
    source = file_path or document.metadata.get('source', '')
    enriched_metadata = extract_document_metadata(cleaned_text, source)

    # Merge with existing metadata
    enriched_metadata.update(document.metadata)

    # Detect structure
    structure = detect_content_structure(cleaned_text)
    enriched_metadata['structure'] = structure

    # Create updated document
    processed_doc = Document(
        page_content=cleaned_text,
        metadata=enriched_metadata
    )

    # Add contextual tags
    final_doc = add_contextual_tags(processed_doc)

    # Serialize metadata for ChromaDB compatibility
    final_doc.metadata = serialize_metadata(final_doc.metadata)

    return final_doc


def preprocess_documents(documents: List[Document], file_paths: List[str] = None) -> List[Document]:
    """
    Preprocess a batch of documents for optimal RAG performance.
    """
    print(f"\n{'='*80}")
    print("PREPROCESSING DOCUMENTS FOR KNOWLEDGE BASE")
    print(f"{'='*80}")

    processed = []

    for i, doc in enumerate(documents):
        file_path = file_paths[i] if file_paths and i < len(file_paths) else None

        try:
            processed_doc = preprocess_document(doc, file_path)
            processed.append(processed_doc)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(documents)} documents...")

        except Exception as e:
            print(f"  Warning: Failed to preprocess document {i}: {e}")
            processed.append(doc)  # Keep original if preprocessing fails

    print(f"  ✓ Successfully preprocessed {len(processed)} documents")
    print(f"{'='*80}\n")

    return processed


# ============================================================================
# DOCUMENT LOADING FUNCTIONS
# ============================================================================


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
    Split documents into semantic-aware chunks for better retrieval.

    Uses intelligent separators to preserve document structure:
    - Respects paragraph boundaries
    - Keeps numbered lists together
    - Preserves section headers with content

    Args:
        documents: List of documents to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """
    print(f"\n{'='*80}")
    print("CHUNKING DOCUMENTS")
    print(f"{'='*80}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Chunk overlap: {chunk_overlap}")

    # Use semantic separators to preserve structure
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Prioritize semantic boundaries
        separators=[
            "\n\n\n",  # Multiple line breaks (section boundaries)
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentences
            " ",       # Words
            "",        # Characters
        ],
        keep_separator=True,
    )

    chunked = splitter.split_documents(documents)

    # Post-process chunks to add chunk-specific metadata
    for i, chunk in enumerate(chunked):
        chunk.metadata['chunk_index'] = i
        chunk.metadata['total_chunks'] = len(chunked)

        # Identify if chunk contains key content types
        content = chunk.page_content.lower()
        chunk_tags = []

        if any(marker in content for marker in ['principle', 'rule', 'fundamental']):
            chunk_tags.append('contains_principles')
        if any(marker in content for marker in ['task:', 'step', 'how to']):
            chunk_tags.append('contains_instructions')
        if '?' in content:
            chunk_tags.append('contains_questions')

        if chunk_tags:
            # Serialize list to comma-separated string
            chunk.metadata['chunk_content_type'] = ', '.join(chunk_tags)

    print(f"  ✓ Created {len(chunked)} semantic chunks")
    print(f"{'='*80}\n")

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
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=True,
        help="Enable advanced preprocessing (default: True)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_false",
        dest="preprocess",
        help="Disable preprocessing",
    )

    args = parser.parse_args()

    print("="*80)
    print("DOCUMENT INGESTION WITH PREPROCESSING")
    print("="*80)
    print(f"Collection: {args.collection}")
    print(f"Input: {args.input}")
    print(f"Format: {args.format}")
    print(f"Preprocessing: {'Enabled' if args.preprocess else 'Disabled'}")
    print(f"Chunking: {'Enabled' if not args.no_chunk else 'Disabled'}")
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

    # Preprocess documents for prompt engineering (unless disabled)
    if args.preprocess:
        file_paths = [doc.metadata.get('source', '') for doc in documents]
        documents = preprocess_documents(documents, file_paths)

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
