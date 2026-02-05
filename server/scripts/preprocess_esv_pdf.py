"""
Preprocess ESV Bible PDF into structured verse format.

This script extracts individual verses with metadata from the ESV PDF,
creating structured JSON optimized for RAG ingestion.

Usage:
    python scripts/preprocess_esv_pdf.py data/english-standard-version-Bible.pdf --output data/bible_structured.json
"""

import sys
import json
import re
import os
from pathlib import Path
import argparse
from langchain_community.document_loaders import PyPDFLoader

# Bible book names and patterns
BIBLE_BOOKS = {
    # Old Testament
    "Genesis": ("OT", 50),
    "Exodus": ("OT", 40),
    "Leviticus": ("OT", 27),
    "Numbers": ("OT", 36),
    "Deuteronomy": ("OT", 34),
    "Joshua": ("OT", 24),
    "Judges": ("OT", 21),
    "Ruth": ("OT", 4),
    "1 Samuel": ("OT", 31),
    "2 Samuel": ("OT", 24),
    "1 Kings": ("OT", 22),
    "2 Kings": ("OT", 25),
    "1 Chronicles": ("OT", 29),
    "2 Chronicles": ("OT", 36),
    "Ezra": ("OT", 10),
    "Nehemiah": ("OT", 13),
    "Esther": ("OT", 10),
    "Job": ("OT", 42),
    "Psalm": ("OT", 150),
    "Proverbs": ("OT", 31),
    "Ecclesiastes": ("OT", 12),
    "Song of Solomon": ("OT", 8),
    "Isaiah": ("OT", 66),
    "Jeremiah": ("OT", 52),
    "Lamentations": ("OT", 5),
    "Ezekiel": ("OT", 48),
    "Daniel": ("OT", 12),
    "Hosea": ("OT", 14),
    "Joel": ("OT", 3),
    "Amos": ("OT", 9),
    "Obadiah": ("OT", 1),
    "Jonah": ("OT", 4),
    "Micah": ("OT", 7),
    "Nahum": ("OT", 3),
    "Habakkuk": ("OT", 3),
    "Zephaniah": ("OT", 3),
    "Haggai": ("OT", 2),
    "Zechariah": ("OT", 14),
    "Malachi": ("OT", 4),
    # New Testament
    "Matthew": ("NT", 28),
    "Mark": ("NT", 16),
    "Luke": ("NT", 24),
    "John": ("NT", 21),
    "Acts": ("NT", 28),
    "Romans": ("NT", 16),
    "1 Corinthians": ("NT", 16),
    "2 Corinthians": ("NT", 13),
    "Galatians": ("NT", 6),
    "Ephesians": ("NT", 6),
    "Philippians": ("NT", 4),
    "Colossians": ("NT", 4),
    "1 Thessalonians": ("NT", 5),
    "2 Thessalonians": ("NT", 3),
    "1 Timothy": ("NT", 6),
    "2 Timothy": ("NT", 4),
    "Titus": ("NT", 3),
    "Philemon": ("NT", 1),
    "Hebrews": ("NT", 13),
    "James": ("NT", 5),
    "1 Peter": ("NT", 5),
    "2 Peter": ("NT", 3),
    "1 John": ("NT", 5),
    "2 John": ("NT", 1),
    "3 John": ("NT", 1),
    "Jude": ("NT", 1),
    "Revelation": ("NT", 22),
}


def extract_verses_from_pdf(pdf_path: str, start_page: int = 40, end_page: int = None) -> list[dict]:
    """
    Extract structured verses from ESV PDF using LangChain's PyPDFLoader.

    Args:
        pdf_path: Path to the ESV PDF
        start_page: Page to start extraction (default 40, after preface)
        end_page: Page to end extraction (default None = all pages)

    Returns:
        List of verse dictionaries with structure:
        {
            "book": "Genesis",
            "chapter": 1,
            "verse": 1,
            "text": "In the beginning...",
            "reference": "Genesis 1:1",
            "testament": "OT"
        }
    """
    print(f"Loading PDF from: {pdf_path}")

    # Use LangChain's PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    total_pages = len(documents)

    if end_page is None:
        end_page = total_pages

    print(f"Processing pages {start_page} to {end_page} (of {total_pages} total)")

    verses = []
    current_book = None
    current_chapter = None

    # Process documents (pages) within the specified range
    for doc in documents[start_page:min(end_page, total_pages)]:
        page_num = doc.metadata.get('page', 0)

        if page_num % 100 == 0:
            print(f"  Processing page {page_num}/{end_page}...")

        text = doc.page_content

        # Try to detect book/chapter headers
        book_match, chapter_match = _detect_book_chapter(text)
        if book_match:
            current_book = book_match
        if chapter_match:
            current_chapter = chapter_match

        # Extract verses from this page
        page_verses = _parse_verses_from_text(
            text,
            current_book,
            current_chapter
        )

        verses.extend(page_verses)

    print(f"Extracted {len(verses)} verses")
    return verses


def _detect_book_chapter(text: str) -> tuple[str, int]:
    """
    Detect book name and chapter number from page text.
    Returns (book_name, chapter_num) or (None, None)
    """
    # Look for book names at start of page
    for book_name in BIBLE_BOOKS.keys():
        if book_name.upper() in text[:200]:  # Check first 200 chars
            # Look for chapter number pattern like "Genesis 1" or "1"
            chapter_pattern = rf"{book_name}\s+(\d+)"
            match = re.search(chapter_pattern, text[:300])
            if match:
                return (book_name, int(match.group(1)))

            # Sometimes just chapter number alone
            chapter_match = re.search(r"^\s*(\d+)\s*$", text[:100], re.MULTILINE)
            if chapter_match:
                return (book_name, int(chapter_match.group(1)))

            return (book_name, None)

    return (None, None)


def _parse_verses_from_text(text: str, book: str, chapter: int) -> list[dict]:
    """
    Parse individual verses from page text.

    ESV format: Verse numbers appear as standalone digits like:
    8
    And they heard the sound of the LORD...
    9
    But the LORD God called...
    """
    if not book or not chapter:
        return []

    verses = []

    # Remove cross-reference markers [20], [21], etc.
    text_clean = re.sub(r'\[\d+\]', '', text)

    # Split by verse numbers that appear on their own line
    # Pattern: newline + verse number + newline
    # This splits text into chunks starting with verse numbers
    parts = re.split(r'\n(\d+)\n', text_clean)

    # parts[0] is text before first verse
    # parts[1] is verse number, parts[2] is verse text
    # parts[3] is verse number, parts[4] is verse text, etc.

    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break

        verse_num_str = parts[i].strip()
        verse_text = parts[i + 1].strip()

        try:
            verse_num = int(verse_num_str)
        except ValueError:
            continue

        # Skip if verse number seems unrealistic
        if verse_num > 200 or verse_num < 1:
            continue

        # Clean up the text
        verse_text = re.sub(r'\s+', ' ', verse_text)  # Normalize whitespace

        # Skip if text is too short (likely not a real verse)
        if len(verse_text) < 10:
            continue

        # Limit length
        verse_text = verse_text[:1000]

        testament = BIBLE_BOOKS.get(book, ("OT",))[0]

        verses.append({
            "book": book,
            "chapter": chapter,
            "verse": verse_num,
            "text": verse_text,
            "reference": f"{book} {chapter}:{verse_num}",
            "testament": testament,
            "translation": "ESV"
        })

    return verses


def validate_and_dedupe(verses: list[dict]) -> list[dict]:
    """Remove duplicates and validate verses."""
    seen = set()
    unique_verses = []

    for verse in verses:
        ref = verse['reference']
        if ref not in seen:
            seen.add(ref)
            unique_verses.append(verse)

    print(f"Removed {len(verses) - len(unique_verses)} duplicate verses")
    return unique_verses


def save_to_json(verses: list[dict], output_path: str) -> None:
    """Save preprocessed verses to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(verses, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(verses)} verses to {output_path}")


def print_statistics(verses: list[dict]) -> None:
    """Print validation statistics."""
    if not verses:
        print("⚠️  No verses found!")
        return

    books = {}
    for verse in verses:
        book = verse['book']
        if book not in books:
            books[book] = 0
        books[book] += 1

    print("\n✓ Verses by book:")
    for book, count in sorted(books.items()):
        print(f"  {book}: {count} verses")

    print(f"\n✓ Total verses: {len(verses)}")
    print(f"✓ Sample verses:")
    for i in range(min(3, len(verses))):
        v = verses[i]
        print(f"  {v['reference']}: {v['text'][:60]}...")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ESV Bible PDF")
    parser.add_argument("pdf_path", help="Path to ESV Bible PDF file")
    parser.add_argument(
        "--output",
        default="data/bible_structured.json",
        help="Output JSON file (default: data/bible_structured.json)"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=40,
        help="Start page (default: 40, skip preface)"
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="End page (default: None, process all)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only N pages for testing"
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    # Calculate end page for sample mode
    end_page = args.end_page
    if args.sample:
        end_page = args.start_page + args.sample
        print(f"Sample mode: processing {args.sample} pages")

    # Extract verses
    verses = extract_verses_from_pdf(args.pdf_path, args.start_page, end_page)

    # Deduplicate
    verses = validate_and_dedupe(verses)

    # Validate
    print_statistics(verses)

    # Save
    if verses:
        save_to_json(verses, args.output)
        print(f"\n✓ Preprocessing complete!")
        print(f"  Next: python scripts/ingest_structured.py {args.output}")
    else:
        print("\n⚠️  No verses extracted. Check PDF format and start_page.")
        sys.exit(1)


if __name__ == "__main__":
    main()
