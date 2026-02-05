"""
Helper script to find chapter boundaries in Bible PDF.

This scans the PDF for verse 1 patterns, which typically indicate
new chapters, helping you build a page mapping config.

Usage:
    python scripts/find_chapter_pages.py data/english-standard-version-Bible.pdf
"""

import sys
import re
from langchain_community.document_loaders import PyPDFLoader


def find_chapter_boundaries(pdf_path: str, start_page: int = 40, end_page: int = None):
    """
    Scan PDF for verse 1 occurrences (likely chapter starts).

    Prints a list of pages where verse 1 appears, which you can use
    to build your page mapping configuration.
    """
    # Use LangChain's PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    total_pages = len(documents)

    if end_page is None or end_page > total_pages:
        end_page = total_pages

    print(f"Scanning pages {start_page} to {end_page} for chapter boundaries...")
    print(f"(Looking for verse 1 patterns)\n")

    chapter_pages = []

    for page_num in range(start_page, end_page):
        if page_num % 100 == 0:
            print(f"  Progress: {page_num}/{end_page}")

        text = documents[page_num].page_content

        # Look for chapter start pattern: chapter number + ":1"
        # Example: "3\n:1\nNow the serpent..."
        lines = text.split('\n')

        chapter_match = None
        for i, line in enumerate(lines):
            # Look for a line with just ":1" which indicates verse 1
            if line.strip() == ':1':
                # Previous line should be chapter number
                if i > 0:
                    prev_line = lines[i-1].strip()
                    if prev_line.isdigit():
                        chapter_num = prev_line
                        # Get context
                        start = max(0, i - 3)
                        end = min(len(lines), i + 5)
                        context = lines[start:end]
                        chapter_match = {
                            'chapter_num': chapter_num,
                            'context': context
                        }
                        break

        if chapter_match:
            context_str = ' | '.join([l.strip()[:50] for l in chapter_match['context'] if l.strip()])

            chapter_pages.append({
                'page': page_num,
                'context': context_str
            })

    print(f"\n\n{'='*80}")
    print(f"Found {len(chapter_pages)} potential chapter starts:")
    print(f"{'='*80}\n")

    for item in chapter_pages:
        print(f"Page {item['page']:4d}: {item['context']}")

    print(f"\n\n{'='*80}")
    print("Next steps:")
    print("1. Review this list and identify which book each page belongs to")
    print("2. Create data/page_mapping.json with structure:")
    print("""
{
  "Genesis": {"start_page": 45, "chapters": 50},
  "Exodus": {"start_page": 78, "chapters": 40},
  ...
}
""")
    print("3. Run: python scripts/preprocess_esv_pdf.py --use-mapping")
    print(f"{'='*80}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/find_chapter_pages.py <pdf_path> [start_page] [end_page]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    start_page = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    end_page = int(sys.argv[3]) if len(sys.argv) > 3 else None

    find_chapter_boundaries(pdf_path, start_page, end_page)


if __name__ == "__main__":
    main()
