# Multi-Source RAG System Guide

This guide explains how to use the multi-source document system to enhance your Bible study RAG application with additional knowledge bases like commentaries, study notes, and theological texts.

## Overview

The multi-source RAG system allows you to:
- Store different document types in separate ChromaDB collections
- Query across multiple collections simultaneously
- Combine Bible verses with commentaries and other resources
- Provide richer, more contextual responses

## Architecture

```
┌─────────────────────────┐
│    User Question        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│   Multi-Source RAG Service          │
├─────────────────────────────────────┤
│ Query multiple collections:         │
│  ├─ Bible verses                   │
│  ├─ Commentary                      │
│  ├─ Study Notes                     │
│  ├─ Theological Texts              │
│  └─ Historical Context             │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│   Combined Response     │
│ - Bible references      │
│ - Commentary insights   │
│ - Study note additions  │
└─────────────────────────┘
```

## Available Collections

| Collection | Purpose | Example Documents |
|-----------|---------|------------------|
| `bible` | Bible verses | Genesis 1:1-31, John 3:16 |
| `commentary` | Biblical commentaries | Matthew Henry, John Gill |
| `study_notes` | Study notes & devotionals | Daily readings, sermon notes |
| `theological` | Theological texts | Systematic theology, doctrines |
| `historical` | Historical context | Ancient Near East, Roman Empire |

## Getting Started

### 1. Add Commentary Documents

**From PDF:**
```bash
python scripts/ingest_additional_documents.py \
  --collection commentary \
  --input data/commentaries/matthew_henry.pdf \
  --format pdf \
  --chunk-size 1000
```

**From directory of PDFs:**
```bash
python scripts/ingest_additional_documents.py \
  --collection commentary \
  --input data/commentaries/ \
  --format directory \
  --pattern "**/*.pdf"
```

### 2. Add Study Notes

**From text files:**
```bash
python scripts/ingest_additional_documents.py \
  --collection study_notes \
  --input data/notes/ \
  --format txt
```

### 3. Add Theological Texts

**From JSON:**
```bash
python scripts/ingest_additional_documents.py \
  --collection theological \
  --input data/theology.json \
  --format json
```

**JSON Format:**
```json
[
  {
    "content": "Justification by faith is the doctrine that...",
    "metadata": {
      "title": "Justification",
      "author": "John Calvin",
      "source": "Institutes"
    }
  }
]
```

## Using the API

### Query Single Collection (Bible Only)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the Bible say about love?",
    "num_passages": 5
  }'
```

**Response:**
```json
{
  "answer": "...",
  "sources": [
    {
      "content": "For God so loved the world...",
      "metadata": {"book": "John", "chapter": 3, "verse": 16}
    }
  ]
}
```

### Query Multiple Collections

```bash
curl -X POST http://localhost:8000/query/multi-source \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the doctrine of justification by faith",
    "collections": ["bible", "commentary", "theological"],
    "k_per_collection": 3
  }'
```

**Response:**
```json
{
  "answer": "Justification by faith is a central doctrine...",
  "sources": {
    "bible": [
      {
        "content": "For by grace you have been saved through faith...",
        "metadata": {"book": "Ephesians", "chapter": 2, "verse": 8}
      }
    ],
    "commentary": [
      {
        "content": "The Apostle Paul emphasizes that justification...",
        "metadata": {"title": "Matthew Henry Commentary", "book": "Romans"}
      }
    ],
    "theological": [
      {
        "content": "Calvin's teaching on justification...",
        "metadata": {"author": "John Calvin", "work": "Institutes"}
      }
    ]
  },
  "collections_searched": ["bible", "commentary", "theological"],
  "total_sources": 9
}
```

### List Available Collections

```bash
curl http://localhost:8000/collections
```

**Response:**
```json
{
  "collections": [
    {
      "name": "bible",
      "document_count": 31102
    },
    {
      "name": "commentary",
      "document_count": 1456
    },
    {
      "name": "study_notes",
      "document_count": 234
    }
  ]
}
```

## Document Preparation

### Recommended Sources

**Commentaries:**
- Matthew Henry Commentary
- John Gill's Exposition
- Adam Clarke Commentary
- Barnes' Notes on the Bible

**Study Resources:**
- Study Bible notes
- Bible dictionaries
- Holman Bible Dictionary

**Theological Texts:**
- Systematic Theology (Berkhof, Grudem, Hodge)
- Calvin's Institutes
- Westminster Confession

**Historical Context:**
- Ancient Near Eastern Texts
- Josephus' Antiquities
- Historical atlases

### Where to Find Documents

1. **Public Domain Commentaries:**
   - Archive.org
   - Project Gutenberg
   - Christian Classics Ethereal Library (CCEL)

2. **Study Bibles:**
   - Purchase or borrow physical copies
   - Scan and OCR text

3. **Theological Works:**
   - Public domain works (pre-1928)
   - CCEL resources
   - Monergism.com

### Document Format Guidelines

**PDF Files:**
- Ensure good OCR quality
- Remove unnecessary formatting
- Split large files if needed

**Text Files:**
- UTF-8 encoding
- Clear section headers
- Include metadata in filename

**JSON Files:**
```json
[
  {
    "content": "Main text content here",
    "metadata": {
      "title": "Document title",
      "author": "Author name",
      "source": "Source book/publication",
      "book": "Bible book (if applicable)",
      "chapter": 1,
      "topic": "Main topic",
      "date": "Publication date"
    }
  }
]
```

## Best Practices

### 1. Chunking Strategy

**For Commentaries:**
- Chunk size: 1000-1500 characters
- Overlap: 200-300 characters
- Preserve verse/section boundaries

**For Theological Texts:**
- Chunk size: 1500-2000 characters
- Overlap: 300 characters
- Keep complete thoughts together

**For Study Notes:**
- Chunk size: 500-1000 characters
- Smaller chunks for quick references

### 2. Metadata Organization

Include these metadata fields:
- `title`: Document/section title
- `author`: Author name
- `source`: Original publication
- `book`: Bible book (if commentary)
- `chapter`: Chapter number (if applicable)
- `topic`: Main topic/theme
- `date`: Publication date

### 3. Query Strategy

**For General Questions:**
```python
collections = ["bible", "commentary"]
```

**For Doctrinal Questions:**
```python
collections = ["bible", "theological", "commentary"]
```

**For Historical Context:**
```python
collections = ["bible", "historical", "commentary"]
```

**For Personal Study:**
```python
collections = ["bible", "study_notes"]
```

### 4. Performance Tips

- Start with smaller document sets
- Monitor collection sizes (check `/collections`)
- Use appropriate `k_per_collection` values (3-5 recommended)
- Consider creating focused collections for specific topics

## Examples

### Example 1: Understanding a Parable

**Query:**
```json
{
  "question": "Explain the parable of the Prodigal Son",
  "collections": ["bible", "commentary", "study_notes"],
  "k_per_collection": 3
}
```

**Result:**
- Bible: Luke 15 passage
- Commentary: Theological interpretation
- Study notes: Practical application

### Example 2: Doctrinal Study

**Query:**
```json
{
  "question": "What is the Trinity?",
  "collections": ["bible", "theological", "commentary"],
  "k_per_collection": 4
}
```

**Result:**
- Bible: Relevant verses (Matt 28:19, 2 Cor 13:14)
- Theological: Formal doctrine explanation
- Commentary: Historical development

### Example 3: Historical Context

**Query:**
```json
{
  "question": "What was life like in Corinth during Paul's time?",
  "collections": ["bible", "historical", "commentary"],
  "k_per_collection": 3
}
```

**Result:**
- Bible: Acts passages about Corinth
- Historical: Ancient Corinth descriptions
- Commentary: Cultural context of letters

## Troubleshooting

### No Results from Collection

**Problem:** Collection returns empty results

**Solutions:**
1. Check if documents were ingested:
   ```bash
   curl http://localhost:8000/collections
   ```
2. Verify collection name is correct
3. Try different search query

### Slow Query Performance

**Problem:** Multi-source queries are slow

**Solutions:**
1. Reduce `k_per_collection` (try 2-3)
2. Query fewer collections
3. Use single-source endpoint for simple queries
4. Ensure ChromaDB is using persistent storage

### Poor Quality Results

**Problem:** Retrieved documents aren't relevant

**Solutions:**
1. Improve chunking (smaller chunks, more overlap)
2. Add more metadata to documents
3. Pre-process documents (remove footers, headers)
4. Use better quality source documents

## Advanced Usage

### Creating Custom Collections

Edit `app/services/multi_collection_store.py`:

```python
COLLECTIONS = {
    # ... existing collections ...
    "sermons": "sermon_collection",
    "devotionals": "daily_devotionals",
}
```

Then ingest:
```bash
python scripts/ingest_additional_documents.py \
  --collection sermons \
  --input data/sermons/
```

### Programmatic Access

```python
from app.services.multi_source_rag import query_multi_source

result = await query_multi_source(
    question="What is faith?",
    collections=["bible", "commentary"],
    k_per_collection=3
)

print(result["answer"])
for collection, sources in result["sources"].items():
    print(f"\n{collection}:")
    for source in sources:
        print(f"  - {source['metadata'].get('reference', 'N/A')}")
```

## Future Enhancements

- [ ] Hybrid search (keyword + semantic)
- [ ] Reranking for better relevance
- [ ] Collection-specific weights
- [ ] Caching for common queries
- [ ] Cross-reference detection
- [ ] Automatic topic extraction

## Contributing

To add new document sources:

1. Prepare documents (PDF, TXT, JSON)
2. Choose appropriate collection
3. Ingest using the script
4. Test queries
5. Document your sources

## License Considerations

When adding documents:
- Ensure you have rights to use them
- Prefer public domain sources
- Respect copyright for modern works
- Consider fair use guidelines
- Attribute sources properly

## Support

For issues or questions:
- Check this guide
- Review example queries
- Check collection counts
- Verify document format
