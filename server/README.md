# Hermeneutic Server - Multi-Source RAG Backend

Backend API for the Hermeneutic Bible study application with multi-source RAG capabilities.

## Features

- **Multi-Collection Vector Store**: Store different document types in separate ChromaDB collections
- **Bible Verse Retrieval**: Bible verses optimized for RAG
- **Commentary Integration**: Add biblical commentaries for deeper insights
- **Study Notes**: Include study notes and devotional materials
- **Theological Texts**: Integrate systematic theology and doctrinal resources
- **Unified Querying**: Query across multiple collections simultaneously

## Quick Start

### 1. Install Dependencies

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```bash
OPENROUTER_API_KEY=your_key_here
LLM_MODEL=anthropic/claude-3.5-sonnet
CHROMA_PERSIST_DIR=./chroma_db
```

### 3. Add Documents (Optional)

```bash
# Add commentaries
python scripts/ingest_additional_documents.py \
  --collection commentary \
  --input data/commentaries/ \
  --format pdf

# Add study notes
python scripts/ingest_additional_documents.py \
  --collection study_notes \
  --input data/notes/ \
  --format txt
```

### 4. Start Server

```bash
uvicorn app.main:app --reload
```

Server runs at: `http://localhost:8000`

## API Endpoints

### Single-Source Query (Bible Only)
```bash
POST /query
{
  "question": "What does the Bible say about love?",
  "num_passages": 5
}
```

### Multi-Source Query (Bible + Other Collections)
```bash
POST /query/multi-source
{
  "question": "Explain justification by faith",
  "collections": ["bible", "commentary", "theological"],
  "k_per_collection": 3
}
```

### List Collections
```bash
GET /collections
```

## Available Collections

- `bible` - Bible verses
- `commentary` - Biblical commentaries
- `study_notes` - Study notes and devotionals
- `theological` - Theological texts
- `historical` - Historical context documents

## Documentation

- [Multi-Source RAG Guide](MULTI_SOURCE_RAG_GUIDE.md) - Complete guide for multi-source system
- API Docs: `http://localhost:8000/docs` (when server is running)

## Architecture

```
data/
  ├── commentaries/                  # Commentary PDFs
  ├── notes/                         # Study notes
  └── theology/                      # Theological texts

scripts/
  └── ingest_additional_documents.py # Ingest documents

app/
  ├── services/
  │   ├── vector_store.py           # Single collection store
  │   ├── multi_collection_store.py # Multi-collection manager
  │   ├── rag.py                    # Basic RAG
  │   └── multi_source_rag.py       # Multi-source RAG
  └── api/
      └── routes.py                 # API endpoints
```

## Example Usage

### Python Client
```python
import httpx

async with httpx.AsyncClient() as client:
    # Query multiple sources
    response = await client.post(
        "http://localhost:8000/query/multi-source",
        json={
            "question": "What is grace?",
            "collections": ["bible", "commentary", "theological"],
            "k_per_collection": 3
        }
    )

    result = response.json()
    print(result["answer"])
```

### cURL
```bash
curl -X POST http://localhost:8000/query/multi-source \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the Trinity",
    "collections": ["bible", "theological"],
    "k_per_collection": 4
  }'
```

## Development

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black app/ scripts/
```

### Type Check
```bash
mypy app/
```

## License

[Your License]
