from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from app.services.rag import query_bible
from app.services.multi_source_rag import query_multi_source
from app.services.multi_collection_store import multi_store

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    num_passages: int = 5


class MultiSourceRequest(BaseModel):
    question: str
    collections: List[str] = ["bible"]
    k_per_collection: int = 3


class Source(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class MultiSourceResponse(BaseModel):
    answer: str
    sources: Dict[str, list[Source]]
    collections_searched: List[str]
    total_sources: int


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the Bible with a question (single source)."""
    try:
        result = await query_bible(request.question, request.num_passages)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/multi-source", response_model=MultiSourceResponse)
async def query_multi(request: MultiSourceRequest):
    """
    Query multiple document collections for enhanced responses.

    This endpoint combines Bible verses with other document collections
    like commentaries, study notes, and theological texts for richer context.

    Available collections:
    - bible: ESV Bible verses
    - commentary: Biblical commentaries
    - study_notes: Study notes and devotional materials
    - theological: Theological texts and doctrinal resources
    - historical: Historical context documents
    """
    try:
        result = await query_multi_source(
            question=request.question,
            collections=request.collections,
            k_per_collection=request.k_per_collection,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections")
async def list_collections():
    """List all available document collections."""
    collections = multi_store.list_collections()
    collection_info = []

    for collection_name in collections:
        count = multi_store.get_collection_count(collection_name)
        collection_info.append({
            "name": collection_name,
            "document_count": count,
        })

    return {"collections": collection_info}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
