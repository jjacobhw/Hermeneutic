from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag import query_bible

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    num_passages: int = 5


class Source(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the Bible with a question."""
    try:
        result = await query_bible(request.question, request.num_passages)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
