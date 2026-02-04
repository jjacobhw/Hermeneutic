from app.services.vector_store import similarity_search
from app.services.llm import get_completion


async def query_bible(question: str, num_passages: int = 5) -> dict:
    """Query the Bible using RAG."""
    # Retrieve relevant passages
    docs = similarity_search(question, k=num_passages)

    if not docs:
        return {
            "answer": "No relevant passages found. Please ensure the Bible has been ingested.",
            "sources": [],
        }

    # Build context from retrieved documents
    context_parts = []
    sources = []
    for doc in docs:
        context_parts.append(doc.page_content)
        sources.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    # Get LLM response
    answer = await get_completion(question, context)

    return {
        "answer": answer,
        "sources": sources,
    }
