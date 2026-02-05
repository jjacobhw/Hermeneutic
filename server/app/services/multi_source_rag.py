"""
Multi-Source RAG Service.

This service combines retrieval from multiple document collections
(Bible, commentaries, study notes, etc.) for richer contextual responses.
"""

from typing import List, Dict, Optional
from langchain.schema import Document
from app.services.multi_collection_store import multi_store
from app.services.llm import get_completion


async def query_multi_source(
    question: str,
    collections: List[str] = ["bible"],
    k_per_collection: int = 3,
) -> Dict:
    """
    Query multiple document collections for enhanced responses.

    Args:
        question: User question
        collections: List of collections to query (bible, commentary, etc.)
        k_per_collection: Number of results per collection

    Returns:
        Dictionary with answer and sources grouped by collection
    """
    # Retrieve from multiple collections
    results_by_collection = multi_store.search_multi_collection(
        query=question,
        collections=collections,
        k_per_collection=k_per_collection,
    )

    if not results_by_collection:
        return {
            "answer": "No relevant documents found. Please ensure documents have been ingested.",
            "sources": {},
            "collections_searched": collections,
        }

    # Build context from all collections
    context_parts = []
    sources = {}

    # Organize by collection type
    for collection_name, docs in results_by_collection.items():
        sources[collection_name] = []

        # Add section header
        collection_label = collection_name.replace("_", " ").title()
        context_parts.append(f"\n=== {collection_label} ===\n")

        for doc in docs:
            context_parts.append(doc.page_content)

            sources[collection_name].append({
                "content": doc.page_content,
                "metadata": doc.metadata,
            })

    context = "\n\n---\n\n".join(context_parts)

    # Get LLM response with multi-source context
    answer = await get_enhanced_completion(question, context, results_by_collection)

    return {
        "answer": answer,
        "sources": sources,
        "collections_searched": list(results_by_collection.keys()),
        "total_sources": sum(len(docs) for docs in results_by_collection.values()),
    }


async def get_enhanced_completion(
    question: str,
    context: str,
    results_by_collection: Dict[str, List[Document]],
) -> str:
    """
    Get LLM completion with enhanced multi-source system prompt.

    Args:
        question: User question
        context: Combined context from all sources
        results_by_collection: Results organized by collection

    Returns:
        LLM response
    """
    # Determine which types of sources we have
    has_bible = "bible" in results_by_collection
    has_commentary = "commentary" in results_by_collection
    has_notes = "study_notes" in results_by_collection
    has_theological = "theological" in results_by_collection

    # Build dynamic system prompt based on available sources
    prompt_parts = ["You are a knowledgeable Bible study assistant with access to:"]

    if has_bible:
        prompt_parts.append("- Scripture passages from the Bible")
    if has_commentary:
        prompt_parts.append("- Biblical commentaries and scholarly analysis")
    if has_notes:
        prompt_parts.append("- Study notes and devotional materials")
    if has_theological:
        prompt_parts.append("- Theological texts and doctrinal resources")

    prompt_parts.append("\nYour response should:")
    prompt_parts.append("1. Primarily reference Scripture when answering")
    prompt_parts.append("2. Use commentaries and other sources to provide deeper insight")
    prompt_parts.append("3. Clearly distinguish between what Scripture says and what commentaries/scholars say")
    prompt_parts.append("4. Always cite specific verses when referencing the Bible")
    prompt_parts.append("5. Indicate the source type when referencing non-Scripture materials")
    prompt_parts.append("\nIf the context doesn't contain relevant information, say so honestly.")

    system_prompt = "\n".join(prompt_parts)

    # Build user message
    user_message = f"""{context}

Question: {question}

Please provide a comprehensive answer that synthesizes insights from all available sources."""

    # Use existing completion function with enhanced prompts
    import httpx
    from app.core.config import settings

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Hermeneutic Bible App",
    }

    payload = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_URL, json=payload, headers=headers, timeout=60.0
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
