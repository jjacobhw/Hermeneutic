import httpx
from app.core.config import settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def get_completion(prompt: str, context: str) -> str:
    """Get completion from OpenRouter API using Claude."""
    system_prompt = """You are a knowledgeable Bible study assistant.
Use the provided scripture passages to answer questions accurately and thoughtfully.
Always cite the specific verses you reference in your response.
If the context doesn't contain relevant information, say so honestly."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context from Scripture:\n{context}\n\nQuestion: {prompt}",
        },
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
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_URL, json=payload, headers=headers, timeout=60.0
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
