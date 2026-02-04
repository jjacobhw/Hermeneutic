import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from app.core.config import settings
from app.services.embeddings import get_embeddings


def get_vector_store() -> Chroma:
    """Get or create ChromaDB vector store."""
    embeddings = get_embeddings()

    client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    return Chroma(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
    )


def similarity_search(query: str, k: int = 5) -> list:
    """Search for similar documents."""
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)
