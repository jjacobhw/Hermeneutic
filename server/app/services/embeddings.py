from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
