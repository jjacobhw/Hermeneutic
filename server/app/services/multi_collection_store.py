"""
Multi-Collection Vector Store Service.

This service manages multiple ChromaDB collections for different document types
(Bible verses, commentaries, study notes, theological texts, etc.) and provides
unified querying across all collections.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Dict, Optional
from app.core.config import settings
from app.services.embeddings import get_embeddings


class MultiCollectionStore:
    """Manager for multiple document collections in ChromaDB."""

    # Define available collections
    COLLECTIONS = {
        "bible": "bible_esv",
        "commentary": "bible_commentary",
        "study_notes": "study_notes",
        "theological": "theological_texts",
        "historical": "historical_context",
    }

    def __init__(self):
        """Initialize multi-collection store."""
        self.embeddings = get_embeddings()
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def get_collection(self, collection_name: str) -> Chroma:
        """
        Get a specific collection.

        Args:
            collection_name: Name of the collection (bible, commentary, etc.)

        Returns:
            Chroma vector store instance
        """
        collection_id = self.COLLECTIONS.get(collection_name, collection_name)

        return Chroma(
            client=self.client,
            collection_name=collection_id,
            embedding_function=self.embeddings,
        )

    def search_collection(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """
        Search within a specific collection.

        Args:
            collection_name: Name of the collection
            query: Search query
            k: Number of results

        Returns:
            List of matching documents
        """
        try:
            collection = self.get_collection(collection_name)
            results = collection.similarity_search(query, k=k)

            # Add collection info to metadata
            for doc in results:
                doc.metadata["collection"] = collection_name

            return results
        except Exception as e:
            print(f"Error searching collection {collection_name}: {e}")
            return []

    def search_multi_collection(
        self,
        query: str,
        collections: List[str] = None,
        k_per_collection: int = 3,
    ) -> Dict[str, List[Document]]:
        """
        Search across multiple collections.

        Args:
            query: Search query
            collections: List of collection names (default: all)
            k_per_collection: Results per collection

        Returns:
            Dictionary mapping collection names to documents
        """
        if collections is None:
            collections = list(self.COLLECTIONS.keys())

        results = {}

        for collection_name in collections:
            docs = self.search_collection(collection_name, query, k=k_per_collection)
            if docs:
                results[collection_name] = docs

        return results

    def search_all_collections(
        self,
        query: str,
        k_per_collection: int = 3,
    ) -> List[Document]:
        """
        Search all collections and return combined results.

        Args:
            query: Search query
            k_per_collection: Results per collection

        Returns:
            Combined list of documents from all collections
        """
        all_results = self.search_multi_collection(
            query=query,
            collections=None,
            k_per_collection=k_per_collection,
        )

        # Flatten results
        combined = []
        for collection_name, docs in all_results.items():
            combined.extend(docs)

        return combined

    def add_documents(
        self,
        collection_name: str,
        documents: List[Document],
    ) -> None:
        """
        Add documents to a specific collection.

        Args:
            collection_name: Name of the collection
            documents: List of documents to add
        """
        collection = self.get_collection(collection_name)
        collection.add_documents(documents)

    def list_collections(self) -> List[str]:
        """
        List all available collections.

        Returns:
            List of collection names
        """
        return list(self.COLLECTIONS.keys())

    def get_collection_count(self, collection_name: str) -> int:
        """
        Get document count in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of documents in collection
        """
        try:
            collection = self.get_collection(collection_name)
            return collection._collection.count()
        except Exception:
            return 0


# Singleton instance
multi_store = MultiCollectionStore()
