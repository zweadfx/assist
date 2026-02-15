from typing import Any, Dict, List, Optional

import chromadb

from src.core.config import settings
from src.core.constants import DRILLS_COLLECTION_NAME
from src.services.rag.utils import format_drill_document


class ChromaDBManager:
    """Manages interactions with the ChromaDB vector store."""

    def __init__(self) -> None:
        """Initializes the ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=DRILLS_COLLECTION_NAME
        )

    def add_drills(
        self, drills: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """
        Adds drill documents and their embeddings to the ChromaDB collection.

        Args:
            drills: A list of drill documents (dictionaries).
            embeddings: A list of corresponding embedding vectors.

        Raises:
            ValueError: If the number of drills and embeddings do not match.
        """
        if len(drills) != len(embeddings):
            raise ValueError(
                f"The number of drills ({len(drills)}) must match the number of "
                f"embeddings ({len(embeddings)})."
            )

        if not drills:
            return

        ids = [drill["id"] for drill in drills]
        documents = [format_drill_document(drill) for drill in drills]

        # Prepare metadata, ensuring all values are simple types for ChromaDB.
        metadatas = []
        for drill in drills:
            metadata = {
                "name": drill["name"],
                "category": drill["category"],
                "difficulty": drill["difficulty"],
                "phase": drill["phase"],
                # Join list into a comma-separated string for metadata compatibility
                "required_equipment": ",".join(drill.get("required_equipment", [])),
            }
            metadatas.append(metadata)

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query_drills(
        self,
        query_texts: List[str],
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        """
        Queries the drills collection for relevant documents.

        Args:
            query_texts: A list of query texts to search for.
            n_results: The number of results to return per query.
            where: An optional dictionary for metadata filtering.

        Returns:
            A dictionary containing the query results.
        """
        results = self.collection.query(
            query_texts=query_texts, n_results=n_results, where=where
        )
        return results


# Create a single instance for the application to use.
chroma_manager = ChromaDBManager()
