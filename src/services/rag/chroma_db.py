import threading
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.core.config import settings
from src.core.constants import (
    DRILLS_COLLECTION_NAME,
    PLAYERS_COLLECTION_NAME,
    SHOES_COLLECTION_NAME,
)
from src.services.rag.formatters import (
    format_drill_document,
    format_player_document,
    format_shoe_document,
)


class ChromaDBManager:
    """Manages interactions with the ChromaDB vector store with lazy initialization."""

    def __init__(self) -> None:
        """
        Initializes the manager without connecting to ChromaDB.
        Actual initialization is deferred until first use to avoid import-time failures.
        """
        self._initialized = False
        self._init_lock = threading.Lock()  # Protects initialization from race conditions
        self.client = None
        self.collection = None
        self.shoes_collection = None
        self.players_collection = None

    def _ensure_initialized(self) -> None:
        """
        Ensures ChromaDB client and collections are initialized.
        Called lazily on first access to avoid import-time side effects.
        Uses double-check locking to prevent race conditions in multi-threaded environments.

        Raises:
            ValueError: If OPENAI_API_KEY is not configured in settings.
        """
        # First check (without lock) - fast path for already initialized case
        if self._initialized:
            return

        # Acquire lock for initialization
        with self._init_lock:
            # Second check (with lock) - prevents race condition
            if self._initialized:
                return

            # Validate API key before attempting to create embedding function
            if not settings.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is not configured. Please set it in your environment "
                    "or configuration file before using ChromaDBManager."
                )

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

            # Use OpenAI embedding function for consistency
            embedding_function = OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            )

            # Create or get collections
            self.collection = self.client.get_or_create_collection(
                name=DRILLS_COLLECTION_NAME,
                embedding_function=embedding_function
            )
            self.shoes_collection = self.client.get_or_create_collection(
                name=SHOES_COLLECTION_NAME,
                embedding_function=embedding_function
            )
            self.players_collection = self.client.get_or_create_collection(
                name=PLAYERS_COLLECTION_NAME,
                embedding_function=embedding_function
            )

            # Set initialized flag (must be last, acts as memory barrier)
            self._initialized = True

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
        self._ensure_initialized()
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
        self._ensure_initialized()
        results = self.collection.query(
            query_texts=query_texts, n_results=n_results, where=where
        )
        return results

    def add_shoes(
        self, shoes: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """
        Adds shoe documents and their embeddings to the ChromaDB collection.

        Args:
            shoes: A list of shoe documents (dictionaries).
            embeddings: A list of corresponding embedding vectors.

        Raises:
            ValueError: If the number of shoes and embeddings do not match.
        """
        self._ensure_initialized()
        if len(shoes) != len(embeddings):
            raise ValueError(
                f"The number of shoes ({len(shoes)}) must match the number of "
                f"embeddings ({len(embeddings)})."
            )

        if not shoes:
            return

        ids = [shoe["id"] for shoe in shoes]
        documents = [format_shoe_document(shoe) for shoe in shoes]

        # Prepare metadata, ensuring all values are simple types for ChromaDB.
        metadatas = []
        for shoe in shoes:
            metadata = {
                "doc_type": "shoe",
                "shoe_id": shoe["id"],
                "brand": shoe["brand"],
                "model_name": shoe["model_name"],
                "price_krw": shoe["price_krw"],
                "weight_g": shoe["weight_g"],
                "cushion_type": shoe["cushion_type"],
                "support_level": shoe["support_level"],
                "player_signature": shoe.get("player_signature") or "None",
                # Join list into a comma-separated string for metadata compatibility
                "sensory_tags": ",".join(shoe.get("sensory_tags", [])),
                "tags": ",".join(shoe.get("tags", [])),
            }
            metadatas.append(metadata)

        self.shoes_collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query_shoes(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        """
        Queries the shoes collection for relevant documents.

        Args:
            query_texts: A list of query texts to search for.
            n_results: The number of results to return per query.
            where: An optional dictionary for metadata filtering.

        Returns:
            A dictionary containing the query results.
        """
        self._ensure_initialized()
        results = self.shoes_collection.query(
            query_texts=query_texts, n_results=n_results, where=where
        )
        return results

    def add_players(
        self, players: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """
        Adds player archetype documents and their embeddings to the ChromaDB collection.

        Args:
            players: A list of player documents (dictionaries).
            embeddings: A list of corresponding embedding vectors.

        Raises:
            ValueError: If the number of players and embeddings do not match.
        """
        self._ensure_initialized()
        if len(players) != len(embeddings):
            raise ValueError(
                f"The number of players ({len(players)}) must match the number of "
                f"embeddings ({len(embeddings)})."
            )

        if not players:
            return

        ids = [player["id"] for player in players]
        documents = [format_player_document(player) for player in players]

        # Prepare metadata, ensuring all values are simple types for ChromaDB.
        metadatas = []
        for player in players:
            # Safely extract preferred_features with defaults to handle malformed data
            preferred = player.get("preferred_features", {})

            metadata = {
                "doc_type": "player",
                "name": player["name"],
                "position": player["position"],
                # Join lists into comma-separated strings for metadata compatibility
                "play_style": ",".join(player.get("play_style", [])),
                "signature_shoes": ",".join(player.get("signature_shoes", [])),
                # Safely access nested preferred_features with string defaults
                "cushion": str(preferred.get("cushion", "")),
                "support": str(preferred.get("support", "")),
                "traction": str(preferred.get("traction", "")),
            }
            metadatas.append(metadata)

        self.players_collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query_players(
        self,
        query_texts: List[str],
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        """
        Queries the players collection for relevant documents.

        Args:
            query_texts: A list of query texts to search for.
            n_results: The number of results to return per query.
            where: An optional dictionary for metadata filtering.

        Returns:
            A dictionary containing the query results.
        """
        self._ensure_initialized()
        results = self.players_collection.query(
            query_texts=query_texts, n_results=n_results, where=where
        )
        return results


# Create a single instance for the application to use.
chroma_manager = ChromaDBManager()
