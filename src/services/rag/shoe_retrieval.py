"""
Shoe retrieval module for Gear Advisor.
Handles sensory keyword-based vector similarity search, player archetype matching,
and multi-filtering with post-processing.
"""

import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from src.services.rag.chroma_db import chroma_manager

logger = logging.getLogger(__name__)


class ShoeRetriever:
    """
    Handles basketball shoe retrieval using semantic search and filtering.

    Provides three main search strategies:
    1. Sensory-based search: Match shoes by sensory tags (e.g., "sticky traction")
    2. Player archetype search: Find shoes matching professional player styles
    3. Cross-analysis: Combine sensory and player preferences for optimal matching
    """

    def __init__(self):
        """Initialize the shoe retriever with ChromaDB manager."""
        self.chroma_manager = chroma_manager

    def search_by_sensory_preferences(
        self,
        sensory_keywords: List[str],
        budget_max_krw: Optional[int] = None,
        position: Optional[str] = None,
        n_results: int = 10,
    ) -> List[Document]:
        """
        Search shoes by sensory preferences using vector similarity.

        Args:
            sensory_keywords: List of sensory descriptors
                (e.g., ["쫀득한 접지", "가벼운 무게"])
            budget_max_krw: Maximum budget in KRW (optional filter)
            position: Player position (guard/forward/center) for filtering (optional)
            n_results: Number of candidate results to retrieve

        Returns:
            List of Document objects with shoe information
        """
        # Early guard: check if sensory keywords are provided
        if not sensory_keywords or not any(k.strip() for k in sensory_keywords):
            logger.info("No sensory keywords provided, returning empty results")
            return []

        # Build search query from sensory keywords
        query_text = " ".join(sensory_keywords).strip()

        # Additional safety check for empty query after joining
        if not query_text:
            logger.info(
                "Sensory keywords resulted in empty query, returning empty results"
            )
            return []

        logger.info(f"Searching shoes by sensory preferences: {sensory_keywords}")

        try:
            # Build where filter for DB-level pre-filtering
            where_conditions = []
            if budget_max_krw and budget_max_krw > 0:
                where_conditions.append({"price_krw": {"$lte": budget_max_krw}})

            where_filter = None
            if len(where_conditions) == 1:
                where_filter = where_conditions[0]
            elif len(where_conditions) > 1:
                where_filter = {"$and": where_conditions}

            # Retrieve candidates from ChromaDB with DB-level filtering
            results = self.chroma_manager.query_shoes(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
            )

            if not results or not results.get("documents"):
                logger.warning("No shoes found matching sensory preferences")
                return []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            # Build all candidate documents first
            all_docs = []
            for i, doc_content in enumerate(documents):
                doc = Document(page_content=doc_content, metadata=metadatas[i])
                all_docs.append(doc)

            # Post-filtering for position (tags are comma-separated, not
            # suitable for DB filter)
            if position and position.lower() in ["guard", "forward", "center"]:
                position_tag_map = {
                    "guard": ["가드", "로우컷"],
                    "forward": ["포워드", "미드컷"],
                    "center": ["센터", "하이컷", "빅맨"],
                }
                target_tags = position_tag_map[position.lower()]

                filtered_docs = [
                    doc
                    for doc in all_docs
                    if any(
                        tag.strip() in target_tags
                        for tag in doc.metadata.get("tags", "").split(",")
                    )
                ]

                # Fallback: if position filter removes all results, return
                # unfiltered to avoid empty response
                if not filtered_docs:
                    logger.warning(
                        "Position filter '%s' removed all candidates, "
                        "returning unfiltered results",
                        position,
                    )
                    filtered_docs = all_docs
            else:
                filtered_docs = all_docs

            logger.info(
                "Retrieved %d shoes after filtering (from %d candidates)",
                len(filtered_docs),
                len(documents),
            )
            return filtered_docs

        except Exception as e:
            logger.exception("Failed to search shoes by sensory preferences")
            raise ValueError("Failed to retrieve shoes from database") from e

    def search_by_player_archetype(
        self, player_name: str, n_results: int = 3
    ) -> List[Document]:
        """
        Search player archetypes to understand playstyle preferences.

        Args:
            player_name: Name of the professional player (e.g., "Stephen Curry")
            n_results: Number of similar players to retrieve

        Returns:
            List of Document objects with player archetype information
        """
        # Early guard: check if player name is provided
        if not player_name or not player_name.strip():
            logger.info("No player name provided, returning empty results")
            return []

        logger.info(f"Searching player archetype: {player_name}")

        try:
            results = self.chroma_manager.query_players(
                query_texts=[player_name], n_results=n_results
            )

            if not results or not results.get("documents"):
                logger.warning(f"No player archetypes found for: {player_name}")
                return []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            player_docs = []
            for i, doc_content in enumerate(documents):
                metadata = metadatas[i]
                doc = Document(page_content=doc_content, metadata=metadata)
                player_docs.append(doc)

            logger.info(f"Retrieved {len(player_docs)} player archetypes")
            return player_docs

        except Exception as e:
            logger.exception("Failed to search player archetypes")
            raise ValueError(
                "Failed to retrieve player archetypes from database"
            ) from e

    def cross_analysis_search(
        self,
        sensory_keywords: List[str],
        player_archetype: Optional[str] = None,
        budget_max_krw: Optional[int] = None,
        position: Optional[str] = None,
        n_shoes: int = 5,
    ) -> Dict[str, List[Document]]:
        """
        Perform cross-analysis combining sensory preferences and player archetype.

        This is the main search method that combines multiple signals:
        1. Semantic search by sensory keywords
        2. Player archetype matching (if specified)
        3. Budget and position filtering

        Args:
            sensory_keywords: List of sensory descriptors
            player_archetype: Name of preferred player (optional)
            budget_max_krw: Maximum budget in KRW (optional)
            position: Player position for filtering (optional)
            n_shoes: Number of shoes to return

        Returns:
            Dictionary with 'shoes' and 'players' lists of Documents
        """
        logger.info(
            f"Cross-analysis search: sensory={sensory_keywords}, "
            f"player={player_archetype}, budget={budget_max_krw}"
        )

        result = {"shoes": [], "players": []}

        # 1. Search shoes by sensory preferences
        sensory_shoes = self.search_by_sensory_preferences(
            sensory_keywords=sensory_keywords,
            budget_max_krw=budget_max_krw,
            position=position,
            n_results=15,  # Get more candidates for better filtering
        )

        # 2. Search player archetypes and retrieve signature shoes if specified
        players = []
        signature_shoes = []
        if player_archetype:
            players = self.search_by_player_archetype(
                player_name=player_archetype, n_results=3
            )

            # Directly retrieve signature shoes from DB by player_signature
            if players:
                player_name = players[0].metadata.get("name", "")
                signature_shoes = self._get_signature_shoes(player_name)
                logger.info(
                    "Retrieved %d signature shoes for %s",
                    len(signature_shoes),
                    player_name,
                )

        # 3. Merge: signature shoes first, then sensory shoes (deduplicated)
        signature_ids = {
            doc.metadata.get("shoe_id") for doc in signature_shoes
        }
        deduplicated_sensory = [
            doc
            for doc in sensory_shoes
            if doc.metadata.get("shoe_id") not in signature_ids
        ]
        merged_shoes = signature_shoes + deduplicated_sensory

        # 4. Limit to top N shoes
        result["shoes"] = merged_shoes[:n_shoes]
        result["players"] = players

        logger.info(
            f"Cross-analysis complete: {len(result['shoes'])} shoes, "
            f"{len(result['players'])} players"
        )
        return result

    def _get_signature_shoes(self, player_name: str) -> List[Document]:
        """
        Directly retrieve shoes associated with a player via player_signature
        metadata in ChromaDB.

        Args:
            player_name: The player's name to match against player_signature.

        Returns:
            List of Document objects for the player's signature shoes.
        """
        if not player_name:
            return []

        try:
            results = self.chroma_manager.get_shoes_by_player(player_name)

            if not results or not results.get("documents"):
                logger.info("No signature shoes found for player: %s", player_name)
                return []

            documents = results["documents"]
            metadatas = results["metadatas"]

            sig_docs = []
            for i, doc_content in enumerate(documents):
                doc = Document(page_content=doc_content, metadata=metadatas[i])
                sig_docs.append(doc)

            return sig_docs

        except Exception as e:
            logger.warning("Failed to retrieve signature shoes: %s", e)
            return []

    def _boost_signature_shoes(
        self, shoes: List[Document], signature_models: List[str]
    ) -> List[Document]:
        """
        Boost ranking of shoes that match player's signature models.

        Args:
            shoes: List of shoe Documents
            signature_models: List of signature shoe model names

        Returns:
            Reordered list with signature shoes prioritized
        """
        if not signature_models:
            return shoes

        # Clean signature model names
        signature_models = [
            model.strip() for model in signature_models if model.strip()
        ]

        signature_shoes = []
        other_shoes = []

        for shoe in shoes:
            model_name = shoe.metadata.get("model_name", "")
            brand = shoe.metadata.get("brand", "")

            # Check if this shoe matches any signature model
            is_signature = False
            for sig_model in signature_models:
                if sig_model.lower() in f"{brand} {model_name}".lower():
                    is_signature = True
                    break

            if is_signature:
                signature_shoes.append(shoe)
            else:
                other_shoes.append(shoe)

        # Return signature shoes first, then others
        return signature_shoes + other_shoes


# Create singleton instance
shoe_retriever = ShoeRetriever()
