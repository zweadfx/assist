"""
Shoe retrieval module for Gear Advisor.
Handles sensory keyword-based vector similarity search, player archetype matching,
and multi-filtering with post-processing.
"""
import logging
from typing import Any, Dict, List, Optional

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
            sensory_keywords: List of sensory descriptors (e.g., ["쫀득한 접지", "가벼운 무게"])
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
            logger.info("Sensory keywords resulted in empty query, returning empty results")
            return []

        logger.info(f"Searching shoes by sensory preferences: {sensory_keywords}")

        try:
            # Retrieve candidates from ChromaDB
            results = self.chroma_manager.query_shoes(
                query_texts=[query_text], n_results=n_results
            )

            if not results or not results.get("documents"):
                logger.warning("No shoes found matching sensory preferences")
                return []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            # Post-filtering
            filtered_docs = []
            for i, doc_content in enumerate(documents):
                metadata = metadatas[i]

                # Budget filter
                if budget_max_krw:
                    try:
                        price = int(metadata.get("price_krw", 0))
                        if price > budget_max_krw:
                            continue
                    except (ValueError, TypeError):
                        continue

                # Position filter (optional - based on tags)
                if position:
                    tags = metadata.get("tags", "").split(",")
                    # Simple position matching logic
                    position_match = False
                    if position.lower() == "guard" and any(
                        tag.strip() in ["가드", "로우컷"] for tag in tags
                    ):
                        position_match = True
                    elif position.lower() == "forward" and any(
                        tag.strip() in ["포워드", "미드컷"] for tag in tags
                    ):
                        position_match = True
                    elif position.lower() == "center" and any(
                        tag.strip() in ["센터", "하이컷", "빅맨"] for tag in tags
                    ):
                        position_match = True

                    if not position_match and position.lower() not in [
                        "guard",
                        "forward",
                        "center",
                    ]:
                        # If position is specified but we can't verify match, include anyway
                        position_match = True

                    if not position_match:
                        continue

                # Create Document
                doc = Document(page_content=doc_content, metadata=metadata)
                filtered_docs.append(doc)

            logger.info(
                f"Retrieved {len(filtered_docs)} shoes after filtering "
                f"(from {len(documents)} candidates)"
            )
            return filtered_docs

        except Exception as e:
            logger.exception("Failed to search shoes by sensory preferences")
            raise ValueError(
                "Failed to retrieve shoes from database"
            ) from e

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
        shoes = self.search_by_sensory_preferences(
            sensory_keywords=sensory_keywords,
            budget_max_krw=budget_max_krw,
            position=position,
            n_results=15,  # Get more candidates for better filtering
        )

        # 2. Search player archetypes if specified
        players = []
        if player_archetype:
            players = self.search_by_player_archetype(
                player_name=player_archetype, n_results=3
            )

            # If player found, enhance shoe search with player's preferred features
            if players:
                player_meta = players[0].metadata
                player_shoes = player_meta.get("signature_shoes", "").split(",")

                # Boost shoes that match player's signature models
                shoes = self._boost_signature_shoes(shoes, player_shoes)

        # 3. Limit to top N shoes
        result["shoes"] = shoes[:n_shoes]
        result["players"] = players

        logger.info(
            f"Cross-analysis complete: {len(result['shoes'])} shoes, "
            f"{len(result['players'])} players"
        )
        return result

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
        signature_models = [model.strip() for model in signature_models if model.strip()]

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
