"""
Rule retrieval module for The Whistle.
Handles situation-based rule search, glossary term lookup,
and hybrid search combining both data sources.
"""

import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from src.services.rag.chroma_db import chroma_manager

logger = logging.getLogger(__name__)


class RuleRetriever:
    """
    Handles basketball rule retrieval using semantic search and filtering.

    Provides three main search strategies:
    1. Situation-based search: Match rules by game situation description
    2. Glossary lookup: Find basketball term definitions
    3. Hybrid search: Combine rules and glossary for comprehensive judgment
    """

    def __init__(self):
        """Initialize the rule retriever with ChromaDB manager."""
        self.chroma_manager = chroma_manager

    def search_by_situation(
        self,
        situation: str,
        rule_type: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Document]:
        """
        Search rules by game situation description using vector similarity.

        Args:
            situation: Description of the basketball situation
            rule_type: Filter by rule type ("FIBA" or "NBA"), None for both
            n_results: Number of results to retrieve

        Returns:
            List of Document objects with rule information
        """
        if not situation or not situation.strip():
            logger.info("No situation provided, returning empty results")
            return []

        logger.info(f"Searching rules for situation: {situation[:80]}...")

        try:
            where_filter = None
            if rule_type:
                where_filter = {"rule_type": rule_type.upper()}

            results = self.chroma_manager.query_rules(
                query_texts=[situation],
                n_results=n_results,
                where=where_filter,
            )

            if not results or not results.get("documents"):
                logger.warning("No rules found for the given situation")
                return []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            rule_docs = []
            for i, doc_content in enumerate(documents):
                doc = Document(page_content=doc_content, metadata=metadatas[i])
                rule_docs.append(doc)

            logger.info(f"Retrieved {len(rule_docs)} rule documents")
            return rule_docs

        except Exception as e:
            logger.exception("Failed to search rules by situation")
            raise ValueError("Failed to retrieve rules from database") from e

    def search_glossary_terms(
        self,
        query: str,
        category: Optional[str] = None,
        n_results: int = 3,
    ) -> List[Document]:
        """
        Search glossary for basketball term definitions.

        Args:
            query: Search query (term name or related description)
            category: Filter by category (violation/foul/technique/position)
            n_results: Number of results to retrieve

        Returns:
            List of Document objects with glossary information
        """
        if not query or not query.strip():
            logger.info("No query provided, returning empty results")
            return []

        logger.info(f"Searching glossary for: {query}")

        try:
            where_filter = None
            if category:
                where_filter = {"category": category}

            results = self.chroma_manager.query_glossary(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
            )

            if not results or not results.get("documents"):
                logger.warning(f"No glossary terms found for: {query}")
                return []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            glossary_docs = []
            for i, doc_content in enumerate(documents):
                doc = Document(page_content=doc_content, metadata=metadatas[i])
                glossary_docs.append(doc)

            logger.info(f"Retrieved {len(glossary_docs)} glossary terms")
            return glossary_docs

        except Exception as e:
            logger.exception("Failed to search glossary terms")
            raise ValueError("Failed to retrieve glossary terms from database") from e

    def hybrid_search(
        self,
        situation: str,
        rule_type: Optional[str] = None,
        n_rules: int = 5,
        n_glossary: int = 3,
    ) -> Dict[str, List[Document]]:
        """
        Perform hybrid search combining rules and glossary results.

        This is the main search method that retrieves both relevant rule
        articles and related basketball terms for comprehensive judgment.

        Args:
            situation: Description of the basketball situation
            rule_type: Filter by rule type ("FIBA" or "NBA"), None for both
            n_rules: Number of rule results to return
            n_glossary: Number of glossary results to return

        Returns:
            Dictionary with 'rules' and 'glossary' lists of Documents
        """
        logger.info(
            f"Hybrid search: situation={(situation or '')[:80]}..., rule_type={rule_type}"
        )

        result = {"rules": [], "glossary": []}

        # 1. Search rules by situation
        rules = self.search_by_situation(
            situation=situation,
            rule_type=rule_type,
            n_results=n_rules,
        )
        result["rules"] = rules

        # 2. Search glossary for related terms
        glossary = self.search_glossary_terms(
            query=situation,
            n_results=n_glossary,
        )
        result["glossary"] = glossary

        logger.info(
            f"Hybrid search complete: {len(result['rules'])} rules, "
            f"{len(result['glossary'])} glossary terms"
        )
        return result


# Create singleton instance
rule_retriever = RuleRetriever()
