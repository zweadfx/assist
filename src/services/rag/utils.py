"""
Utility functions for RAG services.
Re-exports formatters for backward compatibility.
"""
# Re-export formatters for backward compatibility
from src.services.rag.formatters import (
    format_drill_document,
    format_glossary_document,
    format_player_document,
    format_rule_document,
    format_shoe_document,
)

__all__ = [
    "format_drill_document",
    "format_shoe_document",
    "format_player_document",
    "format_rule_document",
    "format_glossary_document",
]
