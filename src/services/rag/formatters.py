"""
Document formatters for RAG embeddings.
Formats raw data dictionaries into consistent text for vector storage.
"""

from typing import Any, Dict

from src.core.constants import SENSORY_TAG_MAP


def format_drill_document(drill: Dict[str, Any]) -> str:
    """Formats a drill dictionary into a consistent string for embedding and storage."""
    return f"Drill: {drill['name']}\nDescription: {drill['description']}"


def format_shoe_document(shoe: Dict[str, Any]) -> str:
    """
    Formats a shoe dictionary into a consistent string for embedding and storage.
    Emphasizes sensory tags and description for better semantic matching.
    Uses Korean labels for sensory tags to improve Korean semantic search.
    """
    sensory_en = shoe.get("sensory_tags", [])
    sensory_kr = shoe.get("sensory_tags_kr", [])
    if not sensory_kr:
        sensory_kr = [SENSORY_TAG_MAP.get(tag, tag) for tag in sensory_en]
    sensory = ", ".join(sensory_kr)
    player_sig = shoe.get("player_signature") or "N/A"
    return (
        f"Brand: {shoe['brand']}\n"
        f"Model: {shoe['model_name']}\n"
        f"Sensory Tags: {sensory}\n"
        f"Player Signature: {player_sig}\n"
        f"Description: {shoe['description']}"
    )


def format_player_document(player: Dict[str, Any]) -> str:
    """
    Formats a player archetype dictionary into a consistent string for embedding and
    storage.
    Emphasizes play style for better semantic matching.
    """
    styles = ", ".join(player.get("play_style", []))
    name_ko = player.get("name_ko", "")
    name_line = f"Player: {player['name']}"
    if name_ko:
        name_line += f" ({name_ko})"
    return (
        f"{name_line}\n"
        f"Position: {player['position']}\n"
        f"Play Style: {styles}\n"
        f"Description: {player['description']}"
    )


def format_rule_document(chunk: Dict[str, Any]) -> str:
    """
    Formats a rule chunk dictionary into a consistent string for embedding and storage.
    Includes rule type and article info for better context.
    """
    article = chunk.get("article", "N/A")
    return (
        f"Rule Type: {chunk['rule_type']}\n"
        f"Article: {article}\n"
        f"Content: {chunk['content']}"
    )


def format_glossary_document(term: Dict[str, Any]) -> str:
    """
    Formats a glossary term dictionary into a consistent string for embedding and
    storage. Combines term, definition, and explanation for comprehensive semantic
    matching.
    """
    examples = ", ".join(term.get("examples", []))
    return (
        f"Term: {term['term']}\n"
        f"Category: {term['category']}\n"
        f"Definition: {term['definition']}\n"
        f"Explanation: {term['detailed_explanation']}\n"
        f"Examples: {examples}"
    )
