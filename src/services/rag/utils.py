from typing import Any, Dict


def format_drill_document(drill: Dict[str, Any]) -> str:
    """Formats a drill dictionary into a consistent string for embedding and storage."""
    return f"Drill: {drill['name']}\nDescription: {drill['description']}"


def format_shoe_document(shoe: Dict[str, Any]) -> str:
    """
    Formats a shoe dictionary into a consistent string for embedding and storage.
    Emphasizes sensory tags and description for better semantic matching.
    """
    sensory = ", ".join(shoe.get("sensory_tags", []))
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
    Formats a player archetype dictionary into a consistent string for embedding and storage.
    Emphasizes play style for better semantic matching.
    """
    styles = ", ".join(player.get("play_style", []))
    return (
        f"Player: {player['name']}\n"
        f"Position: {player['position']}\n"
        f"Play Style: {styles}\n"
        f"Description: {player['description']}"
    )
