from typing import Any, Dict


def format_drill_document(drill: Dict[str, Any]) -> str:
    """Formats a drill dictionary into a consistent string for embedding and storage."""
    return f"Drill: {drill['name']}\nDescription: {drill['description']}"
