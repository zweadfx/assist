import json
from pathlib import Path
from typing import Any, Dict, List


def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads data from a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A list of dictionaries containing the data from the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"The file was not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error decoding JSON from {file_path}: {e.msg}", e.doc, e.pos
            ) from e
    return data
