import json
import logging
import os
import tempfile
from pathlib import Path

# Add project root to sys.path to allow imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.constants import (
    FIBA_RULES_PDF_PATH,
    GLOSSARY_FILE_PATH,
    NBA_RULES_PDF_PATH,
    PLAYERS_FILE_PATH,
    SHOES_FILE_PATH,
    SHOES_EMBEDDINGS_FILE_PATH,
    PLAYERS_EMBEDDINGS_FILE_PATH,
    RULES_EMBEDDINGS_FILE_PATH,
    GLOSSARY_EMBEDDINGS_FILE_PATH,
)
from src.services.rag.embedding import generate_embeddings
from src.services.rag.utils import (
    format_glossary_document,
    format_player_document,
    format_rule_document,
    format_shoe_document,
)
from src.utils.file_loader import load_json_data
from src.utils.pdf_parser import parse_rules_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_embeddings(file_path: Path, embeddings: list):
    """
    Saves embeddings to a JSON file atomically using a temporary file.
    """
    # Create a temporary file in the same directory as the target file
    fd, temp_path = tempfile.mkstemp(dir=file_path.parent)
    try:
        # Write data to the temporary file
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False)
        
        # Atomically replace the target file with the temporary file
        os.replace(temp_path, file_path)
        logger.info(f"Saved embeddings to {file_path}")
    except Exception as e:
        # Clean up the temporary file in case of an error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Failed to save embeddings to {file_path}: {e}")
        raise

def main():
    logger.info("Pre-calculating embeddings...")

    # Shoes (REQUIRED)
    if SHOES_FILE_PATH.exists():
        shoes = load_json_data(SHOES_FILE_PATH)
        if not shoes:
            raise ValueError(f"Required data file is empty: {SHOES_FILE_PATH}")
        shoes_texts = [format_shoe_document(shoe) for shoe in shoes]
        shoes_embeddings = generate_embeddings(shoes_texts)
        save_embeddings(SHOES_EMBEDDINGS_FILE_PATH, shoes_embeddings)
    else:
        raise FileNotFoundError(f"Required data file missing: {SHOES_FILE_PATH}")

    # Players (REQUIRED)
    if PLAYERS_FILE_PATH.exists():
        players = load_json_data(PLAYERS_FILE_PATH)
        if not players:
            raise ValueError(f"Required data file is empty: {PLAYERS_FILE_PATH}")
        players_texts = [format_player_document(player) for player in players]
        players_embeddings = generate_embeddings(players_texts)
        save_embeddings(PLAYERS_EMBEDDINGS_FILE_PATH, players_embeddings)
    else:
        raise FileNotFoundError(f"Required data file missing: {PLAYERS_FILE_PATH}")

    # Rules (REQUIRED)
    all_chunks = []
    if FIBA_RULES_PDF_PATH.exists():
        fiba_chunks = parse_rules_pdf(FIBA_RULES_PDF_PATH, rule_type="FIBA", chunk_method="article_based")
        all_chunks.extend(fiba_chunks)
    if NBA_RULES_PDF_PATH.exists():
        nba_chunks = parse_rules_pdf(NBA_RULES_PDF_PATH, rule_type="NBA", chunk_method="article_based")
        all_chunks.extend(nba_chunks)
        
    if all_chunks:
        rules_texts = [format_rule_document(chunk) for chunk in all_chunks]
        rules_embeddings = generate_embeddings(rules_texts)
        save_embeddings(RULES_EMBEDDINGS_FILE_PATH, rules_embeddings)
    else:
        raise ValueError("Required rules chunks are empty or rules PDFs are missing")

    # Glossary (OPTIONAL)
    if GLOSSARY_FILE_PATH.exists():
        glossary = load_json_data(GLOSSARY_FILE_PATH)
        if glossary:
            glossary_texts = [format_glossary_document(term) for term in glossary]
            glossary_embeddings = generate_embeddings(glossary_texts)
            save_embeddings(GLOSSARY_EMBEDDINGS_FILE_PATH, glossary_embeddings)
        else:
            save_embeddings(GLOSSARY_EMBEDDINGS_FILE_PATH, [])
    else:
        save_embeddings(GLOSSARY_EMBEDDINGS_FILE_PATH, [])

    logger.info("Done.")

if __name__ == "__main__":
    main()
