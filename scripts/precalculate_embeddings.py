import json
import logging
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
    RAW_DATA_DIR
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
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)
    logger.info(f"Saved embeddings to {file_path}")

def main():
    logger.info("Pre-calculating embeddings...")

    # Shoes
    if SHOES_FILE_PATH.exists():
        shoes = load_json_data(SHOES_FILE_PATH)
        shoes_texts = [format_shoe_document(shoe) for shoe in shoes]
        shoes_embeddings = generate_embeddings(shoes_texts)
        save_embeddings(RAW_DATA_DIR / "shoes_embeddings.json", shoes_embeddings)

    # Players
    if PLAYERS_FILE_PATH.exists():
        players = load_json_data(PLAYERS_FILE_PATH)
        players_texts = [format_player_document(player) for player in players]
        players_embeddings = generate_embeddings(players_texts)
        save_embeddings(RAW_DATA_DIR / "players_embeddings.json", players_embeddings)

    # Rules
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
        save_embeddings(RAW_DATA_DIR / "rules_embeddings.json", rules_embeddings)

    # Glossary
    if GLOSSARY_FILE_PATH.exists():
        glossary = load_json_data(GLOSSARY_FILE_PATH)
        glossary_texts = [format_glossary_document(term) for term in glossary]
        glossary_embeddings = generate_embeddings(glossary_texts)
        save_embeddings(RAW_DATA_DIR / "glossary_embeddings.json", glossary_embeddings)

    logger.info("Done.")

if __name__ == "__main__":
    main()
