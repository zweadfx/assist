"""
A module to store project-wide constants.
"""

from pathlib import Path

# Base Directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Data Files
DRILLS_FILE_PATH = RAW_DATA_DIR / "drills.json"
SHOES_FILE_PATH = RAW_DATA_DIR / "shoes.json"
PLAYERS_FILE_PATH = RAW_DATA_DIR / "players.json"
GLOSSARY_FILE_PATH = RAW_DATA_DIR / "glossary.json"
FIBA_RULES_PDF_PATH = RAW_DATA_DIR / "fiba_rules.pdf"
NBA_RULES_PDF_PATH = RAW_DATA_DIR / "nba_rules.pdf"

# ChromaDB Collection Names
DRILLS_COLLECTION_NAME = "basketball_drills"
SHOES_COLLECTION_NAME = "basketball_shoes"
PLAYERS_COLLECTION_NAME = "basketball_players"
RULES_COLLECTION_NAME = "basketball_rules"
GLOSSARY_COLLECTION_NAME = "basketball_glossary"
