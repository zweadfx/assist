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

# Sensory Tag Enum → Korean Label Mapping
SENSORY_TAG_MAP: dict[str, str] = {
    "cushioning": "쿠셔닝",
    "responsive": "반응성",
    "lightweight": "경량",
    "ankle_support": "발목 지지",
    "traction": "접지력",
    "wide_fit": "와이드 핏",
    "narrow_fit": "내로우 핏",
    "stability": "안정성",
    "comfort": "편안함",
    "court_feel": "코트 감각",
    "durability": "내구성",
    "flexibility": "유연성",
    "breathable": "통기성",
    "lockdown": "락다운",
    "impact_protection": "충격 보호",
    "value": "가성비",
}

# ChromaDB Collection Names
DRILLS_COLLECTION_NAME = "basketball_drills"
SHOES_COLLECTION_NAME = "basketball_shoes"
PLAYERS_COLLECTION_NAME = "basketball_players"
RULES_COLLECTION_NAME = "basketball_rules"
GLOSSARY_COLLECTION_NAME = "basketball_glossary"
