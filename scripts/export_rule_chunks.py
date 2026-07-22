"""Export parsed rule chunks to a committed JSON file.

Parses the FIBA/NBA rulebook PDFs with the exact same path the app startup uses
(``parse_rules_pdf(..., chunk_method="article_based")``) and writes the chunks to
``data/parsed/rules_chunks.json``.

Why: the rulebook PDFs are copyrighted material we don't want to redistribute in
the repository. Committing the parsed chunks (plain dicts, same schema the index
build consumes) lets startup build the rules collection without the PDFs.

Usage:
    uv run python scripts/export_rule_chunks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.constants import (  # noqa: E402
    FIBA_RULES_PDF_PATH,
    NBA_RULES_PDF_PATH,
    PARSED_RULES_FILE_PATH,
)
from src.utils.pdf_parser import parse_rules_pdf  # noqa: E402


def main() -> int:
    chunks = []
    for path, rule_type in ((FIBA_RULES_PDF_PATH, "FIBA"), (NBA_RULES_PDF_PATH, "NBA")):
        if not path.exists():
            print(f"PDF not found: {path} — cannot export without the source PDFs.")
            return 1
        parsed = parse_rules_pdf(
            path, rule_type=rule_type, chunk_method="article_based"
        )
        print(f"{rule_type}: {len(parsed)} chunks")
        chunks.extend(parsed)

    PARSED_RULES_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PARSED_RULES_FILE_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(chunks)} chunks → {PARSED_RULES_FILE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
