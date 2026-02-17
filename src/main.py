import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.v1.router import api_router
from src.core.constants import (
    DRILLS_FILE_PATH,
    FIBA_RULES_PDF_PATH,
    GLOSSARY_FILE_PATH,
    NBA_RULES_PDF_PATH,
    PLAYERS_FILE_PATH,
    SHOES_FILE_PATH,
)
from src.services.rag.chroma_db import chroma_manager
from src.services.rag.embedding import generate_embeddings
from src.services.rag.utils import (
    format_drill_document,
    format_glossary_document,
    format_player_document,
    format_rule_document,
    format_shoe_document,
)
from src.utils.file_loader import load_json_data
from src.utils.pdf_parser import parse_rules_pdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    On startup, it initializes the vector database if it's empty.
    """
    logger.info("Application startup...")
    try:
        if chroma_manager.collection.count() == 0:
            logger.info("Drills collection is empty. Initializing...")

            drills = load_json_data(DRILLS_FILE_PATH)
            logger.info(f"Loaded {len(drills)} drills from file.")

            texts_to_embed = [format_drill_document(drill) for drill in drills]
            embeddings = generate_embeddings(texts_to_embed)
            logger.info(f"Generated {len(embeddings)} embeddings.")

            # Validate that the number of drills and embeddings match
            if len(drills) != len(embeddings):
                error_msg = (
                    f"Mismatch between number of drills ({len(drills)}) and "
                    f"embeddings ({len(embeddings)}). Aborting startup."
                )
                logger.critical(error_msg)
                raise ValueError(error_msg)

            chroma_manager.add_drills(drills=drills, embeddings=embeddings)
            logger.info("Successfully added drills to ChromaDB.")
        else:
            logger.info("Drills collection is already initialized.")

        # Initialize shoes collection
        if chroma_manager.shoes_collection.count() == 0:
            logger.info("Shoes collection is empty. Initializing...")

            shoes = load_json_data(SHOES_FILE_PATH)
            logger.info(f"Loaded {len(shoes)} shoes from file.")

            shoes_texts = [format_shoe_document(shoe) for shoe in shoes]
            shoes_embeddings = generate_embeddings(shoes_texts)
            logger.info(f"Generated {len(shoes_embeddings)} shoe embeddings.")

            chroma_manager.add_shoes(shoes=shoes, embeddings=shoes_embeddings)
            logger.info("Successfully added shoes to ChromaDB.")
        else:
            logger.info("Shoes collection is already initialized.")

        # Initialize players collection
        if chroma_manager.players_collection.count() == 0:
            logger.info("Players collection is empty. Initializing...")

            players = load_json_data(PLAYERS_FILE_PATH)
            logger.info(f"Loaded {len(players)} players from file.")

            players_texts = [format_player_document(player) for player in players]
            players_embeddings = generate_embeddings(players_texts)
            logger.info(f"Generated {len(players_embeddings)} player embeddings.")

            chroma_manager.add_players(players=players, embeddings=players_embeddings)
            logger.info("Successfully added players to ChromaDB.")
        else:
            logger.info("Players collection is already initialized.")

        # Initialize rules collection
        if chroma_manager.rules_collection.count() == 0:
            logger.info("Rules collection is empty. Initializing...")

            all_chunks = []

            # Parse FIBA rules PDF
            if FIBA_RULES_PDF_PATH.exists():
                fiba_chunks = parse_rules_pdf(
                    FIBA_RULES_PDF_PATH, rule_type="FIBA"
                )
                all_chunks.extend(fiba_chunks)
                logger.info(f"Parsed {len(fiba_chunks)} chunks from FIBA rules.")
            else:
                logger.warning(f"FIBA rules PDF not found: {FIBA_RULES_PDF_PATH}")

            # Parse NBA rules PDF
            if NBA_RULES_PDF_PATH.exists():
                nba_chunks = parse_rules_pdf(
                    NBA_RULES_PDF_PATH, rule_type="NBA"
                )
                all_chunks.extend(nba_chunks)
                logger.info(f"Parsed {len(nba_chunks)} chunks from NBA rules.")
            else:
                logger.warning(f"NBA rules PDF not found: {NBA_RULES_PDF_PATH}")

            if all_chunks:
                rules_texts = [format_rule_document(chunk) for chunk in all_chunks]
                rules_embeddings = generate_embeddings(rules_texts)
                logger.info(f"Generated {len(rules_embeddings)} rule embeddings.")

                chroma_manager.add_rules(
                    rule_chunks=all_chunks, embeddings=rules_embeddings
                )
                logger.info("Successfully added rules to ChromaDB.")
            else:
                logger.warning("No rules PDF files found. Skipping rules init.")
        else:
            logger.info("Rules collection is already initialized.")

        # Initialize glossary collection
        if chroma_manager.glossary_collection.count() == 0:
            logger.info("Glossary collection is empty. Initializing...")

            if GLOSSARY_FILE_PATH.exists():
                glossary = load_json_data(GLOSSARY_FILE_PATH)
                logger.info(f"Loaded {len(glossary)} glossary terms from file.")

                glossary_texts = [
                    format_glossary_document(term) for term in glossary
                ]
                glossary_embeddings = generate_embeddings(glossary_texts)
                logger.info(
                    f"Generated {len(glossary_embeddings)} glossary embeddings."
                )

                chroma_manager.add_glossary(
                    terms=glossary, embeddings=glossary_embeddings
                )
                logger.info("Successfully added glossary to ChromaDB.")
            else:
                logger.warning(
                    f"Glossary file not found: {GLOSSARY_FILE_PATH}. "
                    "Skipping glossary init."
                )
        else:
            logger.info("Glossary collection is already initialized.")

    except Exception as e:
        logger.critical(
            f"A critical error occurred during startup data initialization: {e}",
            exc_info=True,
        )
        # Re-raise the exception to prevent the app from starting in a broken state
        raise

    yield
    logger.info("Application shutdown.")


app = FastAPI(title="Assist API", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """A simple health check endpoint."""
    return {"message": "Welcome to the Assist API"}
