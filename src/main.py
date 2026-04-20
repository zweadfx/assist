import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.router import api_router
from src.db import models as _db_models  # noqa: F401 — ensure models are registered
from src.db.database import Base, engine
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
from src.services.rag.chroma_db import chroma_manager
from src.services.rag.utils import (
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
        # Migrate saved_plans: drop if old schema (start_date column exists)
        from sqlalchemy import inspect, text
        inspector = inspect(engine)
        if "saved_plans" in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns("saved_plans")]
            if "start_date" in columns:
                with engine.connect() as conn:
                    conn.execute(text("DROP TABLE saved_plans"))
                    conn.commit()
                logger.info("Dropped old saved_plans table (start_date → training_dates migration).")

        # Create relational DB tables (no-op if they already exist)
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized.")

        chroma_manager._ensure_initialized()

        # Initialize shoes collection
        if chroma_manager.shoes_collection.count() == 0:
            logger.info("Shoes collection is empty. Initializing...")

            shoes = load_json_data(SHOES_FILE_PATH)
            logger.info(f"Loaded {len(shoes)} shoes from file.")

            shoes_texts = [format_shoe_document(shoe) for shoe in shoes]
            shoes_embeddings = load_json_data(SHOES_EMBEDDINGS_FILE_PATH)
            logger.info(f"Loaded {len(shoes_embeddings)} shoe embeddings.")

            if len(shoes) != len(shoes_embeddings):
                logger.error(f"Mismatch: {len(shoes)} shoes vs {len(shoes_embeddings)} embeddings ({SHOES_FILE_PATH} vs {SHOES_EMBEDDINGS_FILE_PATH})")
                raise ValueError("Shoes data and embeddings length mismatch")

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
            players_embeddings = load_json_data(PLAYERS_EMBEDDINGS_FILE_PATH)
            logger.info(f"Loaded {len(players_embeddings)} player embeddings.")

            if len(players) != len(players_embeddings):
                logger.error(f"Mismatch: {len(players)} players vs {len(players_embeddings)} embeddings ({PLAYERS_FILE_PATH} vs {PLAYERS_EMBEDDINGS_FILE_PATH})")
                raise ValueError("Players data and embeddings length mismatch")

            chroma_manager.add_players(players=players, embeddings=players_embeddings)
            logger.info("Successfully added players to ChromaDB.")
        else:
            logger.info("Players collection is already initialized.")

        # Re-initialize rules collection if empty or when PDF content changes
        rules_collection_empty = chroma_manager.rules_collection.count() == 0
        rules_pdf_changed = chroma_manager.reinitialize_rules_collection(
            FIBA_RULES_PDF_PATH, NBA_RULES_PDF_PATH
        )
        if rules_collection_empty or rules_pdf_changed:
            if rules_collection_empty:
                logger.info("Rules collection is empty. Initializing...")
            if rules_pdf_changed:
                logger.info("Rules PDF content changed. Re-initializing...")
                
            all_chunks = []

            if FIBA_RULES_PDF_PATH.exists():
                fiba_chunks = parse_rules_pdf(
                    FIBA_RULES_PDF_PATH,
                    rule_type="FIBA",
                    chunk_method="article_based",
                )
                all_chunks.extend(fiba_chunks)
                logger.info(f"Parsed {len(fiba_chunks)} chunks from FIBA rules.")
            else:
                logger.warning(f"FIBA rules PDF not found: {FIBA_RULES_PDF_PATH}")

            if NBA_RULES_PDF_PATH.exists():
                nba_chunks = parse_rules_pdf(
                    NBA_RULES_PDF_PATH,
                    rule_type="NBA",
                    chunk_method="article_based",
                )
                all_chunks.extend(nba_chunks)
                logger.info(f"Parsed {len(nba_chunks)} chunks from NBA rules.")
            else:
                logger.warning(f"NBA rules PDF not found: {NBA_RULES_PDF_PATH}")

            if all_chunks:
                rules_texts = [format_rule_document(chunk) for chunk in all_chunks]
                rules_embeddings = load_json_data(RULES_EMBEDDINGS_FILE_PATH)
                logger.info(f"Loaded {len(rules_embeddings)} rule embeddings.")

                if len(all_chunks) != len(rules_embeddings):
                    logger.error(f"Mismatch: {len(all_chunks)} rule chunks vs {len(rules_embeddings)} embeddings")
                    raise ValueError("Rules data and embeddings length mismatch")

                chroma_manager.add_rules(
                    rule_chunks=all_chunks, embeddings=rules_embeddings
                )
                chroma_manager.commit_rules_hash(
                    FIBA_RULES_PDF_PATH, NBA_RULES_PDF_PATH
                )
                logger.info("Successfully added rules to ChromaDB.")
            else:
                logger.warning("No rules PDF files found. Skipping rules init.")

        # Initialize glossary collection
        if chroma_manager.glossary_collection.count() == 0:
            logger.info("Glossary collection is empty. Initializing...")

            if GLOSSARY_FILE_PATH.exists():
                glossary = load_json_data(GLOSSARY_FILE_PATH)
                logger.info(f"Loaded {len(glossary)} glossary terms from file.")

                glossary_texts = [format_glossary_document(term) for term in glossary]
                glossary_embeddings = load_json_data(GLOSSARY_EMBEDDINGS_FILE_PATH)
                logger.info(
                    f"Loaded {len(glossary_embeddings)} glossary embeddings."
                )

                if len(glossary) != len(glossary_embeddings):
                    logger.error(f"Mismatch: {len(glossary)} glossary terms vs {len(glossary_embeddings)} embeddings ({GLOSSARY_FILE_PATH} vs {GLOSSARY_EMBEDDINGS_FILE_PATH})")
                    raise ValueError("Glossary data and embeddings length mismatch")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://assist-frontend-plum.vercel.app",
    ],
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """A simple health check endpoint."""
    return {"message": "Welcome to the Assist API"}
