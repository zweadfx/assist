import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.v1.router import api_router
from src.core.constants import DRILLS_FILE_PATH
from src.services.rag.chroma_db import chroma_manager
from src.services.rag.embedding import generate_embeddings
from src.utils.file_loader import load_json_data

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

            texts_to_embed = [
                f"Drill: {drill['name']}\nDescription: {drill['description']}"
                for drill in drills
            ]
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
