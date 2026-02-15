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
    """
    logger.info("Application startup...")
    try:
        # Check if the drills collection is empty
        if chroma_manager.collection.count() == 0:
            logger.info("Drills collection is empty. Initializing...")

            # 1. Load drill data from the source file
            drills = load_json_data(DRILLS_FILE_PATH)
            logger.info(f"Loaded {len(drills)} drills from file.")

            # 2. Prepare texts for embedding
            texts_to_embed = [
                f"Drill: {drill['name']}\nDescription: {drill['description']}"
                for drill in drills
            ]

            # 3. Generate embeddings
            embeddings = generate_embeddings(texts_to_embed)
            logger.info("Generated embeddings for all drills.")

            # 4. Add drills to ChromaDB
            chroma_manager.add_drills(drills=drills, embeddings=embeddings)
            logger.info("Successfully added drills to ChromaDB.")
        else:
            logger.info("Drills collection is already initialized.")
    except Exception as e:
        logger.error(f"An error occurred during startup: {e}", exc_info=True)

    yield
    # Cleanup logic can go here if needed
    logger.info("Application shutdown.")


app = FastAPI(title="Assist API", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """A simple health check endpoint."""
    return {"message": "Welcome to the Assist API"}
