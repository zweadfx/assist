from fastapi import FastAPI

from src.api.v1.router import api_router

app = FastAPI(title="Assist API")

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """A simple health check endpoint."""
    return {"message": "Welcome to the Assist API"}
