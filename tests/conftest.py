"""
Shared test fixtures.

Mocks OpenAI embedding calls so that app lifespan can initialize
ChromaDB without hitting the real API in CI environments.
"""

from unittest.mock import patch

import pytest

_EMBEDDING_DIM = 1536  # text-embedding-3-small dimension


@pytest.fixture(autouse=True)
def mock_generate_embeddings():
    """Auto-mock generate_embeddings to return dummy vectors."""
    def _fake_embeddings(texts):
        return [[0.0] * _EMBEDDING_DIM for _ in texts]

    with patch(
        "src.services.rag.embedding.generate_embeddings",
        side_effect=_fake_embeddings,
    ):
        yield
