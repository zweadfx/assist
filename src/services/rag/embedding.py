from typing import List

from openai import OpenAI

from src.core.config import settings

# Initialize the OpenAI client using the API key from settings
client = OpenAI(api_key=settings.OPENAI_API_KEY)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using OpenAI's API.

    Args:
        texts: A list of strings to be embedded.

    Returns:
        A list of embedding vectors (each vector is a list of floats).
    """
    if not texts:
        return []

    # Replace newlines, which can negatively affect performance.
    texts = [text.replace("\n", " ") for text in texts]

    response = client.embeddings.create(input=texts, model="text-embedding-3-small")

    return [embedding.embedding for embedding in response.data]
