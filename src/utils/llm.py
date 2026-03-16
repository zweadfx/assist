import logging

from openai import RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.services.rag.embedding import client as openai_client

logger = logging.getLogger(__name__)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def chat_completion_with_retry(**kwargs):
    """OpenAI chat completion with exponential backoff on RateLimitError."""
    return openai_client.chat.completions.create(**kwargs)
