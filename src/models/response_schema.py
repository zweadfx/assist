from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response wrapper for all APIs."""

    success: bool = True
    message: str = "Request processed successfully."
    data: Optional[T] = None
