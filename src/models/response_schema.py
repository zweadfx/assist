from typing import Generic, Optional, TypeVar

from pydantic.generics import GenericModel

T = TypeVar("T")


class SuccessResponse(GenericModel, Generic[T]):
    """Standard success response wrapper for all APIs."""

    success: bool = True
    message: str = "Request processed successfully."
    data: Optional[T] = None
