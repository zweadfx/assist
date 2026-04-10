"""Pydantic schemas for training plan (calendar) endpoints."""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SavePlanRequest(BaseModel):
    plan_type: Literal["weekly", "skill"]
    title: str = Field(..., min_length=1, max_length=200)
    data: dict[str, Any]
    training_dates: list[date] = Field(..., min_length=1)

    @field_validator("training_dates")
    @classmethod
    def no_duplicate_dates(cls, v: list[date]) -> list[date]:
        if len(v) != len(set(v)):
            raise ValueError("training_dates must not contain duplicates")
        return v


class SavedPlanResponse(BaseModel):
    id: int
    plan_type: str
    title: str
    data: dict[str, Any]
    training_dates: list[date]
    total_days: int
    completed_days: list[int]
    created_at: datetime

    model_config = {"from_attributes": True}


class CompleteDayRequest(BaseModel):
    day_number: int = Field(..., ge=1)
    completed: bool
