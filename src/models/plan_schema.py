"""Pydantic schemas for training plan (calendar) endpoints."""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel


class SavePlanRequest(BaseModel):
    plan_type: Literal["weekly", "skill"]
    title: str
    data: dict[str, Any]
    start_date: date
    total_days: int


class SavedPlanResponse(BaseModel):
    id: int
    plan_type: str
    title: str
    data: dict[str, Any]
    start_date: date
    total_days: int
    completed_days: list[int]
    created_at: datetime

    model_config = {"from_attributes": True}


class CompleteDayRequest(BaseModel):
    day_number: int
    completed: bool
