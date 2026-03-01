from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints


class WeeklyRoutineRequest(BaseModel):
    """Request model for generating a weekly training routine."""

    skill_level: Literal["beginner", "intermediate", "advanced"] = Field(
        ...,
        description="User's basketball skill level.",
        examples=["intermediate"],
    )
    training_days: int = Field(
        ...,
        ge=1,
        le=7,
        description="Number of training days for the week.",
        examples=[3],
    )
    focus_areas: List[
        Literal["dribble", "shooting", "defense", "conditioning"]
    ] = Field(
        ...,
        min_length=1,
        description="Skill categories the user wants to focus on (multiple allowed).",
        examples=[["shooting", "dribble"]],
    )
    available_time_per_day_min: int = Field(
        ...,
        gt=0,
        description="Available training time per day in minutes.",
        examples=[60],
    )
    equipment: List[str] = Field(
        default_factory=list,
        description="List of available equipment for training.",
        examples=[["ball", "cones", "hoop"]],
    )
    language: Literal["en", "ko"] = Field(
        default="en",
        description="Language for the AI-generated response.",
        examples=["en"],
    )
    free_text: Optional[Annotated[str, StringConstraints(max_length=500)]] = Field(
        default=None,
        description="Optional free-text input describing additional training preferences.",
        examples=["I want to work on my weak hand and improve conditioning"],
    )


class WeeklyDrill(BaseModel):
    """Represents a single drill within a weekly routine day."""

    phase: Literal["warmup", "main", "cooldown"] = Field(
        ..., description="The phase of the workout this drill belongs to."
    )
    drill_id: str = Field(..., description="Unique identifier for the drill.")
    name: str = Field(..., description="The name of the drill.")
    duration_min: int = Field(
        ..., gt=0, description="Duration of the drill in minutes."
    )
    description: str = Field(
        ..., description="A brief description of how to perform the drill."
    )
    coaching_tip: str = Field(
        ..., description="An AI-generated tip for performing the drill effectively."
    )
    is_custom: bool = Field(
        default=False,
        description="Whether this drill was generated/adapted by the LLM.",
    )


class DailyPlan(BaseModel):
    """Represents a single day's training plan within the weekly routine."""

    day_number: int = Field(
        ..., ge=1, le=7, description="Day number in the weekly plan."
    )
    day_label: str = Field(
        ...,
        description="Label for the day, e.g. 'Day 1 - Shooting Focus'.",
    )
    focus_areas: List[str] = Field(
        ..., description="Focus areas for this specific day."
    )
    total_duration_min: int = Field(
        ..., gt=0, description="Total duration of this day's training in minutes."
    )
    drills: List[WeeklyDrill] = Field(
        ..., description="List of drills for this day."
    )


class WeeklyRoutineResponse(BaseModel):
    """Response model for the weekly training routine."""

    weekly_title: str = Field(
        ..., description="A title for the entire weekly routine."
    )
    coach_overview: str = Field(
        ..., description="Weekly strategy overview message from the AI coach."
    )
    total_days: int = Field(
        ..., ge=1, le=7, description="Total number of training days."
    )
    days: List[DailyPlan] = Field(
        ..., description="List of daily training plans."
    )
