from typing import List, Literal

from pydantic import BaseModel, Field


class SkillLabRequest(BaseModel):
    """Request model for the AI Skill Lab endpoint.

    Defines the user's input for generating a personalized training routine.
    """

    skill_level: Literal["beginner", "intermediate", "advanced"] = Field(
        ...,
        description="User's basketball skill level.",
        examples=["intermediate"],
    )
    focus_area: Literal["dribble", "shooting", "defense", "conditioning"] = Field(
        ...,
        description="The specific skill category the user wants to focus on.",
        examples=["dribble"],
    )
    available_time_min: int = Field(
        ...,
        gt=0,
        description="Total available time for training in minutes.",
        examples=[30],
    )
    equipment: List[str] = Field(
        default_factory=list,
        description="List of available equipment for training.",
        examples=[["ball", "cones"]],
    )


class Drill(BaseModel):
    """Represents a single training drill within a routine."""

    phase: Literal["warmup", "main", "cooldown"] = Field(
        ..., description="The phase of the workout this drill belongs to."
    )
    drill_id: str = Field(..., description="Unique identifier for the drill.")
    name: str = Field(..., description="The name of the drill.")
    duration_min: int = Field(..., gt=0, description="Duration of the drill in minutes.")
    description: str = Field(
        ..., description="A brief description of how to perform the drill."
    )
    coaching_tip: str = Field(
        ..., description="An AI-generated tip for performing the drill effectively."
    )


class SkillLabResponse(BaseModel):
    """Response model for the AI Skill Lab endpoint.

    Provides a personalized daily routine card.
    """

    routine_title: str = Field(
        ..., description="A catchy title for the generated routine."
    )
    total_duration_min: int = Field(
        ...,
        gt=0,
        description="The total estimated duration of the routine in minutes.",
    )
    coach_message: str = Field(
        ..., description="A personalized motivational message from the AI coach."
    )
    drills: List[Drill] = Field(..., description="A list of drills sequenced for the routine.")
