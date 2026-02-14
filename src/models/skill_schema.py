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
