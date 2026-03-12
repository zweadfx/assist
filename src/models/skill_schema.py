from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints


class SkillLabRequest(BaseModel):
    """Request model for the AI Skill Lab endpoint.

    Defines the user's input for generating a micro-step skill breakdown.
    """

    skill_level: Literal["beginner", "intermediate", "advanced"] = Field(
        ...,
        description="User's basketball skill level.",
        examples=["intermediate"],
    )
    category: Literal["dribble", "shooting", "defense", "conditioning"] = Field(
        ...,
        description="The broad skill category for drill retrieval.",
        examples=["dribble"],
    )
    specific_skill: Optional[Annotated[str, StringConstraints(max_length=100)]] = Field(
        default=None,
        description=(
            "The specific technique to master, e.g. 'euro step', "
            "'between the legs', 'pull-up jumper'. "
            "If omitted, the AI coach picks a suitable skill."
        ),
        examples=["euro step"],
    )
    available_time_min: int = Field(
        ...,
        gt=0,
        description="Total available time for training in minutes.",
        examples=[20],
    )
    equipment: List[str] = Field(
        default_factory=list,
        description="List of available equipment for training.",
        examples=[["ball", "cones"]],
    )
    language: Literal["en", "ko"] = Field(
        default="en",
        description="Language for the AI-generated response.",
        examples=["en"],
    )
    free_text: Optional[Annotated[str, StringConstraints(max_length=500)]] = Field(
        default=None,
        description=(
            "Optional free-text input describing additional training preferences "
            "or goals in natural language."
        ),
        examples=["I want to focus on finishing with my weak hand"],
    )


class Step(BaseModel):
    """A single progressive step within a micro-step skill breakdown."""

    step_number: int = Field(
        ..., ge=1, le=5, description="Step number in the progression (1-5)."
    )
    name: str = Field(..., description="Short name of this step.")
    duration_min: int = Field(
        ..., gt=0, description="Duration of this step in minutes."
    )
    description: str = Field(
        ...,
        description=(
            "2-3 sentences explaining how to perform this step, "
            "including specific reps/sets/targets."
        ),
    )
    focus_point: str = Field(
        ...,
        description="The single key focus point for this step.",
    )
    success_criteria: str = Field(
        ...,
        description=(
            "Clear, measurable criteria to pass this step, "
            "e.g. 'Complete 5 consecutive reps without losing the ball'."
        ),
    )


class SkillLabResponse(BaseModel):
    """Response model for the AI Skill Lab endpoint.

    Provides a micro-step progressive breakdown of a single basketball skill.
    """

    skill_name: str = Field(..., description="The specific skill being mastered.")
    total_duration_min: int = Field(
        ...,
        gt=0,
        description="The total duration of the skill breakdown in minutes.",
    )
    difficulty_level: str = Field(
        ...,
        description=(
            "A short summary of the progression range, "
            "e.g. 'Basics → Game Speed' or '기초 → 실전'."
        ),
    )
    coach_message: str = Field(
        ..., description="A personalized motivational message from the AI coach."
    )
    steps: List[Step] = Field(
        ..., description="Progressive steps from simplest to game-like complexity."
    )
