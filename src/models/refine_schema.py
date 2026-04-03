"""Request schemas for feedback-based refinement of training routines."""

from pydantic import BaseModel, Field

from src.models.skill_schema import SkillLabRequest, SkillLabResponse
from src.models.weekly_schema import WeeklyRoutineRequest, WeeklyRoutineResponse


class SkillRefineRequest(BaseModel):
    """Request to refine a previously generated skill breakdown."""

    original_request: SkillLabRequest
    previous_response: SkillLabResponse
    feedback: str = Field(..., min_length=1, max_length=500)


class WeeklyRefineRequest(BaseModel):
    """Request to refine a previously generated weekly routine."""

    original_request: WeeklyRoutineRequest
    previous_response: WeeklyRoutineResponse
    feedback: str = Field(..., min_length=1, max_length=500)
