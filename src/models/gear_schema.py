from typing import List, Optional

from pydantic import BaseModel, Field


class GearAdvisorRequest(BaseModel):
    """Request model for the Gear Advisor endpoint.

    Defines the user's input for generating personalized shoe recommendations.
    """

    sensory_preferences: List[str] = Field(
        ...,
        min_length=1,
        description="User's sensory preferences (e.g., 'sticky traction', 'cloud cushion')",
        examples=[["쫀득한 접지", "가벼운 무게"]],
    )
    player_archetype: Optional[str] = Field(
        None,
        description="Preferred player's name to match playstyle",
        examples=["Stephen Curry"],
    )
    position: Optional[str] = Field(
        None,
        description="User's playing position",
        examples=["guard"],
    )
    budget_max_krw: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum budget in KRW",
        examples=[200000],
    )


class ShoeRecommendation(BaseModel):
    """Represents a single shoe recommendation."""

    shoe_id: str = Field(..., description="Unique identifier for the shoe")
    brand: str = Field(..., description="Shoe brand name")
    model_name: str = Field(..., description="Shoe model name")
    price_krw: int = Field(..., description="Price in Korean Won")
    sensory_tags: List[str] = Field(
        ..., description="Sensory characteristics of the shoe"
    )
    match_score: int = Field(
        ..., ge=0, le=100, description="Match percentage (0-100)"
    )
    recommendation_reason: str = Field(
        ..., description="AI-generated reasoning for this recommendation"
    )


class GearAdvisorResponse(BaseModel):
    """Response model for the Gear Advisor endpoint.

    Provides personalized basketball shoe recommendations.
    """

    recommendation_title: str = Field(
        ..., description="A catchy title for the recommendation set"
    )
    user_profile_summary: str = Field(
        ..., description="Summary of user's preferences and profile"
    )
    ai_reasoning: str = Field(
        ..., description="Overall recommendation strategy and reasoning"
    )
    shoes: List[ShoeRecommendation] = Field(
        ..., min_length=1, max_length=5, description="List of recommended shoes (1-5)"
    )
