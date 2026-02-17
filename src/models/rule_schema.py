from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class WhistleRequest(BaseModel):
    """Request model for The Whistle endpoint.

    Defines the user's input for requesting a basketball rule judgment.
    """

    situation_description: str = Field(
        ...,
        min_length=1,
        description="Description of the basketball situation to judge.",
        examples=["공을 들고 세 발자국 걸으면 어떤 판정인가요?"],
    )
    rule_type: Optional[Literal["FIBA", "NBA"]] = Field(
        None,
        description="Type of rules to reference. If not specified, both are searched.",
        examples=["FIBA"],
    )


class RuleReference(BaseModel):
    """Represents a reference to a specific rule article."""

    rule_type: str = Field(..., description="Type of rules (FIBA or NBA)")
    article: str = Field(..., description="Article number or identifier")
    clause: str = Field("N/A", description="Specific clause within the article")
    page_number: Optional[int] = Field(
        None, description="Page number in the rules document, if available"
    )
    excerpt: str = Field(..., description="Relevant excerpt from the rule")


class RelatedTerm(BaseModel):
    """Represents a related basketball term from the glossary."""

    term: str = Field(..., description="Basketball term name")
    definition: str = Field(..., description="Definition of the term")


class WhistleResponse(BaseModel):
    """Response model for The Whistle endpoint.

    Provides an AI-generated judgment with rule references.
    """

    judgment_title: str = Field(..., description="A concise title for the judgment")
    situation_summary: str = Field(
        ..., description="Summary of the situation described by the user"
    )
    decision: Literal["violation", "foul", "legal", "other"] = Field(
        ..., description="The judgment decision type"
    )
    reasoning: str = Field(..., description="Detailed reasoning for the judgment")
    rule_references: List[RuleReference] = Field(
        ...,
        min_length=1,
        description="List of rule references supporting the judgment",
    )
    related_terms: List[RelatedTerm] = Field(
        default_factory=list,
        description="List of related basketball terms",
    )
