import json
import logging
from typing import List, TypedDict

import openai
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from src.core.constants import KO_BASKETBALL_TERMINOLOGY
from src.models.skill_schema import Step
from src.utils.llm import chat_completion_with_retry

logger = logging.getLogger(__name__)


class CoachAgentState(TypedDict):
    """
    Represents the state of the CoachAgent workflow. It holds all the data
    that is passed between nodes in the graph.
    """

    messages: List[BaseMessage]
    user_info: dict
    final_response: str


class ParsedFreeText(BaseModel):
    """Structured output from parsing the user's free-text input."""

    additional_focus: str = Field(
        default="",
        description="Specific sub-skill or technique the user wants to work on.",
    )
    additional_equipment: List[str] = Field(
        default_factory=list,
        description="Extra equipment mentioned in the free text.",
    )
    intensity_preference: str = Field(
        default="",
        description="Preferred intensity level (e.g., 'light', 'moderate', 'intense').",
    )
    special_notes: str = Field(
        default="",
        description="Any other relevant details from the user's request.",
    )


def _parse_free_text(free_text: str) -> dict:
    """Parse free-text input using LLM to extract structured preferences."""
    schema_json = json.dumps(ParsedFreeText.model_json_schema(), indent=2)

    prompt = f"""Extract structured training preferences from the following user input.
If a field is not mentioned, leave it as empty string or empty list.

User input: "{free_text}"

Output a JSON object following this schema:
```json
{schema_json}
```

JSON Output:"""

    try:
        response = chat_completion_with_retry(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        if not response.choices or not response.choices[0].message.content:
            logger.warning("Empty response from LLM for free_text parsing")
            return {}

        content = response.choices[0].message.content
        parsed = ParsedFreeText.model_validate_json(content)
        return parsed.model_dump(exclude_defaults=True)

    except Exception as e:
        logger.warning("Failed to parse free_text, skipping: %s", e)
        return {}


def diagnose_user_state(state: CoachAgentState) -> dict:
    """
    Validates that the user_info is present in the state. If free_text is
    provided, parses it with an LLM to enrich the user_info with additional
    preferences.
    """
    logger.info("NODE: Diagnosing User State")
    if not state.get("user_info"):
        raise ValueError("User info is missing from the state.")

    user_info = {**state["user_info"]}

    free_text = user_info.get("free_text")
    if free_text and free_text.strip():
        logger.info("Parsing free_text input: %s", free_text[:100])
        parsed = _parse_free_text(free_text)

        if parsed.get("additional_equipment"):
            existing = set(user_info.get("equipment", []))
            existing.update(parsed["additional_equipment"])
            user_info["equipment"] = list(existing)

        if parsed.get("additional_focus"):
            user_info["additional_focus"] = parsed["additional_focus"]

        if parsed.get("intensity_preference"):
            user_info["intensity_preference"] = parsed["intensity_preference"]

        if parsed.get("special_notes"):
            user_info["special_notes"] = parsed["special_notes"]

        logger.debug("Enriched User Info: %s", user_info)

    return {"user_info": user_info}


class SkillBreakdownCard(BaseModel):
    """Data model for the micro-step skill breakdown output."""

    skill_name: str = Field(description="The specific basketball skill being mastered.")
    total_duration_min: int = Field(
        description="The total duration for the skill breakdown in minutes.",
        gt=0,
    )
    difficulty_level: str = Field(
        description=(
            "A short summary of the progression range, e.g. 'Basics → Game Speed'."
        )
    )
    coach_message: str = Field(
        description="A personalized, encouraging message from the AI coach."
    )
    steps: List[Step]


def generate_skill_breakdown(state: CoachAgentState) -> dict:
    """
    Generates a micro-step progressive breakdown of a single basketball
    skill based on the user's profile and preferences.
    """
    logger.info("NODE: Generating Skill Breakdown")
    user_info = state["user_info"]

    language = user_info.get("language", "en")

    schema_json = json.dumps(SkillBreakdownCard.model_json_schema(), indent=2)

    language_name = "Korean" if language == "ko" else "English"

    available_time = user_info.get("available_time_min", 20)
    specific_skill = user_info.get("specific_skill") or ""
    category = user_info.get("category", "")

    skill_instruction = (
        f'The user wants to master: "{specific_skill}".'
        if specific_skill
        else (
            f"The user did not specify a skill. Pick the most impactful "
            f"{category} technique for a {user_info.get('skill_level', 'intermediate')} "
            f"player and set it as skill_name."
        )
    )

    prompt = f"""You are an expert basketball skills coach who specializes in
breaking down individual techniques into progressive micro-steps.

**User Profile:**
- Skill Level: {user_info.get("skill_level", "intermediate")}
- Category: {category}
- Available Time: {available_time} minutes
- Available Equipment: {user_info.get("equipment")}
- Additional Focus: {user_info.get("additional_focus") or "None"}
- Intensity Preference: {user_info.get("intensity_preference") or "None"}
- Special Notes: {user_info.get("special_notes") or "None"}
- Additional Request (raw): {user_info.get("free_text") or "None"}

**Skill Selection:**
{skill_instruction}

**Language:**
Respond in {language_name}. All string fields (skill_name, coach_message,
step name, description, focus_point, success_criteria) must be in {language_name}.

**Instructions:**
1. Break the skill into 3-5 progressive steps, ordered from simplest to
   most game-like:
   - Step 1: MUST start with no ball or stationary movement (anyone can do it)
   - Each subsequent step adds exactly ONE layer of complexity
     (e.g., add ball → add movement → add speed → add defender/obstacle)
   - The final step MUST simulate a real game situation
2. The sum of all step durations MUST equal exactly {available_time} minutes.
3. Each step must have:
   - A clear, descriptive name
   - A description: step-by-step execution with specific reps, sets, or targets (3-4 sentences minimum)
   - A focus_point: the ONE thing to concentrate on in this step
   - A success_criteria: a measurable goal to pass this step
     (e.g., "Complete 5 consecutive reps without losing the ball")
4. **Step Quality Standards (STRICTLY ENFORCE):**
   - Every step must build toward real in-game application.
     Prefer: movement-based progressions, game-speed reps, adding a defender or obstacle.
     Avoid: purely stationary isolated reps with no game context, drills that train
     a weakness in isolation without connecting to a game action (e.g., do NOT generate
     "weak hand free throw" — instead use "weak hand drive to layup" or
     "weak hand finish off a screen").
   - Each step must include a concrete target or success metric
     (e.g., "8 out of 10 made", "3 sets of 45 seconds", "5 consecutive clean reps").
5. Set difficulty_level to a short phrase showing the progression range
   (e.g., "Basics → Game Speed" or "기초 → 실전").
6. Write a motivating coach_message about mastering this specific skill.
7. If "Additional Request (raw)" is provided, actively reflect its content
   throughout the routine — incorporate the requested elements into step
   names, descriptions, and coach_message. Do not treat it as optional context.
8. Output a JSON object strictly following this schema:

```json
{schema_json}
```

JSON Output:
"""
    messages = []
    if language == "ko":
        messages.append({"role": "system", "content": KO_BASKETBALL_TERMINOLOGY})
    messages.append({"role": "user", "content": prompt})

    try:
        response = chat_completion_with_retry(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
        )

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Received an invalid or empty response from OpenAI API.")

        content = response.choices[0].message.content

        try:
            extracted_data = json.loads(content)
            validated = SkillBreakdownCard.model_validate(extracted_data)
            final_response_str = validated.model_dump_json(indent=2)
            logger.debug("Generated Response: %s", final_response_str)
            return {"final_response": final_response_str}
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                "Failed to parse or validate LLM response: %s (raw content: %.500s)",
                e,
                content,
            )
            raise ValueError("LLM returned an invalid skill breakdown object") from e

    except openai.APIError as e:
        logger.error("OpenAI API error during skill breakdown: %s", e)
        raise ValueError(
            "Failed to generate skill breakdown due to an API error."
        ) from e
    except Exception as e:
        logger.error("An unexpected error occurred during skill breakdown: %s", e)
        raise


# Define the graph workflow
workflow = StateGraph(CoachAgentState)

workflow.add_node("diagnose", diagnose_user_state)
workflow.add_node("generate", generate_skill_breakdown)

workflow.set_entry_point("diagnose")
workflow.add_edge("diagnose", "generate")
workflow.add_edge("generate", END)

coach_agent_graph = workflow.compile()
