"""
Weekly Coach Refine Agent: refines a previously generated weekly routine
based on user feedback, using a LangGraph conditional-edge loop.

Graph:
    classify_feedback ──┬── re_retrieve ──→ refine_generate ──→ END
                        └─────────────────→ refine_generate ──→ END
"""

import json
import logging
from typing import Dict, List, TypedDict

import openai
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.core.constants import KO_BASKETBALL_TERMINOLOGY
from src.models.weekly_schema import WeeklyRoutineResponse
from src.services.agents.weekly_coach_agent import retrieve_drills
from src.utils.llm import chat_completion_with_retry

logger = logging.getLogger(__name__)


class WeeklyRefineState(TypedDict):
    """State for the weekly coach refine workflow."""

    messages: List[BaseMessage]
    user_info: dict
    week_plan: dict
    context: dict  # day-to-drills mapping
    previous_response: str
    feedback: str
    feedback_type: str
    final_response: str


def classify_feedback(state: WeeklyRefineState) -> dict:
    """Classify feedback as needing re-retrieval or regeneration only."""
    logger.info("WEEKLY REFINE NODE: Classifying Feedback")
    feedback = state["feedback"]

    prompt = f"""You are a classifier for a basketball training app.
The user received a weekly training routine and gave feedback. Classify the feedback:

- "re_retrieve": feedback asks for DIFFERENT drills, exercises, or content
  (e.g. "다른 드릴로 바꿔줘", "different exercises", "swap the drills")
- "regenerate_only": feedback asks for ADJUSTMENTS to existing content
  (e.g. "화요일 빼줘", "시간 줄여줘", "make day 2 easier", "remove conditioning")

User Feedback: "{feedback}"

Output ONLY: re_retrieve OR regenerate_only"""

    try:
        response = chat_completion_with_retry(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        msg = response.choices[0].message.content
        feedback_type = msg.strip().lower() if msg else ""

        if feedback_type not in ("re_retrieve", "regenerate_only"):
            logger.warning(
                "Invalid feedback_type '%s', defaulting to 'regenerate_only'",
                feedback_type,
            )
            feedback_type = "regenerate_only"

        logger.info("Feedback classified as: %s", feedback_type)
        return {"feedback_type": feedback_type}

    except Exception:
        logger.exception("Error classifying feedback, defaulting to regenerate_only")
        return {"feedback_type": "regenerate_only"}


def re_retrieve(state: WeeklyRefineState) -> dict:
    """Re-retrieve drills from ChromaDB using existing retrieve_drills."""
    logger.info("WEEKLY REFINE NODE: Re-retrieving Drills")
    return retrieve_drills(state)


def refine_generate(state: WeeklyRefineState) -> dict:
    """Generate a refined weekly routine based on feedback."""
    logger.info("WEEKLY REFINE NODE: Generating Refined Weekly Routine")
    user_info = state["user_info"]
    week_plan = state["week_plan"]
    day_drills = state.get("context", {})
    previous_response = state["previous_response"]
    feedback = state["feedback"]

    training_days = user_info.get("training_days", 3)
    available_time = user_info.get("available_time_per_day_min", 60)
    skill_level = user_info.get("skill_level", "intermediate")
    language = user_info.get("language", "en")
    language_name = "Korean" if language == "ko" else "English"

    # Build per-day context strings
    days_section = ""
    for day_key in sorted(week_plan.keys(), key=int):
        focus = week_plan[day_key]
        drills = day_drills.get(day_key, [])
        if drills:
            context_str = "\n\n".join(
                f"Drill ID: {d.get('id', 'N/A')}\n"
                f"Drill Name: {d['metadata'].get('name_ko') or d['metadata'].get('name', 'N/A') if language == 'ko' else d['metadata'].get('name', 'N/A')}\n"
                f"Phase: {d['metadata'].get('phase', 'N/A')}\n"
                f"Difficulty: {d['metadata'].get('difficulty', 'N/A')}\n"
                f"Suggested Duration: {d['metadata'].get('duration_min', 'N/A')} min\n"
                f"Description: {d['content']}"
                for d in drills
            )
        else:
            context_str = "No specific drills found in the database for this day."
        days_section += f"""
--- Day {day_key} ---
Focus Areas: {", ".join(focus)}
Available Drills from Database:
{context_str}
"""

    schema_json = json.dumps(WeeklyRoutineResponse.model_json_schema(), indent=2)

    warmup_min = max(1, int(available_time * 0.15))
    cooldown_min = max(1, int(available_time * 0.15))
    main_min = max(1, available_time - warmup_min - cooldown_min)

    prompt = f"""You are an expert basketball coach. The user received a weekly
training routine but wants changes based on their feedback.

**User Profile:**
- Skill Level: {skill_level}
- Training Days: {training_days}
- Available Time Per Day: {available_time} minutes
- Focus Areas: {json.dumps(user_info.get("focus_areas", []))}
- Available Equipment: {json.dumps(user_info.get("equipment", []))}

**Previous Response:**
{previous_response}

**User's Feedback:**
"{feedback}"

**Weekly Plan & Available Drills:**
{days_section}

**Language:**
Respond in {language_name}. All string fields must be in {language_name}.

**Instructions:**
1. Incorporate the user's feedback into a revised weekly routine.
2. Preserve all parts of the previous response that are NOT affected
   by the feedback. Only modify what the user explicitly requested.
   Preserve any personalized elements or free_text customizations from
   the user's original request unless the feedback explicitly asks to
   change them.
3. Each day must have exactly 3 phases:
   - "warmup": {warmup_min} min
   - "main": {main_min} min
   - "cooldown": {cooldown_min} min
4. The sum of all drill durations for each day MUST equal exactly
   {available_time} minutes.
5. MINIMIZE drill repetition across days.
6. Output a JSON object strictly following this schema:
{"" if language != "ko" else KO_BASKETBALL_TERMINOLOGY}

```json
{schema_json}
```

JSON Output:
"""
    try:
        response = chat_completion_with_retry(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Received an invalid or empty response from OpenAI API.")

        content = response.choices[0].message.content

        try:
            extracted_data = json.loads(content)
            validated = WeeklyRoutineResponse.model_validate(extracted_data)
            final_response_str = validated.model_dump_json(indent=2)
            logger.debug("Refined Weekly Response: %s", final_response_str)
            return {"final_response": final_response_str}
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                "Failed to parse or validate refined weekly LLM response: %s "
                "(response length: %d)",
                e,
                len(content),
            )
            raise ValueError(
                "LLM returned an invalid refined weekly routine"
            ) from e

    except openai.APIError as e:
        logger.error("OpenAI API error during weekly refinement: %s", e)
        raise ValueError(
            "Failed to refine weekly routine due to an API error."
        ) from e


def route_feedback(state: WeeklyRefineState) -> str:
    """Route based on feedback classification."""
    return state.get("feedback_type", "regenerate_only")


# Build the refine graph
workflow = StateGraph(WeeklyRefineState)

workflow.add_node("classify_feedback", classify_feedback)
workflow.add_node("re_retrieve", re_retrieve)
workflow.add_node("refine_generate", refine_generate)

workflow.set_entry_point("classify_feedback")
workflow.add_conditional_edges(
    "classify_feedback",
    route_feedback,
    {
        "re_retrieve": "re_retrieve",
        "regenerate_only": "refine_generate",
    },
)
workflow.add_edge("re_retrieve", "refine_generate")
workflow.add_edge("refine_generate", END)

weekly_coach_refine_graph = workflow.compile()
