import json
import logging
from typing import List, TypedDict

import openai
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.core.constants import KO_BASKETBALL_TERMINOLOGY
from src.models.weekly_schema import WeeklyRoutineResponse
from src.services.agents.coach_agent import _parse_free_text
from src.utils.llm import chat_completion_with_retry

logger = logging.getLogger(__name__)


class WeeklyCoachState(TypedDict):
    """State for the weekly coach agent workflow."""

    messages: List[BaseMessage]
    user_info: dict
    week_plan: dict  # plan_week node result: day-to-focus mapping
    final_response: str


def diagnose_user_state(state: WeeklyCoachState) -> dict:
    """Validates user_info and enriches it with parsed free_text."""
    logger.info("WEEKLY NODE: Diagnosing User State")
    if not state.get("user_info"):
        raise ValueError("User info is missing from the state.")

    user_info = {**state["user_info"]}

    free_text = user_info.get("free_text")
    if free_text and free_text.strip():
        logger.info("Parsing free_text input: length=%d", len(free_text))
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

    return {"user_info": user_info}


def plan_week(state: WeeklyCoachState) -> dict:
    """Uses LLM to distribute focus areas across training days."""
    logger.info("WEEKLY NODE: Planning Week")
    user_info = state["user_info"]
    training_days = user_info.get("training_days", 3)
    focus_areas = user_info.get("focus_areas") or []
    if not isinstance(focus_areas, list) or not focus_areas:
        raise ValueError("focus_areas must be a non-empty list.")
    skill_level = user_info.get("skill_level", "intermediate")

    additional_focus = user_info.get("additional_focus", "")
    intensity_pref = user_info.get("intensity_preference", "")
    special_notes = user_info.get("special_notes", "")
    free_text = user_info.get("free_text", "")

    prompt = f"""You are an expert basketball training planner.

Given the following training parameters, create a weekly plan that distributes focus areas across training days.

**Parameters:**
- Training Days: {training_days}
- Focus Areas: {json.dumps(focus_areas)}
- Skill Level: {skill_level}
- Additional Focus: {additional_focus or "None"}
- Intensity Preference: {intensity_pref or "None"}
- Special Notes: {special_notes or "None"}
- Free Text Request: {free_text or "None"}

**Rules:**
1. Each day should have 1-2 focus areas from the provided list.
2. Distribute focus areas as evenly as possible across the week.
3. Avoid scheduling high-intensity focus areas on consecutive days (allow recovery).
4. If conditioning is a focus area, pair it with a skill-based focus when possible.
5. Consider muscle recovery: don't put defense-heavy and conditioning-heavy days back-to-back.

Output a JSON object where keys are day numbers (as strings "1", "2", etc.) and values are arrays of focus area strings.

Example for 3 days with ["shooting", "dribble", "conditioning"]:
{{"1": ["shooting"], "2": ["dribble", "conditioning"], "3": ["shooting", "conditioning"]}}

JSON Output:"""

    try:
        response = chat_completion_with_retry(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty response from LLM for week planning")

        content = response.choices[0].message.content
        week_plan = json.loads(content)

        # Validate: ensure all days are present and focus areas are valid
        valid_areas = {"dribble", "shooting", "defense", "conditioning"}
        validated_plan = {}
        for day in range(1, training_days + 1):
            day_key = str(day)
            day_focus = week_plan.get(day_key, focus_areas[:1])
            validated_plan[day_key] = [
                f for f in day_focus if f in valid_areas
            ] or focus_areas[:1]

        logger.info("Week plan: %s", validated_plan)
        return {"week_plan": validated_plan}

    except Exception as e:
        logger.warning("Failed to plan week via LLM, using fallback: %s", e)
        # Fallback: round-robin distribution
        fallback_plan = {}
        for day in range(1, training_days + 1):
            idx = (day - 1) % len(focus_areas)
            fallback_plan[str(day)] = [focus_areas[idx]]
        return {"week_plan": fallback_plan}


def generate_weekly_routine(state: WeeklyCoachState) -> dict:
    """Generates the complete weekly routine using LLM."""
    logger.info("WEEKLY NODE: Generating Weekly Routine")
    user_info = state["user_info"]
    week_plan = state["week_plan"]

    training_days = user_info.get("training_days", 3)
    available_time = user_info.get("available_time_per_day_min", 60)
    skill_level = user_info.get("skill_level", "intermediate")
    language = user_info.get("language", "en")
    language_name = "Korean" if language == "ko" else "English"

    warmup_min = max(1, int(available_time * 0.15))
    cooldown_min = max(1, int(available_time * 0.15))
    main_min = max(1, available_time - warmup_min - cooldown_min)

    # Build the full day plans section for the prompt
    days_section = ""
    for day_key in sorted(week_plan.keys(), key=int):
        focus = week_plan[day_key]
        days_section += f"""
--- Day {day_key} ---
Focus Areas: {", ".join(focus)}
"""

    schema_json = json.dumps(WeeklyRoutineResponse.model_json_schema(), indent=2)

    prompt = f"""You are an expert basketball coach designing a comprehensive weekly training program.

**User Profile:**
- Skill Level: {skill_level}
- Training Days: {training_days}
- Available Time Per Day: {available_time} minutes
- Focus Areas: {json.dumps(user_info.get("focus_areas", []))}
- Available Equipment: {json.dumps(user_info.get("equipment", []))}
- Additional Focus: {user_info.get("additional_focus") or "None"}
- Intensity Preference: {user_info.get("intensity_preference") or "None"}
- Special Notes: {user_info.get("special_notes") or "None"}
- Free Text Request: {user_info.get("free_text") or "None"}

**Weekly Plan:**
{days_section}

**Language:**
Respond in {language_name}. All string fields must be written in {language_name}.

**Instructions:**
1. Create a weekly routine with {training_days} days.
2. Each day must have exactly 3 phases:
   - "warmup": {warmup_min} min
   - "main": {main_min} min
   - "cooldown": {cooldown_min} min
3. The sum of all drill durations for each day MUST equal exactly {available_time} minutes.
4. Design creative, diverse drills tailored to the user's skill level and equipment.
5. MINIMIZE drill repetition across days. Each day should feel fresh with different drills.
6. Each drill needs:
   - drill_id: Use IDs like "day1-warmup-1", "day2-main-2"
   - name: Drill name
   - duration_min: Duration fitting the phase allocation
   - description: Step-by-step execution with specific reps, sets, or distance targets (3-4 sentences minimum)
   - coaching_tip: A practical tip tailored to {skill_level} level
7. **Drill Quality Standards (STRICTLY ENFORCE):**
   - Every drill must be directly applicable to real game situations.
     Prefer: movement-based drills, combination moves, game-speed reps, competitive formats.
     Avoid: purely stationary isolated reps with no game context, drills that only train
     a weakness in isolation without connecting it to a game action (e.g., do NOT generate
     "weak hand free throw" — instead use "weak hand finishing at the rim off a drive" or
     "weak hand dribble penetration to layup").
   - Each drill must include a concrete target or success metric
     (e.g., "10 made shots", "3 sets of 30 seconds", "5 consecutive reps").
   - Use only real, established basketball drill names. Do not invent or combine words
     into drill names that do not exist (e.g., never generate names like "새스트랩 드리블").
     If combining multiple moves, use standard terms like "컴비네이션 드리블" or
     "콤보 드리블".
8. Create a meaningful day_label for each day reflecting its focus (e.g., "Day 1 - Shooting Focus" or "1일차 - 슈팅 집중" in Korean).
9. Write a weekly_title that captures the overall training theme.
10. Write a coach_overview with strategic advice for the week (recovery tips, progression notes, motivation).
11. If "Free Text Request" is provided, actively reflect its content throughout
    the routine — incorporate the requested elements into drill descriptions,
    coaching_tips, day_labels, and coach_overview. Do not treat it as optional context.

**Output a JSON object strictly following this schema:**
```json
{schema_json}
```

JSON Output:"""

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
            validated = WeeklyRoutineResponse.model_validate(extracted_data)
            final_response_str = validated.model_dump_json(indent=2)
            logger.debug("Generated Weekly Response: %s", final_response_str)
            return {"final_response": final_response_str}
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                "Failed to parse or validate LLM response for weekly routine: "
                "%s (response length: %d)",
                e,
                len(content),
            )
            raise ValueError("LLM returned an invalid weekly routine object") from e

    except openai.APIError as e:
        logger.error("OpenAI API error during weekly routine generation: %s", e)
        raise ValueError(
            "Failed to generate weekly routine due to an API error."
        ) from e
    except Exception as e:
        logger.error(
            "An unexpected error occurred during weekly routine generation: %s",
            e,
        )
        raise


# Define the graph workflow
workflow = StateGraph(WeeklyCoachState)

# Add nodes
workflow.add_node("diagnose", diagnose_user_state)
workflow.add_node("plan_week", plan_week)
workflow.add_node("generate", generate_weekly_routine)

# Define edges: diagnose → plan_week → generate → END
workflow.set_entry_point("diagnose")
workflow.add_edge("diagnose", "plan_week")
workflow.add_edge("plan_week", "generate")
workflow.add_edge("generate", END)

# Compile the graph
weekly_coach_agent_graph = workflow.compile()
