"""RAG-based baseline for Skill Lab evaluation.

Simulates the old coach agent pipeline (diagnose → retrieve → generate)
by loading drills from drills.json, filtering by category and equipment,
and injecting them into the same prompt used by the current LLM-only agent.

This baseline mirrors the retrieval logic from the removed retrieve_drills node
(commit d8436c8) to enable a fair RAG vs LLM-only comparison.
"""

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from src.core.constants import KO_BASKETBALL_TERMINOLOGY
from src.models.skill_schema import SkillLabResponse
from src.utils.llm import chat_completion_with_retry

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
MAX_DRILLS = 10


def _load_drills() -> list:
    with open(DATA_DIR / "drills.json", encoding="utf-8") as f:
        return json.load(f)


def _filter_drills(drills: list, category: str, user_equipment: set) -> list:
    """Filter drills by category then by equipment availability.

    Mirrors the original retrieve_drills node logic:
    1. Category filter (same as ChromaDB where_filter)
    2. Equipment filter (required_equipment must be subset of user_equipment)
    3. Cap at MAX_DRILLS results
    """
    category_filtered = [d for d in drills if d.get("category") == category] if category else drills

    equipment_filtered = []
    for drill in category_filtered:
        required = set(drill.get("required_equipment", []))
        if not required or required.issubset(user_equipment):
            equipment_filtered.append(drill)

    return equipment_filtered[:MAX_DRILLS]


def _build_context_str(drills: list, language: str) -> str:
    """Build context string in the same format as the old retrieve_drills node."""
    if not drills:
        return "No specific drills found in the database."

    lines = []
    for drill in drills:
        name = drill.get("name_ko") or drill.get("name", "N/A") if language == "ko" else drill.get("name", "N/A")
        required_equip = drill.get("required_equipment", [])
        equip_str = ", ".join(required_equip) if required_equip else "none"
        lines.append(
            f"Drill Name: {name}\n"
            f"Difficulty: {drill.get('difficulty', 'N/A')}\n"
            f"Suggested Duration: {drill.get('duration_min', 'N/A')} min\n"
            f"Required Equipment: {equip_str}\n"
            f"Description: {drill.get('description', 'N/A')}"
        )
    return "\n\n".join(lines)


def run_rag_baseline_skill(user_input: dict) -> SkillLabResponse:
    """Run skill breakdown with RAG-based drill injection (old pipeline).

    Args:
        user_input: Dict with skill_level, category, available_time_min,
                    equipment, language, and optional specific_skill.

    Returns:
        Validated SkillLabResponse.

    Raises:
        ValueError: If LLM response cannot be parsed/validated.
    """
    category = user_input.get("category", "")
    skill_level = user_input.get("skill_level", "intermediate")
    available_time = user_input.get("available_time_min", 20)
    specific_skill = user_input.get("specific_skill") or ""
    language = user_input.get("language", "en")
    user_equipment = set(user_input.get("equipment", []))

    all_drills = _load_drills()
    filtered_drills = _filter_drills(all_drills, category, user_equipment)
    context_str = _build_context_str(filtered_drills, language)

    logger.info(
        "RAG baseline: %d drills after filtering (category=%s, equipment=%s)",
        len(filtered_drills),
        category,
        user_equipment,
    )

    skill_instruction = (
        f'The user wants to master: "{specific_skill}".'
        if specific_skill
        else (
            f"The user did not specify a skill. Pick the most impactful "
            f"{category} technique for a {skill_level} player and set it as skill_name."
        )
    )

    schema_json = json.dumps(SkillLabResponse.model_json_schema(), indent=2)
    language_name = "Korean" if language == "ko" else "English"

    prompt = f"""You are an expert basketball skills coach who specializes in
breaking down individual techniques into progressive micro-steps.

**User Profile:**
- Skill Level: {skill_level}
- Category: {category}
- Available Time: {available_time} minutes
- Available Equipment: {list(user_equipment)}
- Additional Focus: None
- Intensity Preference: None
- Special Notes: None
- Additional Request (raw): None

**Skill Selection:**
{skill_instruction}

**Reference Drills from Database (use as inspiration):**
{context_str}

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
4. Each step must include a concrete target or success metric.
5. Set difficulty_level to a short phrase showing the progression range.
6. Write a motivating coach_message about mastering this specific skill.
7. Output a JSON object strictly following this schema:

```json
{schema_json}
```

JSON Output:
"""

    messages = []
    if language == "ko":
        messages.append({"role": "system", "content": KO_BASKETBALL_TERMINOLOGY})
    messages.append({"role": "user", "content": prompt})

    response = chat_completion_with_retry(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
    )

    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Empty response from LLM for RAG baseline skill.")

    content = response.choices[0].message.content
    try:
        return SkillLabResponse.model_validate_json(content)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("Failed to parse RAG baseline response: %s (raw: %.500s)", e, content)
        raise ValueError("RAG baseline LLM returned invalid response") from e
