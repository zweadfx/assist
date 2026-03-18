"""No-RAG baseline for Gear Advisor evaluation.

Instead of RAG retrieval, injects all 59 shoes (and matched player)
directly into the prompt. Uses the same LLM, prompt template, and
response schema as the RAG pipeline for fair comparison.
"""

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from src.core.constants import SENSORY_TAG_MAP
from src.models.gear_schema import GearAdvisorResponse
from src.utils.llm import chat_completion_with_retry

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def _load_json(filename: str) -> list:
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def _sensory_tags_str(shoe: dict, language: str) -> str:
    """Convert sensory tags to the appropriate language string."""
    if language == "ko":
        tags_kr = shoe.get("sensory_tags_kr")
        if tags_kr:
            return ", ".join(tags_kr)
        return ", ".join(
            SENSORY_TAG_MAP.get(t, t) for t in shoe.get("sensory_tags", [])
        )
    return ", ".join(shoe.get("sensory_tags", []))


def _build_shoes_context(shoes: list, language: str) -> str:
    """Build context string from all shoes, identical format to gear_agent."""
    return "\n\n".join(
        f"Shoe ID: {s.get('id', 'N/A')}\n"
        f"Brand: {s.get('brand', 'N/A')}\n"
        f"Model: {s.get('model_name', 'N/A')}\n"
        f"Price: {s.get('price_krw', 'N/A')} KRW\n"
        f"Sensory Tags: {_sensory_tags_str(s, language)}\n"
        f"Player Signature: {s.get('player_signature', 'N/A')}\n"
        f"Description: {s.get('description', 'N/A')}"
        for s in shoes
    )


def _build_player_context(players: list, archetype: str) -> str:
    """Find matching player and build context string."""
    if not archetype:
        return ""

    archetype_lower = archetype.lower()
    for p in players:
        if archetype_lower in p.get("name", "").lower():
            play_style = ", ".join(p.get("play_style", []))
            return (
                f"Player: {p.get('name', 'N/A')}\n"
                f"Position: {p.get('position', 'N/A')}\n"
                f"Play Style: {play_style}\n"
                f"Description: {p.get('description', 'N/A')}"
            )
    return ""


def run_no_rag_gear(user_input: dict) -> GearAdvisorResponse:
    """Run gear recommendation without RAG retrieval.

    Loads all shoes/players from JSON and injects them into the same
    prompt template used by gear_agent.py.

    Args:
        user_input: Dict with sensory_preferences, player_archetype,
                    position, budget_max_krw, language.

    Returns:
        Validated GearAdvisorResponse.

    Raises:
        ValueError: If LLM response cannot be parsed/validated.
    """
    all_shoes = _load_json("shoes.json")
    all_players = _load_json("players.json")

    language = user_input.get("language", "en")
    language_name = "Korean" if language == "ko" else "English"

    shoes_context_str = _build_shoes_context(all_shoes, language)

    archetype = user_input.get("player_archetype") or ""
    players_context_str = _build_player_context(all_players, archetype)

    player_section = ""
    if players_context_str:
        player_section = (
            f"**Player Archetype Information:**\n{players_context_str}\n\n"
        )

    schema_json = json.dumps(
        GearAdvisorResponse.model_json_schema(), indent=2
    )

    prompt = f"""
You are an expert basketball gear advisor. Your task is to generate personalized
shoe recommendations based on the user's preferences and the available shoe data.

**User Preferences:**
- Sensory Preferences: {user_input.get("sensory_preferences")}
- Player Archetype: {user_input.get("player_archetype", "Not specified")}
- Position: {user_input.get("position", "Not specified")}
- Budget: {user_input.get("budget_max_krw", "No limit")} KRW

{player_section}**Available Shoes Data:**
{shoes_context_str}

**Language:**
Respond in {language_name}. All string fields (recommendation_title, user_profile_summary, ai_reasoning, recommendation_reason, sensory_tags) must be written in {language_name}.

**Critical Rule:**
You MUST ONLY use shoes from the "Available Shoes Data" above. Do NOT invent or
fabricate any shoe. Every brand, model_name, price_krw, and sensory_tags
in your response MUST exactly match the provided data.

**Instructions:**
1. Recommend at least 1 and up to 5 shoes from the provided data above.
   Never return an empty shoes list. If no shoes perfectly match, recommend the
   closest alternatives from the available data.
2. Calculate a match_score (0-100) for each shoe based on:
   - Sensory tag overlap with user preferences (primary factor)
   - Player archetype compatibility (if specified)
   - Position suitability (if specified)
   - Budget fit (if specified)
3. Write a compelling recommendation_reason for each shoe explaining why it's a good
   match.
4. Provide an overall ai_reasoning explaining your recommendation strategy.
5. Create a catchy recommendation_title for the set.
6. Summarize the user's profile in user_profile_summary.
7. Your final output **must** be a JSON object that strictly follows this Pydantic
   schema:

```json
{schema_json}
```

JSON Output:
"""

    response = chat_completion_with_retry(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Empty response from LLM for baseline gear.")

    content = response.choices[0].message.content

    try:
        extracted_data = json.loads(content)
        validated = GearAdvisorResponse.model_validate(extracted_data)
        return validated
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(
            "Failed to parse baseline response: %s (raw: %.500s)",
            e,
            content,
        )
        raise ValueError("Baseline LLM returned invalid response") from e
