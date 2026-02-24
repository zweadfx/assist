import json
import logging
from typing import List, TypedDict

import openai
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from src.models.skill_schema import Drill
from src.services.rag.chroma_db import chroma_manager
from src.services.rag.embedding import client as openai_client

logger = logging.getLogger(__name__)


class CoachAgentState(TypedDict):
    """
    Represents the state of the CoachAgent workflow. It holds all the data
    that is passed between nodes in the graph.
    """

    # The conversation history. The last message is the user's request.
    messages: List[BaseMessage]

    # Information about the user (e.g., skill level, available time, equipment).
    # This will be extracted from the user's request.
    user_info: dict

    # A list of relevant drills retrieved from the RAG store.
    context: List[Document]

    # The final generated "Daily Routine Card" in JSON format.
    final_response: str


class UserDrillPreferences(BaseModel):
    """
    A model to hold the structured user preferences for a drill session,
    extracted from their natural language request.
    """

    focus_area: str = Field(
        description=(
            "The primary basketball skill the user wants to improve, e.g., "
            "'dribbling', 'shooting'."
        )
    )
    available_time_min: int = Field(
        description="The total available time for the training session in minutes."
    )
    equipment: List[str] = Field(
        description=(
            "A list of equipment the user has available, e.g., ['ball', 'hoop', "
            "'cones']."
        )
    )


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
        response = openai_client.chat.completions.create(
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

    # Parse free_text if provided
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


def retrieve_drills(state: CoachAgentState) -> dict:
    """
    Retrieves relevant drills from the vector store based on user_info. This
    node performs a semantic search and then post-filters the results based
    on the user's available equipment.
    """
    logger.info("NODE: Retrieving Drills")
    user_info = state["user_info"]
    focus_area = user_info.get("focus_area", "")
    skill_level = user_info.get("skill_level", "")
    user_equipment = set(user_info.get("equipment", []))

    # Build enriched query with user context for better semantic matching
    level_phrase = f"{skill_level} " if skill_level else ""
    equipment_str = (
        ", ".join(sorted(user_equipment)) if user_equipment else "no equipment"
    )
    query_text = (
        f"A {level_phrase}basketball drill focusing on improving "
        f"{focus_area} skills using {equipment_str}."
    )
    logger.info("Querying for drills related to: %s", focus_area)

    unfiltered_docs = []
    try:
        # Build metadata filter for DB-level pre-filtering
        where_filter = {"category": focus_area} if focus_area else None

        # Retrieve candidates with DB-level category filtering
        results = chroma_manager.query_drills(
            query_texts=[query_text], n_results=10, where=where_filter
        )
        if results and results.get("documents"):
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            for i, doc_content in enumerate(documents):
                doc = Document(page_content=doc_content, metadata=metadatas[i])
                unfiltered_docs.append(doc)
            logger.info("Retrieved %d candidate drills", len(unfiltered_docs))
        else:
            logger.warning("No drills retrieved from DB")
    except Exception as e:
        logger.exception("Failed to retrieve drills from RAG")
        raise ValueError("Failed to retrieve drills from database") from e

    # Post-filter the results based on available equipment
    filtered_docs = []
    for doc in unfiltered_docs:
        required_equipment_str = doc.metadata.get("required_equipment", "")
        if (
            not required_equipment_str
        ):  # If no equipment is required, it's a valid drill
            filtered_docs.append(doc)
            continue

        required_equipment = set(required_equipment_str.split(","))
        if required_equipment.issubset(user_equipment):
            filtered_docs.append(doc)

    logger.info("Filtered down to %d drills based on equipment", len(filtered_docs))
    return {"context": filtered_docs}


class DailyRoutineCard(BaseModel):
    """Data model for the final daily routine card output."""

    routine_title: str = Field(
        description="A catchy and relevant title for the routine."
    )
    total_duration_min: int = Field(
        description="The total calculated duration for the entire routine in minutes.",
        gt=0,
    )
    coach_message: str = Field(
        description="A personalized, encouraging message from the AI coach."
    )
    drills: List[Drill]


def generate_routine(state: CoachAgentState) -> dict:
    """
    Generates the final "Daily Routine Card" by synthesizing the user's
    preferences and the retrieved drills using an LLM.
    """
    logger.info("NODE: Generating Routine")
    user_info = state["user_info"]
    context_docs = state["context"]

    # Prepare context string from retrieved documents with full metadata
    context_str = "\n\n".join(
        [
            f"Drill Name: {doc.metadata.get('name', 'N/A')}\n"
            f"Phase: {doc.metadata.get('phase', 'N/A')}\n"
            f"Difficulty: {doc.metadata.get('difficulty', 'N/A')}\n"
            f"Suggested Duration: {doc.metadata.get('duration_min', 'N/A')} min\n"
            f"Required Equipment: {doc.metadata.get('required_equipment', 'none')}\n"
            f"Description: {doc.page_content}"
            for doc in context_docs
        ]
    )
    if not context_str:
        context_str = "No specific drills found in the database."

    # Prepare the JSON schema for the prompt to ensure valid JSON output.
    schema_json = json.dumps(DailyRoutineCard.model_json_schema(), indent=2)

    language = user_info.get("language", "en")
    language_name = "Korean" if language == "ko" else "English"

    available_time = user_info.get("available_time_min", 30)

    prompt = f"""
    You are an expert basketball coach with years of experience designing
    effective training programs for players of all levels.

    **User Profile:**
    - Skill Level: {user_info.get("skill_level", "intermediate")}
    - Skill to Improve: {user_info.get("focus_area")}
    - Available Time: {available_time} minutes
    - Available Equipment: {user_info.get("equipment")}
    - Additional Focus: {user_info.get("additional_focus") or "None"}
    - Intensity Preference: {user_info.get("intensity_preference") or "None"}
    - Special Notes: {user_info.get("special_notes") or "None"}
    - Additional Request (raw): {user_info.get("free_text") or "None"}

    **Retrieved Drills from Database:**
    {context_str}

    **Language:**
    Respond in {language_name}. All string fields (routine_title, coach_message,
    drill name, description, coaching_tip) must be written in {language_name}.

    **Instructions:**
    1. Design a realistic routine with exactly 3 phases:
       - "warmup": ~15% of total time ({round(available_time * 0.15)} min).
         Light movement, stretching, or low-intensity ball handling.
       - "main": ~70% of total time ({round(available_time * 0.70)} min).
         Core skill-building drills matching the user's focus area.
         Include 2-4 drills with varied intensity.
       - "cooldown": ~15% of total time ({round(available_time * 0.15)} min).
         Stretching, free throws, or light shooting to wind down.
    2. Prioritize drills from "Retrieved Drills" that match the user's skill
       level. For beginners, use simpler drills with clear steps. For advanced
       players, add complexity and game-like scenarios.
    3. The sum of all drill durations MUST equal exactly {available_time} minutes.
    4. Each drill must have:
       - A unique drill_id (e.g., "warmup-1", "main-1", "cooldown-1")
       - A clear, actionable description (2-3 sentences explaining how to
         perform the drill with specific reps, sets, or targets)
       - A practical coaching_tip tailored to the user's skill level
         (technique cue, common mistake to avoid, or progression suggestion)
    5. If "Additional Request" is provided, incorporate the user's specific
       preferences or goals into drill selection and coaching tips.
    6. Write a motivating coach_message that references the user's specific
       focus area and encourages consistent practice.
    7. Create a descriptive routine_title that reflects the focus area and
       intensity level.
    8. Output a JSON object strictly following this schema:

    ```json
    {schema_json}
    ```

    JSON Output:
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Received an invalid or empty response from OpenAI API.")

        content = response.choices[0].message.content

        try:
            extracted_data = json.loads(content)
            # Validate the data with the Pydantic model
            validated_routine = DailyRoutineCard.model_validate(extracted_data)
            final_response_str = validated_routine.model_dump_json(indent=2)
            logger.debug("Generated Response: %s", final_response_str)
            return {"final_response": final_response_str}
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("Failed to parse or validate LLM response for routine: %s", e)
            raise ValueError(
                f"LLM returned an invalid routine object: {content}"
            ) from e

    except openai.APIError as e:
        logger.error("OpenAI API error during routine generation: %s", e)
        raise ValueError("Failed to generate routine due to an API error.") from e
    except Exception as e:
        logger.error("An unexpected error occurred during routine generation: %s", e)
        raise


# Define the graph workflow
workflow = StateGraph(CoachAgentState)

# Add nodes to the graph
workflow.add_node("diagnose", diagnose_user_state)
workflow.add_node("retrieve", retrieve_drills)
workflow.add_node("generate", generate_routine)

# Define the edges for the graph
workflow.set_entry_point("diagnose")
workflow.add_edge("diagnose", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph into a runnable object
coach_agent_graph = workflow.compile()
