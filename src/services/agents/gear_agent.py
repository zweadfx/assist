import json
import logging
import re
from typing import List, TypedDict

import openai
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.core.constants import SENSORY_TAG_MAP
from src.core.exceptions import BudgetInsufficientError
from src.models.gear_schema import GearAdvisorResponse
from src.services.rag.shoe_retrieval import shoe_retriever
from src.utils.llm import chat_completion_with_retry

logger = logging.getLogger(__name__)

_MAX_PREF_LENGTH = 100
_MAX_PLAYER_LENGTH = 100
_BLOCKED_PATTERNS = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions"
    r"|forget\s+(all\s+)?above"
    r"|you\s+are\s+now"
    r"|disregard\s+(all\s+)?prior",
    re.IGNORECASE,
)


def _sanitize_gear_text(text: str, max_length: int) -> str:
    """Strip control characters, injection patterns, and enforce max length."""
    text = text[:max_length]
    text = re.sub(r"[\r\n\t\x00-\x1f\x7f]", " ", text)
    text = _BLOCKED_PATTERNS.sub("", text)
    return text.strip()


class GearAgentState(TypedDict):
    """
    Represents the state of the GearAgent workflow. It holds all the data
    that is passed between nodes in the graph.
    """

    # The conversation history. The last message is the user's request.
    messages: List[BaseMessage]

    # Information about the user (e.g., sensory preferences, player archetype, budget).
    # This will be extracted from the user's request.
    user_info: dict

    # A list of relevant shoes and player archetypes retrieved from the RAG store.
    context: List[Document]

    # The final generated shoe recommendations in JSON format.
    final_response: str


def analyze_preferences(state: GearAgentState) -> dict:
    """
    Validates that the user_info is present and sanitizes all user-controlled
    string fields before they reach the prompt builder.
    """
    logger.info("NODE: Analyzing User Preferences")
    if not state.get("user_info"):
        raise ValueError("User info is missing from the state.")

    user_info = state["user_info"]
    raw_prefs = user_info.get("sensory_preferences")
    if not raw_prefs:
        raise ValueError("Sensory preferences are required for gear recommendations.")

    # Sanitize each sensory preference item
    sanitized_prefs = [
        _sanitize_gear_text(p, _MAX_PREF_LENGTH)
        for p in raw_prefs
        if isinstance(p, str)
    ]
    sanitized_prefs = [p for p in sanitized_prefs if p]
    if not sanitized_prefs:
        raise ValueError("All sensory preferences were empty after sanitization.")

    # Sanitize optional free-text fields
    raw_player = user_info.get("player_archetype")
    sanitized_player = (
        _sanitize_gear_text(raw_player, _MAX_PLAYER_LENGTH)
        if isinstance(raw_player, str)
        else None
    )

    sanitized_info = {
        **user_info,
        "sensory_preferences": sanitized_prefs,
        "player_archetype": sanitized_player or None,
    }

    logger.debug(f"User Info (sanitized): {sanitized_info}")
    return {"user_info": sanitized_info}


def retrieve_shoes_and_players(state: GearAgentState) -> dict:
    """
    Retrieves relevant shoes and player archetypes using the ShoeRetriever.
    Uses cross-analysis search combining sensory preferences and player archetype.
    """
    logger.info("NODE: Retrieving Shoes and Players")
    user_info = state["user_info"]
    sensory_preferences = user_info.get("sensory_preferences", [])
    player_archetype = user_info.get("player_archetype")
    budget_max_krw = user_info.get("budget_max_krw")
    position = user_info.get("position")

    logger.debug(
        f"Search params: sensory={sensory_preferences}, "
        f"player={player_archetype}, budget={budget_max_krw}, position={position}"
    )

    try:
        # Use ShoeRetriever's cross-analysis search
        search_results = shoe_retriever.cross_analysis_search(
            sensory_keywords=sensory_preferences,
            player_archetype=player_archetype,
            budget_max_krw=budget_max_krw,
            position=position,
            n_shoes=5,
        )

        # Raise budget error only when retrieval confirmed budget was the cause
        if search_results.get("empty_reason") == "budget":
            raise BudgetInsufficientError(
                f"No shoes available within budget of {budget_max_krw:,} KRW"
            )

        # Combine shoes and players into context
        context_docs = search_results["players"] + search_results["shoes"]

        logger.info(
            f"Retrieved {len(search_results['shoes'])} shoes, "
            f"{len(search_results['players'])} players"
        )
        return {"context": context_docs}

    except BudgetInsufficientError:
        raise
    except Exception as e:
        logger.exception("Failed to retrieve shoes and players from RAG")
        # Re-raise the exception to prevent hallucinations with empty context
        raise ValueError("Failed to retrieve shoe recommendations from database") from e


def generate_recommendations(state: GearAgentState) -> dict:
    """
    Generates the final shoe recommendations by synthesizing the user's
    preferences and the retrieved shoes/players using an LLM.
    """
    logger.info("NODE: Generating Recommendations")
    user_info = state["user_info"]
    context_docs = state["context"]

    # Separate shoes and players from context (mutually exclusive)
    # Prefer explicit doc_type; fall back to field presence for legacy data
    shoe_docs = []
    player_docs = []
    for doc in context_docs:
        meta = doc.metadata
        doc_type = meta.get("doc_type")
        if doc_type == "shoe":
            shoe_docs.append(doc)
        elif doc_type == "player":
            player_docs.append(doc)
        elif meta.get("brand"):
            shoe_docs.append(doc)
        elif meta.get("play_style"):
            player_docs.append(doc)

    # Resolve language early so it's available for shoes context construction
    language = user_info.get("language", "en")
    language_name = "Korean" if language == "ko" else "English"

    # Prepare shoes context string
    def _sensory_tags(doc: Document, language: str) -> str:
        """Convert sensory tag enums to the appropriate language for the prompt."""
        if language == "ko":
            tags_kr = doc.metadata.get("sensory_tags_kr")
            if tags_kr:
                return tags_kr
        tags_en = doc.metadata.get("sensory_tags", "")
        if tags_en:
            en_list = [t.strip() for t in tags_en.split(",") if t.strip()]
            if language == "ko":
                return ", ".join(SENSORY_TAG_MAP.get(t, t) for t in en_list)
            return ", ".join(en_list)
        return "N/A"

    shoes_context_str = "\n\n".join(
        [
            f"Brand: {doc.metadata.get('brand', 'N/A')}\n"
            f"Model: {doc.metadata.get('model_name', 'N/A')}\n"
            f"Price: {doc.metadata.get('price_krw', 'N/A')} KRW\n"
            f"Sensory Tags: {_sensory_tags(doc, language)}\n"
            f"Player Signature: {doc.metadata.get('player_signature', 'N/A')}\n"
            f"Description: {doc.page_content}"
            for doc in shoe_docs
        ]
    )
    if not shoe_docs:
        raise ValueError("No shoe data available to generate recommendations")
    logger.info(
        "Shoes context for prompt (%d docs): %.500s", len(shoe_docs), shoes_context_str
    )

    # Prepare player context string
    players_context_str = ""
    if player_docs:
        players_context_str = "\n\n".join(
            [
                f"Player: {doc.metadata.get('name', 'N/A')}\n"
                f"Position: {doc.metadata.get('position', 'N/A')}\n"
                f"Play Style: {doc.metadata.get('play_style', 'N/A')}\n"
                f"Description: {doc.page_content}"
                for doc in player_docs
            ]
        )

    # Prepare the JSON schema for the prompt
    schema_json = json.dumps(GearAdvisorResponse.model_json_schema(), indent=2)

    # Build player section separately to avoid nested f-string
    player_section = ""
    if players_context_str:
        player_section = f"**Player Archetype Information:**\n{players_context_str}\n\n"

    prompt = f"""
You are an expert basketball gear advisor. Your task is to generate personalized
shoe recommendations based on the user's preferences and the available shoe data.

**User Preferences:**
- Sensory Preferences: {user_info.get("sensory_preferences")}
- Player Archetype: {user_info.get("player_archetype", "Not specified")}
- Position: {user_info.get("position", "Not specified")}
- Budget: {user_info.get("budget_max_krw", "No limit")} KRW

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
            # Validate the data with the Pydantic model
            validated_response = GearAdvisorResponse.model_validate(extracted_data)
            final_response_str = validated_response.model_dump_json(indent=2)
            logger.debug(f"Generated Response: {final_response_str}")
            return {"final_response": final_response_str}
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                "Failed to parse or validate LLM response for "
                "recommendations: %s (raw content: %.500s)",
                e,
                content,
            )
            raise ValueError("LLM returned an invalid recommendations object") from e

    except openai.APIError as e:
        logger.error(f"OpenAI API error during recommendations generation: {e}")
        raise ValueError(
            "Failed to generate recommendations due to an API error."
        ) from e
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during recommendations generation: {e}"
        )
        raise


# Define the graph workflow
workflow = StateGraph(GearAgentState)

# Add nodes to the graph
workflow.add_node("analyze", analyze_preferences)
workflow.add_node("retrieve", retrieve_shoes_and_players)
workflow.add_node("generate", generate_recommendations)

# Define the edges for the graph
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph into a runnable object
gear_agent_graph = workflow.compile()
