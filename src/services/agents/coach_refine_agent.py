"""
Coach Refine Agent: refines a previously generated skill breakdown
based on user feedback, using a LangGraph conditional-edge loop.

Graph:
    classify_feedback ──┬── re_retrieve ──→ refine_generate ──→ END
                        └─────────────────→ refine_generate ──→ END
"""

import json
import logging
from typing import List, TypedDict

import openai
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.services.agents.coach_agent import SkillBreakdownCard, retrieve_drills
from src.utils.llm import chat_completion_with_retry

logger = logging.getLogger(__name__)


class CoachRefineState(TypedDict):
    """State for the coach refine workflow."""

    messages: List[BaseMessage]
    user_info: dict
    context: List[Document]
    previous_response: str
    feedback: str
    feedback_type: str
    final_response: str


def classify_feedback(state: CoachRefineState) -> dict:
    """Classify feedback as needing re-retrieval or regeneration only."""
    logger.info("REFINE NODE: Classifying Feedback")
    feedback = state["feedback"]

    prompt = f"""You are a classifier for a basketball training app.
The user received a training routine and gave feedback. Classify the feedback:

- "re_retrieve": feedback asks for DIFFERENT drills, exercises, or content
  (e.g. "다른 드릴로 바꿔줘", "different exercises", "swap the drill")
- "regenerate_only": feedback asks for ADJUSTMENTS to existing content
  (e.g. "난이도 낮춰줘", "시간 줄여줘", "make it easier", "more detailed description")

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


def re_retrieve(state: CoachRefineState) -> dict:
    """Re-retrieve drills from ChromaDB using existing retrieve_drills."""
    logger.info("REFINE NODE: Re-retrieving Drills")
    return retrieve_drills(state)


def refine_generate(state: CoachRefineState) -> dict:
    """Generate a refined skill breakdown based on feedback."""
    logger.info("REFINE NODE: Generating Refined Skill Breakdown")
    user_info = state["user_info"]
    previous_response = state["previous_response"]
    feedback = state["feedback"]
    context_docs = state.get("context", [])

    language = user_info.get("language", "en")
    language_name = "Korean" if language == "ko" else "English"

    context_str = "\n\n".join(
        f"Drill Name: {doc.metadata.get('name_ko') or doc.metadata.get('name', 'N/A') if language == 'ko' else doc.metadata.get('name', 'N/A')}\n"
        f"Difficulty: {doc.metadata.get('difficulty', 'N/A')}\n"
        f"Suggested Duration: {doc.metadata.get('duration_min', 'N/A')} min\n"
        f"Required Equipment: {doc.metadata.get('required_equipment', 'none')}\n"
        f"Description: {doc.page_content}"
        for doc in context_docs
    )
    if not context_str:
        context_str = "No specific drills found in the database."

    schema_json = json.dumps(SkillBreakdownCard.model_json_schema(), indent=2)
    available_time = user_info.get("available_time_min", 20)

    prompt = f"""You are an expert basketball skills coach. The user received
a skill breakdown but wants changes based on their feedback.

**User Profile:**
- Skill Level: {user_info.get("skill_level", "intermediate")}
- Category: {user_info.get("category", "")}
- Available Time: {available_time} minutes
- Available Equipment: {user_info.get("equipment")}

**Previous Response:**
{previous_response}

**User's Feedback:**
"{feedback}"

**Reference Drills from Database (use as inspiration):**
{context_str}

**Language:**
Respond in {language_name}. All string fields must be in {language_name}.
{"" if language != "ko" else '''
**한국어 농구 용어 규칙:**
- 한국 농구에서 실제로 통용되는 표현을 사용할 것
- 영어 기술명은 한글 음차 그대로 표기 (예: crossover → 크로스오버, behind the back → 비하인드 더 백, step-back → 스텝백)
- 번역하면 어색한 용어는 번역하지 말 것 (예: weak hand → "약손" ✓, "비우수 손" ✗)
- figure-8 → 8자 드리블, floater → 플로터, euro step → 유로스텝
- Reference Drills의 Drill Name이 이미 한국어로 제공된 경우 그대로 사용할 것
'''}
**Instructions:**
1. Incorporate the user's feedback into a revised skill breakdown.
2. Preserve all parts of the previous response that are NOT affected
   by the feedback. Only modify what the user explicitly requested.
3. The sum of all step durations MUST equal exactly {available_time} minutes.
4. Each step must have: name, duration_min, description, focus_point,
   success_criteria.
5. Keep the progressive structure (simplest → most game-like).
6. Output a JSON object strictly following this schema:

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
            validated = SkillBreakdownCard.model_validate(extracted_data)
            final_response_str = validated.model_dump_json(indent=2)
            logger.debug("Refined Response: %s", final_response_str)
            return {"final_response": final_response_str}
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                "Failed to parse or validate refined LLM response: %s "
                "(raw content: %.500s)",
                e,
                content,
            )
            raise ValueError(
                "LLM returned an invalid refined skill breakdown"
            ) from e

    except openai.APIError as e:
        logger.error("OpenAI API error during skill refinement: %s", e)
        raise ValueError(
            "Failed to refine skill breakdown due to an API error."
        ) from e


def route_feedback(state: CoachRefineState) -> str:
    """Route based on feedback classification."""
    return state.get("feedback_type", "regenerate_only")


# Build the refine graph
workflow = StateGraph(CoachRefineState)

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

coach_refine_graph = workflow.compile()
