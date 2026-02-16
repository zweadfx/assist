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


def diagnose_user_state(state: CoachAgentState) -> dict:
    """
    Validates that the user_info is present in the state. In a more complex
    scenario, this node could be used to further refine or validate the user
    profile. For now, it's a pass-through and logging step.
    """
    logger.info("NODE: Diagnosing User State")
    if not state.get("user_info"):
        raise ValueError("User info is missing from the state.")

    logger.debug("User Info: %s", state["user_info"])
    # The user_info is already structured and passed in the initial call
    # so we just pass it along to the next node.
    return {"user_info": state["user_info"]}


def retrieve_drills(state: CoachAgentState) -> dict:
    """
    Retrieves relevant drills from the vector store based on user_info. This
    node performs a semantic search and then post-filters the results based
    on the user's available equipment.
    """
    logger.info("NODE: Retrieving Drills")
    user_info = state["user_info"]
    focus_area = user_info.get("focus_area", "")
    user_equipment = set(user_info.get("equipment", []))
    query_text = f"A basketball drill focusing on improving {focus_area} skills."
    logger.info("Querying for drills related to: %s", focus_area)

    unfiltered_docs = []
    try:
        # Retrieve a larger pool of candidates for filtering
        results = chroma_manager.query_drills(query_texts=[query_text], n_results=10)
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
        logger.error("An error occurred during drill retrieval: %s", e)
        return {"context": []}

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

    # Prepare context string from retrieved documents
    context_str = "\n\n".join(
        [
            f"Drill Name: {doc.metadata.get('name', 'N/A')}\n"
            f"Description: {doc.page_content}"
            for doc in context_docs
        ]
    )
    if not context_str:
        context_str = "No specific drills found in the database."

    # Prepare the JSON schema for the prompt to ensure valid JSON output.
    schema_json = json.dumps(DailyRoutineCard.model_json_schema(), indent=2)

    prompt = f"""
    You are an expert basketball coach. Your task is to create a personalized
    training routine for a user based on their preferences and a list of
    retrieved drills.

    **User Preferences:**
    - Skill to Improve: {user_info.get("focus_area")}
    - Available Time: {user_info.get("available_time_min")} minutes
    - Available Equipment: {user_info.get("equipment")}

    **Retrieved Drills from Database:**
    {context_str}

    **Instructions:**
    1. Create a complete routine with 'warmup', 'main', and 'cooldown' phases.
    2. For the 'main' phase, select the most relevant drill(s) from the
       "Retrieved Drills". If none are relevant or available, create a
       fundamental drill appropriate for the user's "Skill to Improve".
    3. Allocate the user's "Available Time" intelligently across the drills.
       The sum of drill durations should be close to this time.
    4. For each drill, provide a specific, personalized "coaching_tip".
    5. Write an overall encouraging "coach_message".
    6. Your final output **must** be a JSON object that strictly follows this
       Pydantic schema:

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
