import json
import logging
from typing import List, TypedDict

import openai
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from src.services.rag.chroma_db import chroma_manager
from src.services.rag.embedding import client as openai_client


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
    print("---NODE: Diagnosing User State---")
    if not state.get("user_info"):
        raise ValueError("User info is missing from the state.")

    print(f"---User Info: {state['user_info']}---")
    # The user_info is already structured and passed in the initial call
    # so we just pass it along to the next node.
    return {"user_info": state["user_info"]}


def retrieve_drills(state: CoachAgentState) -> dict:
    """
    Retrieves relevant drills from the vector store based on user_info.
    This node performs a RAG search.
    """
    print("---NODE: Retrieving Drills---")
    user_info = state["user_info"]
    focus_area = user_info.get("focus_area", "")

    # 1. Transform user input into a query
    query_text = f"A basketball drill focusing on improving {focus_area} skills."
    print(f"---Querying for drills related to: {focus_area}---")

    # 2. Call the search function
    # TODO: Implement metadata filtering for equipment
    results = chroma_manager.query_drills(query_texts=[query_text], n_results=3)

    # 3. Process the results into Document objects
    retrieved_docs = []
    if results and results.get("documents"):
        # The result is a list containing one list of documents for our single query
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for i, doc_content in enumerate(documents):
            doc = Document(page_content=doc_content, metadata=metadatas[i])
            retrieved_docs.append(doc)
        print(f"---Retrieved {len(retrieved_docs)} drills.---")
    else:
        print("---No drills retrieved.---")

    return {"context": retrieved_docs}


class Drill(BaseModel):
    """Data model for a single drill within a routine."""

    phase: str = Field(
        description="The phase of the workout, e.g., 'warmup', 'main', 'cooldown'."
    )
    drill_id: str = Field(description="A unique identifier for the drill.")
    name: str = Field(description="The name of the drill.")
    duration_min: int = Field(description="The duration of the drill in minutes.")
    description: str = Field(
        description="A brief description of how to perform the drill."
    )
    coaching_tip: str = Field(description="A personalized coaching tip for the user.")


class DailyRoutineCard(BaseModel):
    """Data model for the final daily routine card output."""

    routine_title: str = Field(
        description="A catchy and relevant title for the routine."
    )
    total_duration_min: int = Field(
        description="The total calculated duration for the entire routine in minutes."
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
    print("---NODE: Generating Routine---")
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
    {DailyRoutineCard.model_json_schema()}
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
            print(f"---Generated Response: {final_response_str}---")
            return {"final_response": final_response_str}
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(f"Failed to parse or validate LLM response for routine: {e}")
            raise ValueError(
                f"LLM returned an invalid routine object: {content}"
            ) from e

    except openai.APIError as e:
        logging.error(f"OpenAI API error during routine generation: {e}")
        raise ValueError("Failed to generate routine due to an API error.") from e
    except Exception as e:
        logging.error(f"An unexpected error occurred during routine generation: {e}")
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
