from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
import json
from pydantic import BaseModel, Field
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
    Takes the user's natural language request and extracts structured information
    about their training preferences using an LLM. This forms the first node
    in our graph.
    """
    print("---NODE: Diagnosing User State---")
    # Get the latest user message
    if not state["messages"]:
        raise ValueError("No messages found in state.")
    user_message = state["messages"][-1].content

    # Create the prompt for the LLM
    prompt = f"""
    You are an expert basketball coach's assistant. A user has sent the
    following request for a training plan. Your task is to extract the key
    details needed to create the plan. Please extract the information and format
    it as a JSON object that strictly follows this Pydantic schema:

    ```json
    {UserDrillPreferences.model_json_schema()}
    ```

    User Request: "{user_message}"

    JSON Output:
    """

    # Use the OpenAI client to get structured output
    response = openai_client.chat.completions.create(
        # Using a model that is good at following instructions and JSON formatting
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    extracted_data = json.loads(response.choices[0].message.content)

    # Validate the data with the Pydantic model
    validated_info = UserDrillPreferences.model_validate(extracted_data)
    print(f"---Extracted User Info: {validated_info.model_dump_json()}---")

    return {"user_info": validated_info.model_dump()}


def retrieve_drills(state: CoachAgentState) -> dict:
    """
    (Placeholder) Retrieves relevant drills from the vector store based on user_info.
    This node will perform a RAG search.
    """
    print("---NODE: Retrieving Drills---")
    user_info = state["user_info"]
    print(f"---Querying for drills related to: {user_info['focus_area']}---")

    # TODO: Implement actual RAG search using ChromaDBManager
    # For now, return a dummy document
    dummy_doc = Document(
        page_content="This is a placeholder for a retrieved basketball drill.",
        metadata={"name": "Dummy Drill", "category": user_info["focus_area"]},
    )

    return {"context": [dummy_doc]}


def generate_routine(state: CoachAgentState) -> dict:
    """
    (Placeholder) Generates the final "Daily Routine Card" based on the
    retrieved drills and user preferences.
    """
    print("---NODE: Generating Routine---")
    context = state["context"]
    user_info = state["user_info"]

    # TODO: Implement LLM call to generate a structured routine
    # For now, return a dummy JSON string
    dummy_routine = {
        "routine_title": f"Personalized {user_info['focus_area'].title()} Routine",
        "total_duration_min": user_info["available_time_min"],
        "coach_message": (
            "Here is a personalized routine to help you improve. "
            "Let's get to work!"
        ),
        "drills": [
            {
                "phase": "warmup",
                "drill_id": "dummy-01",
                "name": "Stretching",
                "duration_min": 5,
                "description": "Dynamic stretches to prepare your body.",
                "coaching_tip": "Focus on fluid movements.",
            },
            {
                "phase": "main",
                "drill_id": context[0].metadata["name"],
                "name": context[0].metadata["name"],
                "duration_min": 15,
                "description": context[0].page_content,
                "coaching_tip": "Maintain good form.",
            },
            {
                "phase": "cooldown",
                "drill_id": "dummy-02",
                "name": "Cool-down Jog",
                "duration_min": 5,
                "description": "Light jogging and static stretches.",
                "coaching_tip": "Help your body recover.",
            },
        ],
    }

    final_response_str = json.dumps(dummy_routine, indent=2)
    print(f"---Generated Response: {final_response_str}---")

    return {"final_response": final_response_str}
