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
