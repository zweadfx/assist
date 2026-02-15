from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


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
