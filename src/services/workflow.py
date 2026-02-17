"""
Unified workflow integrating all agent functionalities.
Routes user requests to appropriate agents (Skill Lab, Gear Advisor, Rule Expert).
"""

import json
import logging
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from src.services.agents.coach_agent import coach_agent_graph
from src.services.agents.gear_agent import gear_agent_graph
from src.services.rag.embedding import client as openai_client

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Unified state shared across all nodes in the workflow graph.
    """

    # Conversation history between user and agent
    messages: List[BaseMessage]

    # User's question intent (skill_lab, shoe_recommendation, rule_query)
    intent: str

    # Retrieved documents from vector DB (RAG data)
    context: List[Document]

    # User profile information (height, weight, position, skill level, etc.)
    user_info: dict

    # Router's routing decision
    routing_decision: str

    # Final generated response
    final_response: str


def router_node(state: AgentState) -> dict:
    """
    Router Node: Analyzes user's question and routes to appropriate agent.

    Uses LLM to classify user intent into one of:
    - 'skill_lab': Training routine generation
    - 'shoe_recommendation': Basketball shoe recommendations
    - 'rule_query': Basketball rules inquiry

    Args:
        state: Current agent state with messages and user_info

    Returns:
        Updated state with routing_decision and intent
    """
    print("---NODE: Router (Intent Detection)---")

    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages found in state for routing.")

    # Get the latest user message
    latest_message = messages[-1].content

    # Prepare routing prompt
    routing_prompt = f"""
You are a routing assistant for a basketball training app. Analyze the user's question and classify it into ONE of these categories:

Categories:
1. "skill_lab" - User wants training drills, workout routines, or skill improvement plans
2. "shoe_recommendation" - User wants basketball shoe recommendations based on preferences or playing style
3. "rule_query" - User has questions about basketball rules, regulations, or game situations

User's Question: "{latest_message}"

Respond with ONLY the category name (skill_lab, shoe_recommendation, or rule_query).
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for faster routing
            messages=[{"role": "user", "content": routing_prompt}],
            temperature=0.0,  # Deterministic routing
        )

        # Safely extract content with null-safety check
        msg = response.choices[0].message.content
        intent = msg.strip().lower() if msg else ""

        # Validate intent
        valid_intents = ["skill_lab", "shoe_recommendation", "rule_query"]
        if intent not in valid_intents:
            logger.warning(f"Invalid intent '{intent}', defaulting to 'skill_lab'")
            intent = "skill_lab"

        print(f"---Detected Intent: {intent}---")

        return {"routing_decision": intent, "intent": intent}

    except Exception as e:
        logger.error(f"Error in router_node: {e}")
        # Default to skill_lab on error
        return {"routing_decision": "skill_lab", "intent": "skill_lab"}


def skill_lab_node(state: AgentState) -> dict:
    """
    Skill Lab Node: Generates personalized training routines.
    Invokes the CoachAgent graph.
    """
    print("---NODE: Skill Lab (Training Routine Generation)---")

    try:
        # Prepare initial state for coach agent
        coach_state = {
            "messages": state.get("messages", []),
            "user_info": state.get("user_info", {}),
        }

        # Invoke coach agent
        final_state = coach_agent_graph.invoke(coach_state)

        return {"final_response": final_state.get("final_response", "")}

    except Exception:
        # Log full exception details server-side for debugging
        logger.exception("Error in skill_lab_node")
        # Return generic error message without exposing internal details
        return {
            "final_response": json.dumps(
                {
                    "error": "Failed to generate training routine",
                    "message": "An internal error occurred while processing your request. Please try again later.",
                }
            )
        }


def shoe_recommendation_node(state: AgentState) -> dict:
    """
    Shoe Recommendation Node: Generates personalized shoe recommendations.
    Invokes the GearAgent graph.
    """
    print("---NODE: Shoe Recommendation (Gear Advisor)---")

    try:
        # Prepare initial state for gear agent
        gear_state = {
            "messages": state.get("messages", []),
            "user_info": state.get("user_info", {}),
        }

        # Invoke gear agent
        final_state = gear_agent_graph.invoke(gear_state)

        return {"final_response": final_state.get("final_response", "")}

    except Exception:
        # Log full exception details server-side for debugging
        logger.exception("Error in shoe_recommendation_node")
        # Return generic error message without exposing internal details
        return {
            "final_response": json.dumps(
                {
                    "error": "Failed to generate shoe recommendations",
                    "message": "An internal error occurred while processing your request. Please try again later.",
                }
            )
        }


def rule_query_node(state: AgentState) -> dict:
    """
    Rule Query Node: Answers basketball rules questions.
    (Placeholder for future implementation)
    """
    print("---NODE: Rule Query (Not Implemented)---")

    return {
        "final_response": json.dumps(
            {
                "message": "Rule query feature is not yet implemented.",
                "intent": "rule_query",
            }
        )
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional routing function.
    Determines which agent node to execute based on routing_decision.

    Args:
        state: Current agent state with routing_decision

    Returns:
        Name of the next node to execute
    """
    routing_decision = state.get("routing_decision", "skill_lab")

    if routing_decision == "shoe_recommendation":
        return "shoe_recommendation"
    elif routing_decision == "skill_lab":
        return "skill_lab"
    elif routing_decision == "rule_query":
        return "rule_query"
    else:
        # Default to skill_lab
        return "skill_lab"


# Build the workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("skill_lab", skill_lab_node)
workflow.add_node("shoe_recommendation", shoe_recommendation_node)
workflow.add_node("rule_query", rule_query_node)

# Define edges
workflow.set_entry_point("router")

# Conditional routing from router to specific agents
workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "skill_lab": "skill_lab",
        "shoe_recommendation": "shoe_recommendation",
        "rule_query": "rule_query",
    },
)

# All agent nodes lead to END
workflow.add_edge("skill_lab", END)
workflow.add_edge("shoe_recommendation", END)
workflow.add_edge("rule_query", END)

# Compile the workflow
unified_workflow = workflow.compile()
