import asyncio
import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from src.models.response_schema import SuccessResponse
from src.models.gear_schema import GearAdvisorRequest, GearAdvisorResponse
from src.services.agents.gear_agent import gear_agent_graph

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recommend", response_model=SuccessResponse[GearAdvisorResponse])
async def recommend_gear(
    request: GearAdvisorRequest,
) -> SuccessResponse[GearAdvisorResponse]:
    """
    Receives user's gear preferences and returns personalized shoe recommendations
    by invoking the GearAgent.

    Endpoint: POST /api/v1/gear/recommend
    """
    try:
        # The agent expects a list of messages, but our primary input is the
        # structured user_info. We can pass a synthetic message for context.
        initial_state = {
            "messages": [
                HumanMessage(
                    content=f"Recommend shoes for: {', '.join(request.sensory_preferences)}"
                )
            ],
            "user_info": request.model_dump(),
        }

        # Invoke the agent graph in a separate thread to avoid blocking the event loop
        final_state = await asyncio.to_thread(gear_agent_graph.invoke, initial_state)

        # The agent's final response is a JSON string, parse and validate it
        if final_response_str := final_state.get("final_response"):
            response_data = GearAdvisorResponse.model_validate_json(final_response_str)
            return SuccessResponse(data=response_data)
        else:
            raise HTTPException(
                status_code=500, detail="Agent failed to produce a final response."
            )

    except Exception as e:
        # If the exception is already an HTTPException, re-raise it to preserve
        # the specific status code and detail.
        if isinstance(e, HTTPException):
            raise
        # For any other unexpected errors from the agent workflow, log the full
        # exception details internally and return a generic error message.
        logger.exception("An unexpected error occurred during gear recommendation")
        raise HTTPException(
            status_code=500, detail="Internal server error"
        )
