import asyncio

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from src.models.response_schema import SuccessResponse
from src.models.skill_schema import SkillLabRequest, SkillLabResponse
from src.services.agents.coach_agent import coach_agent_graph

router = APIRouter()


@router.post("/", response_model=SuccessResponse[SkillLabResponse])
async def create_skill_routine(
    request: SkillLabRequest,
) -> SuccessResponse[SkillLabResponse]:
    """
    Receives user's skill profile and returns a personalized training routine
    by invoking the CoachAgent.
    """
    try:
        # The agent expects a list of messages, but our primary input is the
        # structured user_info. We can pass a synthetic message for context.
        initial_state = {
            "messages": [
                HumanMessage(
                    content=f"Generate a training routine for {request.focus_area}"
                )
            ],
            "user_info": request.model_dump(),
        }

        # Invoke the agent graph in a separate thread to avoid blocking the event loop
        final_state = await asyncio.to_thread(coach_agent_graph.invoke, initial_state)

        # The agent's final response is a JSON string, parse and validate it
        if final_response_str := final_state.get("final_response"):
            response_data = SkillLabResponse.model_validate_json(final_response_str)
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
        # For any other unexpected errors from the agent workflow, wrap them
        # in a generic 500 error.
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {e}"
        ) from e
