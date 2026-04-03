import asyncio
import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

from src.models.refine_schema import SkillRefineRequest, WeeklyRefineRequest
from src.models.response_schema import SuccessResponse
from src.models.skill_schema import SkillLabRequest, SkillLabResponse
from src.models.weekly_schema import WeeklyRoutineRequest, WeeklyRoutineResponse
from src.services.agents.coach_agent import coach_agent_graph
from src.services.agents.coach_refine_agent import coach_refine_graph
from src.services.agents.weekly_coach_agent import weekly_coach_agent_graph
from src.services.agents.weekly_coach_refine_agent import weekly_coach_refine_graph

router = APIRouter()


@router.post("/", response_model=SuccessResponse[SkillLabResponse])
async def create_skill_routine(
    request: SkillLabRequest,
) -> SuccessResponse[SkillLabResponse]:
    """
    Receives user's skill profile and returns a micro-step progressive
    breakdown of a single basketball skill by invoking the CoachAgent.
    """
    try:
        skill_desc = request.specific_skill or request.category
        initial_state = {
            "messages": [
                HumanMessage(
                    content=(f"Generate a micro-step skill breakdown for {skill_desc}")
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
        if isinstance(e, HTTPException):
            raise
        logger.exception("Unexpected error in create_skill_routine")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        ) from e


@router.post("/weekly", response_model=SuccessResponse[WeeklyRoutineResponse])
async def generate_weekly_routine(
    request: WeeklyRoutineRequest,
) -> SuccessResponse[WeeklyRoutineResponse]:
    """
    Receives user's training preferences and returns a personalized weekly
    training routine by invoking the WeeklyCoachAgent.
    """
    try:
        initial_state = {
            "messages": [
                HumanMessage(
                    content=(
                        f"Generate a {request.training_days}-day weekly training "
                        f"routine focusing on {', '.join(request.focus_areas)}"
                    )
                )
            ],
            "user_info": request.model_dump(),
        }

        final_state = await asyncio.wait_for(
            asyncio.to_thread(weekly_coach_agent_graph.invoke, initial_state),
            timeout=120,
        )

        if final_response_str := final_state.get("final_response"):
            response_data = WeeklyRoutineResponse.model_validate_json(
                final_response_str
            )
            return SuccessResponse(data=response_data)
        else:
            raise HTTPException(
                status_code=500,
                detail="Agent failed to produce a weekly routine response.",
            )

    except asyncio.TimeoutError:
        logger.error("Weekly routine generation timed out")
        raise HTTPException(
            status_code=504,
            detail="Weekly routine generation timed out. Please try again.",
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.exception("Unexpected error in generate_weekly_routine")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        ) from e


@router.post("/refine", response_model=SuccessResponse[SkillLabResponse])
async def refine_skill_routine(
    request: SkillRefineRequest,
) -> SuccessResponse[SkillLabResponse]:
    """
    Refines a previously generated skill breakdown based on user feedback.
    Classifies feedback to determine if RAG re-retrieval is needed,
    then generates a revised response.
    """
    try:
        initial_state = {
            "messages": [
                HumanMessage(
                    content=f"Refine skill breakdown based on feedback: {request.feedback}"
                )
            ],
            "user_info": request.original_request.model_dump(),
            "context": [],
            "previous_response": request.previous_response.model_dump_json(),
            "feedback": request.feedback,
            "feedback_type": "",
            "final_response": "",
        }

        final_state = await asyncio.to_thread(
            coach_refine_graph.invoke, initial_state
        )

        if final_response_str := final_state.get("final_response"):
            response_data = SkillLabResponse.model_validate_json(final_response_str)
            return SuccessResponse(data=response_data)
        else:
            raise HTTPException(
                status_code=500,
                detail="Agent failed to produce a refined response.",
            )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.exception("Unexpected error in refine_skill_routine")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        ) from e


@router.post("/weekly/refine", response_model=SuccessResponse[WeeklyRoutineResponse])
async def refine_weekly_routine(
    request: WeeklyRefineRequest,
) -> SuccessResponse[WeeklyRoutineResponse]:
    """
    Refines a previously generated weekly routine based on user feedback.
    Classifies feedback to determine if RAG re-retrieval is needed,
    then generates a revised response.
    """
    try:
        # Extract week_plan from the previous response's day structure
        week_plan = {}
        for day in request.previous_response.days:
            week_plan[str(day.day_number)] = day.focus_areas

        initial_state = {
            "messages": [
                HumanMessage(
                    content=f"Refine weekly routine based on feedback: {request.feedback}"
                )
            ],
            "user_info": request.original_request.model_dump(),
            "week_plan": week_plan,
            "context": {},
            "previous_response": request.previous_response.model_dump_json(),
            "feedback": request.feedback,
            "feedback_type": "",
            "final_response": "",
        }

        final_state = await asyncio.wait_for(
            asyncio.to_thread(weekly_coach_refine_graph.invoke, initial_state),
            timeout=120,
        )

        if final_response_str := final_state.get("final_response"):
            response_data = WeeklyRoutineResponse.model_validate_json(
                final_response_str
            )
            return SuccessResponse(data=response_data)
        else:
            raise HTTPException(
                status_code=500,
                detail="Agent failed to produce a refined weekly response.",
            )

    except asyncio.TimeoutError:
        logger.error("Weekly routine refinement timed out")
        raise HTTPException(
            status_code=504,
            detail="Weekly routine refinement timed out. Please try again.",
        ) from None
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.exception("Unexpected error in refine_weekly_routine")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        ) from e
