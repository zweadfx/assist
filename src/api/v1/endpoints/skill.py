import asyncio
import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

from src.core.constants import SKILL_TIMEOUT_SECONDS, WEEKLY_TIMEOUT_SECONDS
from src.models.response_schema import SuccessResponse
from src.models.skill_schema import SkillLabRequest, SkillLabResponse
from src.models.weekly_schema import WeeklyRoutineRequest, WeeklyRoutineResponse
from src.services.agents.coach_agent import coach_agent_graph
from src.services.agents.weekly_coach_agent import weekly_coach_agent_graph

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

        final_state = await asyncio.wait_for(
            coach_agent_graph.ainvoke(initial_state),
            timeout=SKILL_TIMEOUT_SECONDS,
        )

        # The agent's final response is a JSON string, parse and validate it
        if final_response_str := final_state.get("final_response"):
            response_data = SkillLabResponse.model_validate_json(final_response_str)
            return SuccessResponse(data=response_data)
        else:
            raise HTTPException(
                status_code=500, detail="Agent failed to produce a final response."
            )

    except asyncio.TimeoutError as e:
        logger.error("Skill routine generation timed out after %ds", SKILL_TIMEOUT_SECONDS)
        raise HTTPException(
            status_code=504, detail="Skill routine generation timed out. Please try again."
        ) from e
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
            weekly_coach_agent_graph.ainvoke(initial_state),
            timeout=WEEKLY_TIMEOUT_SECONDS,
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

    except asyncio.TimeoutError as e:
        logger.error("Weekly routine generation timed out after %ds", WEEKLY_TIMEOUT_SECONDS)
        raise HTTPException(
            status_code=504,
            detail="Weekly routine generation timed out. Please try again.",
        ) from e
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.exception("Unexpected error in generate_weekly_routine")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        ) from e


