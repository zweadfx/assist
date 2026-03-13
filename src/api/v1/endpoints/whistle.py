import asyncio
import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from src.models.response_schema import SuccessResponse
from src.models.rule_schema import WhistleRequest, WhistleResponse
from src.services.agents.judge_agent import judge_agent_graph

logger = logging.getLogger(__name__)
router = APIRouter()

JUDGMENT_TIMEOUT_SECONDS = 60


@router.post("/judge", response_model=SuccessResponse[WhistleResponse])
async def judge_situation(
    request: WhistleRequest,
) -> SuccessResponse[WhistleResponse]:
    """
    Receives a basketball situation description and returns an AI-generated
    judgment with rule references by invoking the JudgeAgent.

    Endpoint: POST /api/v1/whistle/judge
    """
    initial_state = {
        "messages": [HumanMessage(content=request.situation_description)],
        "user_info": request.model_dump(),
    }

    try:
        final_state = await asyncio.wait_for(
            asyncio.to_thread(judge_agent_graph.invoke, initial_state),
            timeout=JUDGMENT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("Judgment timed out after %ds", JUDGMENT_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="Judgment timed out. Please try again.")
    except ValueError as e:
        msg = str(e)
        if "retrieve" in msg.lower() or "database" in msg.lower():
            logger.exception("RAG retrieval failed")
            raise HTTPException(status_code=503, detail="Rule database unavailable. Please try again.")
        logger.exception("Agent pipeline error")
        raise HTTPException(status_code=500, detail="Failed to generate judgment.")
    except Exception:
        logger.exception("Unexpected error during judgment")
        raise HTTPException(status_code=500, detail="Internal server error")

    if not (final_response_str := final_state.get("final_response")):
        raise HTTPException(status_code=500, detail="Agent failed to produce a final response.")

    response_data = WhistleResponse.model_validate_json(final_response_str)
    return SuccessResponse(data=response_data)
