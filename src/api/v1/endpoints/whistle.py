import asyncio
import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from src.models.response_schema import SuccessResponse
from src.models.rule_schema import WhistleRequest, WhistleResponse
from src.services.agents.judge_agent import judge_agent_graph

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/judge", response_model=SuccessResponse[WhistleResponse])
async def judge_situation(
    request: WhistleRequest,
) -> SuccessResponse[WhistleResponse]:
    """
    Receives a basketball situation description and returns an AI-generated
    judgment with rule references by invoking the JudgeAgent.

    Endpoint: POST /api/v1/whistle/judge
    """
    try:
        initial_state = {
            "messages": [
                HumanMessage(content=request.situation_description)
            ],
            "user_info": request.model_dump(),
        }

        final_state = await asyncio.to_thread(
            judge_agent_graph.invoke, initial_state
        )

        if final_response_str := final_state.get("final_response"):
            try:
                response_data = WhistleResponse.model_validate_json(
                    final_response_str
                )
            except ValidationError as e:
                logger.error(
                    "LLM returned invalid JSON for WhistleResponse: %s\n"
                    "Raw response: %s",
                    e,
                    final_response_str,
                )
                raise HTTPException(
                    status_code=422,
                    detail="LLM returned invalid judgment response",
                ) from e
            return SuccessResponse(data=response_data)
        else:
            raise HTTPException(
                status_code=500,
                detail="Agent failed to produce a final response.",
            )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.exception(
            "An unexpected error occurred during rule judgment"
        )
        raise HTTPException(
            status_code=500, detail="Internal server error"
        ) from e
