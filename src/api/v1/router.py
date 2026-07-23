from fastapi import APIRouter, Depends

from src.api.v1.endpoints import auth, gear, plans, skill, whistle
from src.core.rate_limit import enforce_rate_limit

api_router = APIRouter()

# 레이트리밋은 LLM 호출 라우터(skill·gear·whistle)에만 적용 — auth·plans 제외
api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(
    skill.router,
    prefix="/skill",
    tags=["Skill Lab"],
    dependencies=[Depends(enforce_rate_limit)],
)
api_router.include_router(
    gear.router,
    prefix="/gear",
    tags=["Gear Advisor"],
    dependencies=[Depends(enforce_rate_limit)],
)
api_router.include_router(
    whistle.router,
    prefix="/whistle",
    tags=["The Whistle"],
    dependencies=[Depends(enforce_rate_limit)],
)
api_router.include_router(plans.router, prefix="/plans", tags=["Plans"])
