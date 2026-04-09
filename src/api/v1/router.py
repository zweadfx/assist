from fastapi import APIRouter

from src.api.v1.endpoints import auth, gear, plans, skill, whistle

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(skill.router, prefix="/skill", tags=["Skill Lab"])
api_router.include_router(gear.router, prefix="/gear", tags=["Gear Advisor"])
api_router.include_router(whistle.router, prefix="/whistle", tags=["The Whistle"])
api_router.include_router(plans.router, prefix="/plans", tags=["Plans"])
