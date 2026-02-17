from fastapi import APIRouter

from src.api.v1.endpoints import gear, skill, whistle

api_router = APIRouter()

api_router.include_router(skill.router, prefix="/skill", tags=["Skill Lab"])
api_router.include_router(gear.router, prefix="/gear", tags=["Gear Advisor"])
api_router.include_router(whistle.router, prefix="/whistle", tags=["The Whistle"])
