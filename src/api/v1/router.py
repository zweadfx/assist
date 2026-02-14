from fastapi import APIRouter

from src.api.v1.endpoints import skill

api_router = APIRouter()

api_router.include_router(skill.router, prefix="/skill", tags=["Skill Lab"])
