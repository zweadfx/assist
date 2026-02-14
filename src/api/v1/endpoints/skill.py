from fastapi import APIRouter

from src.models.response_schema import SuccessResponse
from src.models.skill_schema import Drill, SkillLabRequest, SkillLabResponse

router = APIRouter()


@router.post("/", response_model=SuccessResponse[SkillLabResponse])
async def create_skill_routine(
    request: SkillLabRequest,
) -> SuccessResponse[SkillLabLResponse]:
    """
    Receives user's skill profile and returns a personalized training routine.
    (This is a mock response for now)
    """
    # Mock response data based on the output interface specification
    mock_drills = [
        Drill(
            phase="warmup",
            drill_id="drill-001",
            name="Mikan Drill",
            duration_min=5,
            description="Classic drill for practicing layups and touch around the rim.",
            coaching_tip="Focus on using the backboard and keeping your eyes on the target.",
        ),
        Drill(
            phase="main",
            drill_id="drill-002",
            name="Stationary Dribbling Series",
            duration_min=15,
            description="A series of stationary dribbling exercises: crossovers, between the legs, behind the back.",
            coaching_tip="Keep your head up to see the court and use your fingertips to control the ball.",
        ),
        Drill(
            phase="cooldown",
            drill_id="drill-003",
            name="Static Stretching",
            duration_min=5,
            description="Hold various stretches for major muscle groups (quads, hamstrings, calves, shoulders).",
            coaching_tip="Hold each stretch for at least 30 seconds without bouncing.",
        ),
    ]

    mock_response_data = SkillLabResponse(
        routine_title=f"Personalized {request.focus_area.capitalize()} Workout",
        total_duration_min=sum(drill.duration_min for drill in mock_drills),
        coach_message=f"Here is a routine to improve your {request.focus_area}. Let's get to work!",
        drills=mock_drills,
    )

    return SuccessResponse(data=mock_response_data)
