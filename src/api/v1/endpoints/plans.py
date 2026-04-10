"""Training plan (calendar) endpoints — member only."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.v1.endpoints.auth import get_current_user
from src.db.database import get_db
from src.db.models import SavedPlan, User
from src.models.plan_schema import CompleteDayRequest, SavedPlanResponse, SavePlanRequest
from src.models.response_schema import SuccessResponse

router = APIRouter()


@router.post("/", response_model=SuccessResponse[SavedPlanResponse])
def save_plan(
    req: SavePlanRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SuccessResponse[SavedPlanResponse]:
    plan = SavedPlan(
        user_id=current_user.id,
        plan_type=req.plan_type,
        title=req.title,
        data=req.data,
        training_dates=[d.isoformat() for d in req.training_dates],
        total_days=len(req.training_dates),
        completed_days=[],
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return SuccessResponse(data=SavedPlanResponse.model_validate(plan))


@router.get("/", response_model=SuccessResponse[list[SavedPlanResponse]])
def get_plans(
    year: int = Query(..., ge=1, le=9999),
    month: int = Query(..., ge=1, le=12),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SuccessResponse[list[SavedPlanResponse]]:
    plans = (
        db.query(SavedPlan)
        .filter(SavedPlan.user_id == current_user.id)
        .all()
    )

    # Filter: plans with any training_date in the requested month
    month_prefix = f"{year:04d}-{month:02d}-"
    result = []
    for plan in plans:
        if any(d.startswith(month_prefix) for d in (plan.training_dates or [])):
            result.append(SavedPlanResponse.model_validate(plan))

    return SuccessResponse(data=result)


@router.patch("/{plan_id}/complete", response_model=SuccessResponse[SavedPlanResponse])
def complete_plan_day(
    plan_id: int,
    req: CompleteDayRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SuccessResponse[SavedPlanResponse]:
    plan = db.query(SavedPlan).filter(
        SavedPlan.id == plan_id,
        SavedPlan.user_id == current_user.id,
    ).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    if req.day_number < 1 or req.day_number > len(plan.training_dates or []):
        raise HTTPException(status_code=422, detail="day_number out of range")

    completed: list[int] = list(plan.completed_days or [])
    if req.completed:
        if req.day_number not in completed:
            completed.append(req.day_number)
    else:
        completed = [d for d in completed if d != req.day_number]

    plan.completed_days = completed
    db.commit()
    db.refresh(plan)
    return SuccessResponse(data=SavedPlanResponse.model_validate(plan))


@router.delete("/{plan_id}", response_model=SuccessResponse[None])
def delete_plan(
    plan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SuccessResponse[None]:
    plan = db.query(SavedPlan).filter(
        SavedPlan.id == plan_id,
        SavedPlan.user_id == current_user.id,
    ).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    db.delete(plan)
    db.commit()
    return SuccessResponse(message="Plan deleted successfully.")
