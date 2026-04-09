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
        start_date=req.start_date,
        total_days=req.total_days,
        completed_days=[],
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return SuccessResponse(data=SavedPlanResponse.model_validate(plan))


@router.get("/", response_model=SuccessResponse[list[SavedPlanResponse]])
def get_plans(
    year: int = Query(...),
    month: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SuccessResponse[list[SavedPlanResponse]]:
    from calendar import monthrange
    from datetime import date

    first_day = date(year, month, 1)
    last_day = date(year, month, monthrange(year, month)[1])

    plans = (
        db.query(SavedPlan)
        .filter(
            SavedPlan.user_id == current_user.id,
            SavedPlan.start_date <= last_day,
        )
        .order_by(SavedPlan.start_date)
        .all()
    )

    # Filter: plans whose date range overlaps with the requested month
    result = []
    for plan in plans:
        from datetime import timedelta

        plan_end = plan.start_date + timedelta(days=plan.total_days - 1)
        if plan_end >= first_day:
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
