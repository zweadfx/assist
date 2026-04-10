"""SQLAlchemy ORM models."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    nickname: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class SavedPlan(Base):
    __tablename__ = "saved_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    plan_type: Mapped[str] = mapped_column(String(10), nullable=False)  # "weekly" | "skill"
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    data: Mapped[Any] = mapped_column(JSON, nullable=False)
    training_dates: Mapped[Any] = mapped_column(JSON, nullable=False)  # ["2026-04-10", "2026-04-12", ...]
    total_days: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_days: Mapped[Any] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
