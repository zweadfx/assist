"""Database engine and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=(
        {"check_same_thread": False}
        if settings.DATABASE_URL.startswith("sqlite")
        else {}
    ),
)

SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
