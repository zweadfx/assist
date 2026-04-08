"""Database engine and session management."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.core.config import settings

database_url = settings.DATABASE_URL
# Render provides "postgres://" but SQLAlchemy requires "postgresql://"
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

if database_url.startswith("sqlite"):
    db_path = database_url.replace("sqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    database_url,
    connect_args=(
        {"check_same_thread": False}
        if database_url.startswith("sqlite")
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
