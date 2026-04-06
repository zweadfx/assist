"""Authentication endpoints: signup and login."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.db.database import get_db
from src.db.models import User
from src.models.auth_schema import (
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency that extracts and validates the current user."""
    email = decode_access_token(token)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.post(
    "/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """Register a new user."""
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered.",
        )

    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        nickname=request.nickname,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info("New user registered: %s", user.email)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and return a JWT access token."""
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token = create_access_token(subject=user.email)
    logger.info("User logged in: %s", user.email)
    return TokenResponse(access_token=token)
