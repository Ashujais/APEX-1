from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from apex_api.config import Settings
from apex_api.dependencies import current_user, get_db, settings_from_request
from apex_api.models import AuthSession, User
from apex_api.schemas import (
    AuthTokens,
    LoginRequest,
    LogoutRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    PasswordResetResponse,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    UserView,
    VerifyEmailRequest,
)
from apex_api.security import (
    create_refresh_token,
    create_signed_token,
    decode_signed_token,
    hash_opaque_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/v1/auth", tags=["authentication"])


def issue_session_tokens(
    settings: Settings,
    db: Session,
    user: User,
    request: Request,
    existing_session: AuthSession | None = None,
    response: Response | None = None,
) -> AuthTokens:
    refresh_token = create_refresh_token()
    expires_at = datetime.now(UTC) + timedelta(days=settings.refresh_token_days)
    if existing_session is None:
        auth_session = AuthSession(
            user_id=user.id,
            refresh_token_hash=hash_opaque_token(refresh_token),
            user_agent=request.headers.get("user-agent", "")[:512] or None,
            ip_address=request.client.host if request.client else None,
            expires_at=expires_at,
        )
        db.add(auth_session)
        db.flush()
    else:
        auth_session = existing_session
        auth_session.refresh_token_hash = hash_opaque_token(refresh_token)
        auth_session.expires_at = expires_at
    access = create_signed_token(
        settings,
        user.id,
        "access",
        timedelta(minutes=settings.access_token_minutes),
        sid=auth_session.id,
        tenant=user.tenant_id,
    )
    db.commit()
    tokens = AuthTokens(
        access_token=access,
        refresh_token=refresh_token,
        expires_in=settings.access_token_minutes * 60,
    )
    if response is not None:
        response.set_cookie(
            key="apex_refresh",
            value=refresh_token,
            max_age=settings.refresh_token_days * 86_400,
            httponly=True,
            secure=settings.env.lower() == "production",
            samesite="strict",
            path="/v1/auth",
        )
    return tokens


def rotate_refresh_token(
    refresh_token: str,
    request: Request,
    response: Response,
    db: Session,
) -> AuthTokens:
    token_hash = hash_opaque_token(refresh_token)
    auth_session = db.scalar(
        select(AuthSession).where(AuthSession.refresh_token_hash == token_hash)
    )
    now = datetime.now(UTC)
    expires = auth_session.expires_at if auth_session is not None else None
    if expires is not None and expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    if (
        auth_session is None
        or auth_session.revoked_at is not None
        or expires is None
        or expires <= now
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    user = db.get(User, auth_session.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    return issue_session_tokens(
        settings_from_request(request),
        db,
        user,
        request,
        existing_session=auth_session,
        response=response,
    )


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> RegisterResponse:
    email = payload.email.lower()
    if db.scalar(select(User).where(User.email == email)) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Account already exists")
    user = User(
        email=email, name=payload.name.strip(), password_hash=hash_password(payload.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    settings = settings_from_request(request)
    verification_token = create_signed_token(
        settings, user.id, "email_verification", timedelta(hours=24), email=user.email
    )
    return RegisterResponse(
        user=UserView.model_validate(user),
        development_verification_token=(
            verification_token if settings.expose_development_tokens else None
        ),
    )


@router.post("/verify-email", response_model=UserView)
def verify_email(
    payload: VerifyEmailRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> UserView:
    try:
        claims = decode_signed_token(
            settings_from_request(request), payload.token, "email_verification"
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification token"
        ) from exc
    user = db.get(User, claims["sub"])
    if user is None or user.email != claims.get("email"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification token"
        )
    user.email_verified = True
    db.commit()
    return UserView.model_validate(user)


@router.post("/login", response_model=AuthTokens)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthTokens:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if user is None or not verify_password(user.password_hash, payload.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )
    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Email verification required"
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")
    return issue_session_tokens(
        settings_from_request(request), db, user, request, response=response
    )


@router.post("/refresh", response_model=AuthTokens)
def refresh(
    payload: RefreshRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthTokens:
    return rotate_refresh_token(payload.refresh_token, request, response, db)


@router.post("/browser/refresh", response_model=AuthTokens)
def browser_refresh(
    request: Request,
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias="apex_refresh"),
    db: Session = Depends(get_db),
) -> AuthTokens:
    if refresh_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Browser session unavailable"
        )
    return rotate_refresh_token(refresh_token, request, response, db)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(payload: LogoutRequest, db: Session = Depends(get_db)) -> None:
    token_hash = hash_opaque_token(payload.refresh_token)
    auth_session = db.scalar(
        select(AuthSession).where(AuthSession.refresh_token_hash == token_hash)
    )
    if auth_session is not None and auth_session.revoked_at is None:
        auth_session.revoked_at = datetime.now(UTC)
        db.commit()


@router.post("/browser/logout", status_code=status.HTTP_204_NO_CONTENT)
def browser_logout(
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias="apex_refresh"),
    db: Session = Depends(get_db),
) -> None:
    if refresh_token is not None:
        token_hash = hash_opaque_token(refresh_token)
        auth_session = db.scalar(
            select(AuthSession).where(AuthSession.refresh_token_hash == token_hash)
        )
        if auth_session is not None and auth_session.revoked_at is None:
            auth_session.revoked_at = datetime.now(UTC)
            db.commit()
    response.delete_cookie(key="apex_refresh", path="/v1/auth")


@router.post("/password-reset/request", response_model=PasswordResetResponse)
def request_password_reset(
    payload: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> PasswordResetResponse:
    settings = settings_from_request(request)
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    reset_token = None
    if user is not None:
        reset_token = create_signed_token(
            settings, user.id, "password_reset", timedelta(minutes=30), email=user.email
        )
    return PasswordResetResponse(
        message="If an account exists, password reset instructions will be sent.",
        development_reset_token=(
            reset_token if settings.expose_development_tokens and reset_token else None
        ),
    )


@router.post("/password-reset/confirm", status_code=status.HTTP_204_NO_CONTENT)
def confirm_password_reset(
    payload: PasswordResetConfirm,
    request: Request,
    db: Session = Depends(get_db),
) -> None:
    try:
        claims = decode_signed_token(
            settings_from_request(request), payload.token, "password_reset"
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid reset token"
        ) from exc
    user = db.get(User, claims["sub"])
    if user is None or user.email != claims.get("email"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid reset token")
    user.password_hash = hash_password(payload.new_password)
    now = datetime.now(UTC)
    for auth_session in user.sessions:
        if auth_session.revoked_at is None:
            auth_session.revoked_at = now
    db.commit()


@router.get("/me", response_model=UserView)
def me(user: User = Depends(current_user)) -> UserView:
    return UserView.model_validate(user)
