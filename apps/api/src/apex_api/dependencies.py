from __future__ import annotations

from collections.abc import Generator

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from apex_api.config import Settings
from apex_api.models import AuthSession, User
from apex_api.security import decode_signed_token

bearer = HTTPBearer(auto_error=False)


def get_db(request: Request) -> Generator[Session, None, None]:
    yield from request.app.state.database.session()


def settings_from_request(request: Request) -> Settings:
    return request.app.state.settings


def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    settings = settings_from_request(request)
    try:
        payload = decode_signed_token(settings, credentials.credentials, "access")
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token"
        ) from exc

    user = db.get(User, payload["sub"])
    session_id = payload.get("sid")
    auth_session = db.get(AuthSession, session_id) if session_id else None
    if (
        user is None
        or not user.is_active
        or auth_session is None
        or auth_session.revoked_at is not None
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is no longer active"
        )
    return user
