from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from apex_api.config import Settings

password_hasher = PasswordHasher(time_cost=3, memory_cost=65_536, parallelism=2)


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def hash_opaque_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def create_signed_token(
    settings: Settings,
    subject: str,
    token_type: str,
    expires_delta: timedelta,
    **claims: Any,
) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "typ": token_type,
        "iss": settings.issuer,
        "iat": now,
        "nbf": now,
        "exp": now + expires_delta,
        "jti": secrets.token_hex(16),
        **claims,
    }
    secret = settings.secret_key
    assert secret is not None
    return jwt.encode(payload, secret.get_secret_value(), algorithm="HS256")


def decode_signed_token(settings: Settings, token: str, expected_type: str) -> dict[str, Any]:
    secret = settings.secret_key
    assert secret is not None
    payload = jwt.decode(
        token,
        secret.get_secret_value(),
        algorithms=["HS256"],
        issuer=settings.issuer,
        options={"require": ["exp", "iat", "nbf", "sub", "typ", "jti"]},
    )
    if payload.get("typ") != expected_type:
        raise jwt.InvalidTokenError("unexpected token type")
    return payload
