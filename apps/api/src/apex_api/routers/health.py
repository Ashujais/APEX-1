from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

from apex_api.redis_health import redis_ping

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "apex-api"}


@router.get("/ready")
def ready(request: Request) -> dict[str, object]:
    database_ok = False
    try:
        with request.app.state.database.session_factory() as session:
            session.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        database_ok = False
    settings = request.app.state.settings
    redis_configured = bool(settings.redis_url)
    redis_ok = (
        redis_ping(settings.redis_url, settings.redis_timeout_seconds)
        if settings.redis_url
        else False
    )
    redis_ready = redis_ok if settings.redis_required else True
    return {
        "status": "ready" if database_ok and redis_ready else "not_ready",
        "checks": {
            "database": database_ok,
            "model_router": True,
            "redis": {
                "configured": redis_configured,
                "required": settings.redis_required,
                "available": redis_ok,
                "status": (
                    "available"
                    if redis_ok
                    else "unavailable"
                    if redis_configured
                    else "not_configured"
                ),
            },
        },
    }


@router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return (
        "# HELP apex_api_up APEX API process availability\n"
        "# TYPE apex_api_up gauge\n"
        "apex_api_up 1\n"
    )
