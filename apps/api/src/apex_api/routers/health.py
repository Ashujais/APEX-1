from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

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
    return {
        "status": "ready" if database_ok else "not_ready",
        "checks": {"database": database_ok, "model_router": True},
    }


@router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return (
        "# HELP apex_api_up APEX API process availability\n"
        "# TYPE apex_api_up gauge\n"
        "apex_api_up 1\n"
    )
