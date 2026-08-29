from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from apex_api.config import Settings, get_settings
from apex_api.database import Database
from apex_api.providers import ModelRouter
from apex_api.routers import auth, chat, health, models


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    database = Database(runtime_settings.database_url)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        database.create_schema()
        yield
        database.dispose()

    application = FastAPI(
        title="APEX-1 API",
        version="0.1.0",
        description=(
            "Verified platform foundation; see docs/CURRENT_STATUS.md for capability status."
        ),
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.database = database
    application.state.model_router = ModelRouter()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))[:128]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    application.include_router(health.router)
    application.include_router(models.router)
    application.include_router(auth.router)
    application.include_router(chat.router)
    return application


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
