from __future__ import annotations

from fastapi import APIRouter, Request

from apex_api.schemas import ModelView

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("", response_model=list[ModelView])
def list_models(request: Request) -> list[ModelView]:
    return [
        ModelView(
            id=item.id,
            name=item.name,
            status=item.status,
            description=item.description,
            modalities=list(item.modalities),
            capabilities=list(item.capabilities),
        )
        for item in request.app.state.model_router.list()
    ]
