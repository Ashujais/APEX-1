from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from apex_api.dependencies import current_user, get_db
from apex_api.models import Conversation, Message, User, utc_now
from apex_api.schemas import ChatRequest, ConversationCreate, ConversationView

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])


def scoped_conversation(db: Session, conversation_id: str, user: User) -> Conversation:
    conversation = db.scalar(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == user.tenant_id,
            Conversation.user_id == user.id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


@router.get("", response_model=list[ConversationView])
def list_conversations(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[ConversationView]:
    items = db.scalars(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.tenant_id == user.tenant_id, Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    ).all()
    return [ConversationView.model_validate(item) for item in items]


@router.post("", response_model=ConversationView, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> ConversationView:
    try:
        request.app.state.model_router.get(payload.model_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    item = Conversation(
        tenant_id=user.tenant_id,
        user_id=user.id,
        title=payload.title,
        model_id=payload.model_id,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return ConversationView.model_validate(item)


@router.get("/{conversation_id}", response_model=ConversationView)
def get_conversation(
    conversation_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> ConversationView:
    return ConversationView.model_validate(scoped_conversation(db, conversation_id, user))


@router.post("/{conversation_id}/stream")
def stream_message(
    conversation_id: str,
    payload: ChatRequest,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    conversation = scoped_conversation(db, conversation_id, user)
    try:
        provider = request.app.state.model_router.get(conversation.model_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    user_message = Message(
        conversation_id=conversation.id,
        tenant_id=user.tenant_id,
        role="user",
        content=payload.prompt,
    )
    conversation.updated_at = utc_now()
    db.add(user_message)
    db.commit()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    database = request.app.state.database

    def events() -> Iterator[str]:
        yield _event("metadata", {"request_id": request_id, "model": provider.descriptor.id})
        chunks: list[str] = []
        for chunk in provider.stream(payload.prompt):
            chunks.append(chunk)
            yield _event("delta", {"text": chunk})
        content = "".join(chunks)
        with database.session_factory() as stream_db:
            stream_db.add(
                Message(
                    conversation_id=conversation.id,
                    tenant_id=user.tenant_id,
                    role="assistant",
                    content=content,
                    model_id=provider.descriptor.id,
                )
            )
            persisted_conversation = stream_db.get(Conversation, conversation.id)
            if persisted_conversation is not None:
                persisted_conversation.updated_at = utc_now()
            stream_db.commit()
        yield _event("done", {"verified": False, "status": provider.descriptor.status})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _event(name: str, data: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
