from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from apex_api.dependencies import current_user, get_db
from apex_api.models import Conversation, FileAsset, User, new_id
from apex_api.schemas import FileView
from apex_api.storage import InvalidUpload, UploadTooLarge

router = APIRouter(prefix="/v1/files", tags=["files"])


def scoped_file(db: Session, file_id: str, user: User) -> FileAsset:
    asset = db.scalar(
        select(FileAsset).where(
            FileAsset.id == file_id,
            FileAsset.tenant_id == user.tenant_id,
            FileAsset.user_id == user.id,
        )
    )
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return asset


def file_view(asset: FileAsset) -> FileView:
    return FileView(
        id=asset.id,
        owner=asset.user_id,
        project_id=asset.project_id,
        conversation_id=asset.conversation_id,
        filename=asset.filename,
        mime_type=asset.mime_type,
        size_bytes=asset.size_bytes,
        checksum_sha256=asset.checksum_sha256,
        uploaded_at=asset.uploaded_at,
        processing_status=asset.processing_status,
        storage_location=asset.storage_key,
    )


@router.get("", response_model=list[FileView])
def list_files(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[FileView]:
    assets = db.scalars(
        select(FileAsset)
        .where(FileAsset.tenant_id == user.tenant_id, FileAsset.user_id == user.id)
        .order_by(FileAsset.uploaded_at.desc())
    ).all()
    return [file_view(asset) for asset in assets]


@router.post("", response_model=FileView, status_code=status.HTTP_201_CREATED)
async def upload_file(
    request: Request,
    file: Annotated[UploadFile, File(description="File bytes")],
    conversation_id: Annotated[str | None, Form()] = None,
    project_id: Annotated[str | None, Form()] = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> FileView:
    normalized_project = project_id.strip() if project_id else None
    if normalized_project and len(normalized_project) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="project_id must be at most 100 characters",
        )
    if conversation_id is not None:
        conversation = db.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == user.tenant_id,
                Conversation.user_id == user.id,
            )
        )
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

    file_id = new_id()
    storage = request.app.state.file_storage
    try:
        stored = await storage.store(file, user.tenant_id, file_id)
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidUpload as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc

    asset = FileAsset(
        id=file_id,
        tenant_id=user.tenant_id,
        user_id=user.id,
        conversation_id=conversation_id,
        project_id=normalized_project,
        filename=stored.filename,
        extension=stored.extension,
        mime_type=stored.mime_type,
        size_bytes=stored.size_bytes,
        checksum_sha256=stored.checksum_sha256,
        processing_status="stored",
        storage_key=stored.storage_key,
    )
    try:
        db.add(asset)
        db.commit()
        db.refresh(asset)
    except Exception:
        db.rollback()
        storage.delete(stored.storage_key)
        raise
    return file_view(asset)


@router.get("/{file_id}", response_model=FileView)
def get_file(
    file_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> FileView:
    return file_view(scoped_file(db, file_id, user))


@router.get("/{file_id}/content", response_class=FileResponse)
def download_file(
    file_id: str,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    asset = scoped_file(db, file_id, user)
    path = request.app.state.file_storage.resolve(asset.storage_key)
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_410_GONE, detail="Stored file content is unavailable"
        )
    return FileResponse(path, media_type=asset.mime_type, filename=asset.filename)
