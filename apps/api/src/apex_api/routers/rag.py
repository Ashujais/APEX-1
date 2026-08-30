from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from apex_api.dependencies import current_user, get_db
from apex_api.models import FileAsset, RagChunk, RagDocument, User, utc_now
from apex_api.rag import ExtractionError, chunk_text, clean_text, embed_text, extract_text
from apex_api.rag_store import retrieve_chunks
from apex_api.schemas import (
    RagCitation,
    RagDocumentView,
    RagIngestRequest,
    RagQueryRequest,
    RagQueryResponse,
    RagSearchResult,
)

router = APIRouter(prefix="/v1/rag", tags=["retrieval"])


def document_view(document: RagDocument) -> RagDocumentView:
    return RagDocumentView(
        id=document.id,
        file_id=document.file_id,
        source_name=document.source_name,
        checksum_sha256=document.checksum_sha256,
        processing_status=document.processing_status,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.get("/documents", response_model=list[RagDocumentView])
def list_documents(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[RagDocumentView]:
    documents = db.scalars(
        select(RagDocument)
        .where(RagDocument.tenant_id == user.tenant_id, RagDocument.user_id == user.id)
        .order_by(RagDocument.created_at.desc())
    ).all()
    return [document_view(document) for document in documents]


@router.post(
    "/ingest", response_model=RagDocumentView, status_code=status.HTTP_201_CREATED
)
def ingest_document(
    payload: RagIngestRequest,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> RagDocumentView:
    settings = request.app.state.settings
    asset: FileAsset | None = None
    if payload.file_id is not None:
        asset = db.scalar(
            select(FileAsset).where(
                FileAsset.id == payload.file_id,
                FileAsset.tenant_id == user.tenant_id,
                FileAsset.user_id == user.id,
            )
        )
        if asset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
        asset.processing_status = "processing"
        db.commit()
        source_name = payload.source_name or asset.filename
        try:
            path = request.app.state.file_storage.resolve(asset.storage_key)
            text = extract_text(
                path, asset.extension, settings.rag_max_extracted_characters
            )
        except (ExtractionError, OSError, ValueError) as exc:
            asset.processing_status = "failed"
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        checksum = asset.checksum_sha256
    else:
        text = clean_text(payload.text or "")
        if not text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No extractable text was provided",
            )
        source_name = payload.source_name or "inline-document"
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()

    chunks = chunk_text(
        text, settings.rag_chunk_characters, settings.rag_chunk_overlap
    )
    if not chunks:
        if asset is not None:
            asset.processing_status = "failed"
            db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document did not produce any searchable chunks",
        )

    document = RagDocument(
        tenant_id=user.tenant_id,
        user_id=user.id,
        file_id=asset.id if asset is not None else None,
        source_name=source_name,
        checksum_sha256=checksum,
        processing_status="processing",
        chunk_count=0,
    )
    db.add(document)
    db.flush()
    for position, content in enumerate(chunks):
        db.add(
            RagChunk(
                document_id=document.id,
                tenant_id=user.tenant_id,
                user_id=user.id,
                position=position,
                content=content,
                character_count=len(content),
                embedding=embed_text(content, settings.rag_embedding_dimensions),
                metadata_json={
                    **payload.metadata,
                    "source_name": source_name,
                    "file_id": document.file_id,
                },
            )
        )
    document.chunk_count = len(chunks)
    document.processing_status = "indexed"
    document.updated_at = utc_now()
    if asset is not None:
        asset.processing_status = "indexed"
    db.commit()
    db.refresh(document)
    return document_view(document)


@router.post("/query", response_model=RagQueryResponse)
def query_documents(
    payload: RagQueryRequest,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> RagQueryResponse:
    if payload.document_id is not None:
        owned_document = db.scalar(
            select(RagDocument).where(
                RagDocument.id == payload.document_id,
                RagDocument.tenant_id == user.tenant_id,
                RagDocument.user_id == user.id,
            )
        )
        if owned_document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="RAG document not found"
            )
    settings = request.app.state.settings
    ranked = retrieve_chunks(
        db,
        user.tenant_id,
        user.id,
        payload.query,
        dimensions=settings.rag_embedding_dimensions,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )
    results: list[RagSearchResult] = []
    for item in ranked:
        citation = RagCitation(
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            file_id=item.file_id,
            source_name=item.source_name,
            chunk_position=item.position,
        )
        results.append(
            RagSearchResult(
                content=item.content,
                score=item.score,
                vector_score=item.vector_score,
                lexical_score=item.lexical_score,
                citation=citation,
            )
        )
    context = "\n\n".join(
        f"[{index}] {result.citation.source_name} "
        f"(chunk {result.citation.chunk_position})\n{result.content}"
        for index, result in enumerate(results, start=1)
    )
    try:
        provider = request.app.state.model_router.get(payload.model_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if context:
        model_prompt = (
            "Use the retrieved context below when responding. "
            "Citations are numbered and must remain attributable.\n\n"
            f"{context}\n\nQuestion: {payload.query}"
        )
        answer = "".join(provider.stream(model_prompt))
    else:
        answer = (
            "No relevant indexed context was found. "
            "The development responder was not asked to invent an answer."
        )
    return RagQueryResponse(
        query=payload.query,
        context=context,
        results=results,
        citations=[result.citation for result in results],
        model_id=provider.descriptor.id,
        model_status=provider.descriptor.status,
        answer=answer,
    )
