from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from apex_api.models import RagChunk, RagDocument
from apex_api.rag import cosine_similarity, embed_text, lexical_relevance


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    document_id: str
    file_id: str | None
    source_name: str
    position: int
    content: str
    score: float
    vector_score: float
    lexical_score: float


def retrieve_chunks(
    db: Session,
    tenant_id: str,
    user_id: str,
    query: str,
    *,
    dimensions: int,
    top_k: int,
    document_id: str | None = None,
) -> list[RankedChunk]:
    statement = (
        select(RagChunk, RagDocument)
        .join(RagDocument, RagDocument.id == RagChunk.document_id)
        .where(
            RagChunk.tenant_id == tenant_id,
            RagChunk.user_id == user_id,
            RagDocument.tenant_id == tenant_id,
            RagDocument.user_id == user_id,
            RagDocument.processing_status == "indexed",
        )
    )
    if document_id is not None:
        statement = statement.where(RagDocument.id == document_id)
    query_embedding = embed_text(query, dimensions)
    ranked: list[RankedChunk] = []
    for chunk, document in db.execute(statement):
        vector_score = max(
            0.0,
            cosine_similarity(query_embedding, [float(value) for value in chunk.embedding]),
        )
        lexical_score = lexical_relevance(query, chunk.content)
        score = 0.8 * vector_score + 0.2 * lexical_score
        if score <= 0:
            continue
        ranked.append(
            RankedChunk(
                chunk_id=chunk.id,
                document_id=document.id,
                file_id=document.file_id,
                source_name=document.source_name,
                position=chunk.position,
                content=chunk.content,
                score=round(score, 6),
                vector_score=round(vector_score, 6),
                lexical_score=round(lexical_score, 6),
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.position, item.chunk_id))
    return ranked[:top_k]
