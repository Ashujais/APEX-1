from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from conftest import register_and_login


def authorization(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def test_tiny_document_ingest_query_relevance_and_citations(client: TestClient) -> None:
    tokens = register_and_login(client)
    headers = authorization(tokens)
    corpus = Path("examples/data/tiny-corpus.txt").read_bytes()
    uploaded = client.post(
        "/v1/files",
        headers=headers,
        files={"file": ("tiny-corpus.txt", corpus, "text/plain")},
    )
    assert uploaded.status_code == 201, uploaded.text
    file_id = uploaded.json()["id"]

    ingested = client.post(
        "/v1/rag/ingest",
        headers=headers,
        json={
            "file_id": file_id,
            "source_name": "APEX tiny corpus",
            "metadata": {"purpose": "readiness"},
        },
    )
    assert ingested.status_code == 201, ingested.text
    document = ingested.json()
    assert document["file_id"] == file_id
    assert document["processing_status"] == "indexed"
    assert document["chunk_count"] >= 1

    query = client.post(
        "/v1/rag/query",
        headers=headers,
        json={
            "query": "What reproducible evidence does APEX record?",
            "document_id": document["id"],
            "top_k": 2,
        },
    )
    assert query.status_code == 200, query.text
    result = query.json()
    assert result["results"]
    assert "reproducible evidence" in result["results"][0]["content"].lower()
    assert result["results"][0]["score"] > 0
    assert result["citations"][0]["document_id"] == document["id"]
    assert result["citations"][0]["file_id"] == file_id
    assert result["context"]
    assert result["model_id"] == "apex-dev"
    assert result["model_status"] == "experimental"
    assert "not a trained model result" in result["answer"]

    refreshed_file = client.get(f"/v1/files/{file_id}", headers=headers)
    assert refreshed_file.json()["processing_status"] == "indexed"


def test_rag_requires_authentication_and_enforces_ownership(client: TestClient) -> None:
    first = register_and_login(client, "rag-first@example.com")
    second = register_and_login(client, "rag-second@example.com")
    first_headers = authorization(first)
    second_headers = authorization(second)

    assert (
        client.post(
            "/v1/rag/ingest",
            json={"text": "Private tenant context", "source_name": "private"},
        ).status_code
        == 401
    )
    document = client.post(
        "/v1/rag/ingest",
        headers=first_headers,
        json={"text": "Private tenant context", "source_name": "private"},
    )
    assert document.status_code == 201
    document_id = document.json()["id"]
    hidden = client.post(
        "/v1/rag/query",
        headers=second_headers,
        json={"query": "private context", "document_id": document_id},
    )
    assert hidden.status_code == 404
    assert client.get("/v1/rag/documents", headers=second_headers).json() == []
