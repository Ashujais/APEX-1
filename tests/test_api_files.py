from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from conftest import register_and_login


def authorization(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def test_upload_metadata_listing_and_content_retrieval(client: TestClient) -> None:
    tokens = register_and_login(client)
    headers = authorization(tokens)
    conversation = client.post(
        "/v1/conversations",
        headers=headers,
        json={"title": "File test", "model_id": "apex-dev"},
    ).json()
    content = b"APEX file storage keeps tenant-owned bytes outside the database.\n"

    uploaded = client.post(
        "/v1/files",
        headers=headers,
        data={"conversation_id": conversation["id"], "project_id": "apex-core"},
        files={"file": ("readiness.md", content, "text/markdown")},
    )
    assert uploaded.status_code == 201, uploaded.text
    metadata = uploaded.json()
    assert metadata["filename"] == "readiness.md"
    assert metadata["conversation_id"] == conversation["id"]
    assert metadata["project_id"] == "apex-core"
    assert metadata["mime_type"] == "text/markdown"
    assert metadata["size_bytes"] == len(content)
    assert metadata["checksum_sha256"] == hashlib.sha256(content).hexdigest()
    assert metadata["processing_status"] == "stored"
    assert metadata["storage_location"].endswith(f"{metadata['id']}.md")
    assert metadata["owner"]
    assert metadata["uploaded_at"]

    listed = client.get("/v1/files", headers=headers)
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [metadata["id"]]
    fetched = client.get(f"/v1/files/{metadata['id']}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json() == metadata
    downloaded = client.get(f"/v1/files/{metadata['id']}/content", headers=headers)
    assert downloaded.status_code == 200
    assert downloaded.content == content


def test_upload_validation_size_authentication_and_ownership(client: TestClient) -> None:
    first = register_and_login(client, "files-first@example.com")
    second = register_and_login(client, "files-second@example.com")
    first_headers = authorization(first)
    second_headers = authorization(second)

    unauthorized = client.post(
        "/v1/files",
        files={"file": ("notes.txt", b"private", "text/plain")},
    )
    assert unauthorized.status_code == 401

    invalid = client.post(
        "/v1/files",
        headers=first_headers,
        files={"file": ("payload.exe", b"MZ", "application/x-msdownload")},
    )
    assert invalid.status_code == 415

    oversized = client.post(
        "/v1/files",
        headers=first_headers,
        files={"file": ("large.txt", b"x" * 1025, "text/plain")},
    )
    assert oversized.status_code == 413

    uploaded = client.post(
        "/v1/files",
        headers=first_headers,
        files={"file": ("owned.txt", b"owner-only", "text/plain")},
    )
    assert uploaded.status_code == 201
    file_id = uploaded.json()["id"]
    assert client.get(f"/v1/files/{file_id}", headers=second_headers).status_code == 404
    assert (
        client.get(f"/v1/files/{file_id}/content", headers=second_headers).status_code
        == 404
    )
    assert client.get("/v1/files", headers=second_headers).json() == []
