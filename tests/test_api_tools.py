from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import register_and_login


def authorization(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def register_builtin(
    client: TestClient, headers: dict[str, str], permissions: list[str] | None = None
) -> dict[str, object]:
    response = client.post(
        "/v1/mcp/servers",
        headers=headers,
        json={
            "name": "APEX built-ins",
            "transport": "builtin",
            **({"permissions": permissions} if permissions is not None else {}),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_builtin_registry_execution_rag_and_audit(client: TestClient) -> None:
    tokens = register_and_login(client)
    headers = authorization(tokens)
    server = register_builtin(client, headers)
    assert server["capability_status"] == "implemented"
    assert "credential_reference" not in server

    tools = client.get(
        f"/v1/mcp/servers/{server['id']}/tools", headers=headers
    ).json()
    text_tool = next(tool for tool in tools if tool["name"] == "apex.text.statistics")
    stats = client.post(
        f"/v1/tools/{text_tool['id']}/execute",
        headers={**headers, "X-Request-ID": "tool-test-request"},
        json={"arguments": {"text": "one two\nthree"}},
    )
    assert stats.status_code == 200, stats.text
    assert stats.json()["output"] == {"characters": 13, "words": 3, "lines": 2}
    assert stats.json()["request_id"] == "tool-test-request"

    invalid = client.post(
        f"/v1/tools/{text_tool['id']}/execute",
        headers=headers,
        json={"arguments": {"unexpected": True}},
    )
    assert invalid.status_code == 422

    ingested = client.post(
        "/v1/rag/ingest",
        headers=headers,
        json={
            "text": "APEX records reproducible evidence for every verified benchmark.",
            "source_name": "tool-corpus",
        },
    )
    assert ingested.status_code == 201, ingested.text
    rag_tool = next(tool for tool in tools if tool["name"] == "apex.rag.search")
    searched = client.post(
        f"/v1/tools/{rag_tool['id']}/execute",
        headers=headers,
        json={"arguments": {"query": "reproducible evidence", "top_k": 2}},
    )
    assert searched.status_code == 200, searched.text
    results = searched.json()["output"]["results"]
    assert results
    assert results[0]["document_id"] == ingested.json()["id"]

    audit = client.get("/v1/tools/audit", headers=headers)
    assert audit.status_code == 200
    entries = audit.json()
    assert {entry["outcome"] for entry in entries} == {"completed", "failed"}
    assert all("arguments" not in entry and "output" not in entry for entry in entries)


def test_tool_permission_and_tenant_ownership(client: TestClient) -> None:
    first = register_and_login(client, "tools-first@example.com")
    second = register_and_login(client, "tools-second@example.com")
    first_headers = authorization(first)
    second_headers = authorization(second)
    server = register_builtin(client, first_headers, permissions=["system:read"])
    tools = client.get(
        f"/v1/mcp/servers/{server['id']}/tools", headers=first_headers
    ).json()
    text_tool = next(tool for tool in tools if tool["name"] == "apex.text.statistics")

    denied = client.post(
        f"/v1/tools/{text_tool['id']}/execute",
        headers=first_headers,
        json={"arguments": {"text": "private"}},
    )
    assert denied.status_code == 403
    assert client.get("/v1/mcp/servers", headers=second_headers).json() == []
    assert (
        client.get(
            f"/v1/mcp/servers/{server['id']}", headers=second_headers
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/v1/tools/{text_tool['id']}/execute",
            headers=second_headers,
            json={"arguments": {"text": "private"}},
        ).status_code
        == 404
    )
