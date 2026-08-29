from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import register_and_login


def authorization(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def test_authenticated_chat_stream_is_persisted(client: TestClient) -> None:
    tokens = register_and_login(client)
    created = client.post(
        "/v1/conversations",
        headers=authorization(tokens),
        json={"title": "Pipeline check", "model_id": "apex-dev"},
    )
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    with client.stream(
        "POST",
        f"/v1/conversations/{conversation_id}/stream",
        headers=authorization(tokens),
        json={"prompt": "Verify the streaming path"},
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    assert "event: metadata" in body
    assert "event: delta" in body
    assert "event: done" in body

    persisted = client.get(f"/v1/conversations/{conversation_id}", headers=authorization(tokens))
    assert [message["role"] for message in persisted.json()["messages"]] == ["user", "assistant"]
    assert "not a trained model result" in persisted.json()["messages"][1]["content"]


def test_conversations_are_isolated_between_personal_tenants(client: TestClient) -> None:
    first = register_and_login(client, "first@example.com")
    second = register_and_login(client, "second@example.com")
    conversation = client.post(
        "/v1/conversations",
        headers=authorization(first),
        json={"title": "Private", "model_id": "apex-dev"},
    ).json()
    assert (
        client.get(
            f"/v1/conversations/{conversation['id']}", headers=authorization(second)
        ).status_code
        == 404
    )
    assert client.get("/v1/conversations", headers=authorization(second)).json() == []
