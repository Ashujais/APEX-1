from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import register_and_login


def test_health_and_model_registry_are_honest(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["checks"]["database"] is True
    models = client.get("/v1/models").json()
    assert [model["id"] for model in models] == ["apex-dev"]
    assert "not an LLM" in models[0]["description"]


def test_registration_verification_login_and_refresh_rotation(client: TestClient) -> None:
    password = "correct horse battery staple"
    response = client.post(
        "/v1/auth/register",
        json={"email": "owner@example.com", "name": "Owner", "password": password},
    )
    assert response.status_code == 201
    token = response.json()["development_verification_token"]
    assert token
    blocked = client.post(
        "/v1/auth/login", json={"email": "owner@example.com", "password": password}
    )
    assert blocked.status_code == 403

    assert client.post("/v1/auth/verify-email", json={"token": token}).status_code == 200
    login = client.post("/v1/auth/login", json={"email": "owner@example.com", "password": password})
    assert login.status_code == 200
    first_refresh = login.json()["refresh_token"]
    me = client.get(
        "/v1/auth/me", headers={"Authorization": f"Bearer {login.json()['access_token']}"}
    )
    assert me.status_code == 200
    assert me.json()["email"] == "owner@example.com"

    rotated = client.post("/v1/auth/refresh", json={"refresh_token": first_refresh})
    assert rotated.status_code == 200
    assert rotated.json()["refresh_token"] != first_refresh
    assert client.post("/v1/auth/refresh", json={"refresh_token": first_refresh}).status_code == 401

    browser_rotated = client.post("/v1/auth/browser/refresh")
    assert browser_rotated.status_code == 200

    logout = client.post(
        "/v1/auth/logout", json={"refresh_token": browser_rotated.json()["refresh_token"]}
    )
    assert logout.status_code == 204
    revoked = client.get(
        "/v1/auth/me",
        headers={"Authorization": f"Bearer {browser_rotated.json()['access_token']}"},
    )
    assert revoked.status_code == 401


def test_password_reset_revokes_existing_sessions(client: TestClient) -> None:
    tokens = register_and_login(client)
    reset = client.post("/v1/auth/password-reset/request", json={"email": "researcher@example.com"})
    reset_token = reset.json()["development_reset_token"]
    assert reset_token
    confirmed = client.post(
        "/v1/auth/password-reset/confirm",
        json={"token": reset_token, "new_password": "a completely new passphrase"},
    )
    assert confirmed.status_code == 204
    assert (
        client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code
        == 401
    )
