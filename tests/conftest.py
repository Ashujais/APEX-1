from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from apex_api.config import Settings
from apex_api.main import create_app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        env="test",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        secret_key=SecretStr("test-secret-that-is-long-and-random-enough-for-tests"),
        expose_development_tokens=True,
        file_storage_root=tmp_path / "files",
        max_upload_bytes=1024,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def register_and_login(client: TestClient, email: str = "researcher@example.com") -> dict[str, str]:
    password = "correct horse battery staple"
    registration = client.post(
        "/v1/auth/register",
        json={"email": email, "name": "Apex Researcher", "password": password},
    )
    assert registration.status_code == 201, registration.text
    verification_token = registration.json()["development_verification_token"]
    verified = client.post("/v1/auth/verify-email", json={"token": verification_token})
    assert verified.status_code == 200, verified.text
    login = client.post("/v1/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    return login.json()
