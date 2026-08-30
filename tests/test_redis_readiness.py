from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from apex_api.config import Settings
from apex_api.main import create_app


def settings(tmp_path: Path, *, required: bool) -> Settings:
    return Settings(
        env="test",
        database_url=f"sqlite:///{tmp_path / 'redis-ready.db'}",
        secret_key=SecretStr("test-secret-that-is-long-and-random-enough"),
        file_storage_root=tmp_path / "files",
        redis_url="redis://127.0.0.1:6379/0",
        redis_required=required,
    )


def test_required_redis_controls_readiness(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("apex_api.routers.health.redis_ping", lambda *_args: False)
    with TestClient(create_app(settings(tmp_path, required=True))) as client:
        readiness = client.get("/ready").json()
    assert readiness["status"] == "not_ready"
    assert readiness["checks"]["redis"]["configured"] is True
    assert readiness["checks"]["redis"]["required"] is True
    assert readiness["checks"]["redis"]["status"] == "unavailable"


def test_configured_optional_redis_reports_available(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("apex_api.routers.health.redis_ping", lambda *_args: True)
    with TestClient(create_app(settings(tmp_path, required=False))) as client:
        readiness = client.get("/ready").json()
    assert readiness["status"] == "ready"
    assert readiness["checks"]["redis"]["available"] is True
    assert readiness["checks"]["redis"]["status"] == "available"
