from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APEX_",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "development"
    database_url: str = "sqlite:///./data/apex.db"
    secret_key: SecretStr | None = None
    access_token_minutes: int = 15
    refresh_token_days: int = 30
    cors_origins: str = "http://localhost:3000"
    expose_development_tokens: bool = False
    issuer: str = "apex-1"
    file_storage_root: Path = Path("data/files")
    max_upload_bytes: int = 25 * 1024 * 1024
    rag_chunk_characters: int = 800
    rag_chunk_overlap: int = 120
    rag_embedding_dimensions: int = 384
    rag_max_extracted_characters: int = 2_000_000
    redis_url: str | None = None
    redis_required: bool = False
    redis_timeout_seconds: float = 1.0
    tool_max_calls: int = 8
    tool_default_timeout_seconds: float = 5.0
    tool_default_retries: int = 1
    mcp_allowed_hosts: str = ""
    mcp_protocol_version: str = "2026-07-28"
    mcp_max_response_bytes: int = 2 * 1024 * 1024
    mcp_max_discovered_tools: int = 256

    @model_validator(mode="after")
    def validate_secrets(self) -> Settings:
        if self.secret_key is None:
            if self.env.lower() == "production":
                raise ValueError("APEX_SECRET_KEY is required in production")
            self.secret_key = SecretStr(secrets.token_urlsafe(48))
        if self.max_upload_bytes < 1:
            raise ValueError("APEX_MAX_UPLOAD_BYTES must be positive")
        if self.rag_chunk_characters < 64:
            raise ValueError("APEX_RAG_CHUNK_CHARACTERS must be at least 64")
        if not 0 <= self.rag_chunk_overlap < self.rag_chunk_characters:
            raise ValueError("APEX_RAG_CHUNK_OVERLAP must be smaller than the chunk size")
        if self.rag_embedding_dimensions < 32:
            raise ValueError("APEX_RAG_EMBEDDING_DIMENSIONS must be at least 32")
        if self.rag_max_extracted_characters < self.rag_chunk_characters:
            raise ValueError(
                "APEX_RAG_MAX_EXTRACTED_CHARACTERS must be at least one chunk"
            )
        if self.tool_max_calls < 1:
            raise ValueError("APEX_TOOL_MAX_CALLS must be positive")
        if self.mcp_max_response_bytes < 1024:
            raise ValueError("APEX_MCP_MAX_RESPONSE_BYTES must be at least 1024")
        if self.mcp_max_discovered_tools < 1:
            raise ValueError("APEX_MCP_MAX_DISCOVERED_TOOLS must be positive")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def mcp_allowed_host_set(self) -> set[str]:
        return {
            host.strip().lower()
            for host in self.mcp_allowed_hosts.split(",")
            if host.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
