from __future__ import annotations

import secrets
from functools import lru_cache

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

    @model_validator(mode="after")
    def validate_secrets(self) -> Settings:
        if self.secret_key is None:
            if self.env.lower() == "production":
                raise ValueError("APEX_SECRET_KEY is required in production")
            self.secret_key = SecretStr(secrets.token_urlsafe(48))
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
