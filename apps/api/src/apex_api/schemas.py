from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=12, max_length=256)


class VerifyEmailRequest(BaseModel):
    token: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=12, max_length=256)


class UserView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr
    name: str
    email_verified: bool
    created_at: datetime


class AuthTokens(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RegisterResponse(BaseModel):
    user: UserView
    verification_required: bool = True
    development_verification_token: str | None = None


class PasswordResetResponse(BaseModel):
    message: str
    development_reset_token: str | None = None


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=200)
    model_id: str = "apex-dev"


class MessageView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    model_id: str | None
    created_at: datetime


class ConversationView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    model_id: str
    archived: bool
    pinned: bool
    created_at: datetime
    updated_at: datetime
    messages: list[MessageView] = Field(default_factory=list)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)


class ModelView(BaseModel):
    id: str
    name: str
    status: str
    description: str
    modalities: list[str]
    capabilities: list[str]


class FileView(BaseModel):
    id: str
    owner: str
    project_id: str | None
    conversation_id: str | None
    filename: str
    mime_type: str
    size_bytes: int
    checksum_sha256: str
    uploaded_at: datetime
    processing_status: str
    storage_location: str


class RagIngestRequest(BaseModel):
    file_id: str | None = None
    text: str | None = Field(default=None, min_length=1, max_length=1_000_000)
    source_name: str | None = Field(default=None, min_length=1, max_length=255)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source(self) -> RagIngestRequest:
        if (self.file_id is None) == (self.text is None):
            raise ValueError("Provide exactly one of file_id or text")
        return self


class RagDocumentView(BaseModel):
    id: str
    file_id: str | None
    source_name: str
    checksum_sha256: str
    processing_status: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8_000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    model_id: str = "apex-dev"


class RagCitation(BaseModel):
    document_id: str
    chunk_id: str
    file_id: str | None
    source_name: str
    chunk_position: int


class RagSearchResult(BaseModel):
    content: str
    score: float
    vector_score: float
    lexical_score: float
    citation: RagCitation


class RagQueryResponse(BaseModel):
    query: str
    context: str
    results: list[RagSearchResult]
    citations: list[RagCitation]
    model_id: str
    model_status: str
    answer: str


class McpServerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    transport: Literal["builtin", "streamable-http"] = "builtin"
    endpoint: str | None = Field(default=None, max_length=500)
    authentication_mode: Literal["none", "bearer-reference"] = "none"
    credential_reference: str | None = Field(
        default=None, pattern=r"^[A-Z0-9_]{1,64}$"
    )
    permissions: list[str] = Field(
        default_factory=lambda: [
            "system:read",
            "text:analyze",
            "rag:read",
            "mcp:invoke",
        ],
        max_length=64,
    )
    timeout_seconds: float = Field(default=5.0, ge=0.1, le=30.0)
    max_retries: int = Field(default=1, ge=0, le=3)

    @model_validator(mode="after")
    def validate_transport(self) -> McpServerCreate:
        if len(set(self.permissions)) != len(self.permissions):
            raise ValueError("permissions must be unique")
        if any(not permission or len(permission) > 120 for permission in self.permissions):
            raise ValueError("permissions must be between 1 and 120 characters")
        if self.transport == "builtin":
            if (
                self.endpoint is not None
                or self.authentication_mode != "none"
                or self.credential_reference is not None
            ):
                raise ValueError("builtin servers cannot configure an endpoint or credentials")
            return self
        if self.endpoint is None:
            raise ValueError("streamable-http servers require an endpoint")
        if self.authentication_mode == "bearer-reference":
            if self.credential_reference is None:
                raise ValueError("bearer-reference authentication requires a credential reference")
        elif self.credential_reference is not None:
            raise ValueError("credential references require bearer-reference authentication")
        return self


class McpServerView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str
    transport: str
    endpoint: str | None
    authentication_mode: str
    capabilities: dict[str, object]
    capability_status: str
    permissions: list[str]
    timeout_seconds: float
    max_retries: int
    enabled: bool
    created_at: datetime
    updated_at: datetime


class ToolView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    server_id: str
    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object] | None
    required_permission: str
    status: str
    enabled: bool
    created_at: datetime


class ToolExecutionRequest(BaseModel):
    arguments: dict[str, object] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, ge=0.1, le=30.0)


class ToolExecutionResponse(BaseModel):
    tool_id: str
    request_id: str
    status: Literal["completed"]
    output: dict[str, object]
    attempts: int
    duration_ms: float


class ToolAuditView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    server_id: str
    tool_id: str
    request_id: str
    outcome: str
    attempts: int
    duration_ms: float
    error_code: str | None
    created_at: datetime
