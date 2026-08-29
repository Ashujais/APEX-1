from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


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
