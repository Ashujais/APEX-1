from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from apex_api.dependencies import current_user, get_db
from apex_api.mcp_client import McpClientError, McpProtocolError, StreamableHttpMcpClient
from apex_api.models import McpServer, ToolAuditLog, ToolDefinition, User
from apex_api.schemas import (
    McpServerCreate,
    McpServerView,
    ToolAuditView,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolView,
)
from apex_api.tool_runtime import (
    BUILTIN_TOOL_SPECS,
    ToolExecutionError,
    ToolRuntime,
)

mcp_router = APIRouter(prefix="/v1/mcp", tags=["mcp"])
tools_router = APIRouter(prefix="/v1/tools", tags=["tools"])


def scoped_server(db: Session, server_id: str, user: User) -> McpServer:
    server = db.scalar(
        select(McpServer).where(
            McpServer.id == server_id,
            McpServer.tenant_id == user.tenant_id,
            McpServer.user_id == user.id,
        )
    )
    if server is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found")
    return server


def scoped_tool(db: Session, tool_id: str, user: User) -> ToolDefinition:
    tool = db.scalar(
        select(ToolDefinition).where(
            ToolDefinition.id == tool_id,
            ToolDefinition.tenant_id == user.tenant_id,
            ToolDefinition.user_id == user.id,
        )
    )
    if tool is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool not found")
    return tool


@mcp_router.get("/servers", response_model=list[McpServerView])
def list_servers(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[McpServer]:
    return list(
        db.scalars(
            select(McpServer)
            .where(McpServer.tenant_id == user.tenant_id, McpServer.user_id == user.id)
            .order_by(McpServer.created_at.desc())
        ).all()
    )


@mcp_router.post(
    "/servers", response_model=McpServerView, status_code=status.HTTP_201_CREATED
)
def register_server(
    payload: McpServerCreate,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> McpServer:
    capabilities: dict[str, Any]
    discovered_tools: list[dict[str, Any]]
    if payload.transport == "builtin":
        capabilities = {"tools": {"listChanged": False}}
        discovered_tools = list(BUILTIN_TOOL_SPECS)
        capability_status = "implemented"
    else:
        try:
            client = StreamableHttpMcpClient(
                payload.endpoint or "",
                request.app.state.settings,
                timeout_seconds=payload.timeout_seconds,
                credential_reference=payload.credential_reference,
            )
            discovery = client.discover()
        except (McpClientError, McpProtocolError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="MCP server discovery failed",
            ) from exc
        capabilities = {
            **discovery.capabilities,
            "serverInfo": discovery.server_info,
        }
        discovered_tools = _normalize_remote_tools(discovery.tools)
        capability_status = "experimental"

    server = McpServer(
        tenant_id=user.tenant_id,
        user_id=user.id,
        name=payload.name,
        description=payload.description,
        transport=payload.transport,
        endpoint=payload.endpoint,
        authentication_mode=payload.authentication_mode,
        credential_reference=payload.credential_reference,
        capabilities=capabilities,
        capability_status=capability_status,
        permissions=payload.permissions,
        timeout_seconds=payload.timeout_seconds,
        max_retries=payload.max_retries,
    )
    db.add(server)
    db.flush()
    for specification in discovered_tools:
        db.add(
            ToolDefinition(
                server_id=server.id,
                tenant_id=user.tenant_id,
                user_id=user.id,
                name=specification["name"],
                description=specification["description"],
                input_schema=specification["input_schema"],
                output_schema=specification.get("output_schema"),
                required_permission=specification.get("permission", "mcp:invoke"),
                handler=specification.get("handler", "mcp.remote"),
                status=specification.get("status", "experimental"),
            )
        )
    try:
        db.commit()
        db.refresh(server)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="MCP server returned duplicate tool names",
        ) from exc
    return server


@mcp_router.get("/servers/{server_id}", response_model=McpServerView)
def get_server(
    server_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> McpServer:
    return scoped_server(db, server_id, user)


@mcp_router.get("/servers/{server_id}/tools", response_model=list[ToolView])
def list_server_tools(
    server_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[ToolDefinition]:
    server = scoped_server(db, server_id, user)
    return list(
        db.scalars(
            select(ToolDefinition)
            .where(
                ToolDefinition.server_id == server.id,
                ToolDefinition.tenant_id == user.tenant_id,
                ToolDefinition.user_id == user.id,
            )
            .order_by(ToolDefinition.name)
        ).all()
    )


@tools_router.get("", response_model=list[ToolView])
def list_tools(
    server_id: str | None = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[ToolDefinition]:
    statement = select(ToolDefinition).where(
        ToolDefinition.tenant_id == user.tenant_id,
        ToolDefinition.user_id == user.id,
    )
    if server_id is not None:
        scoped_server(db, server_id, user)
        statement = statement.where(ToolDefinition.server_id == server_id)
    return list(db.scalars(statement.order_by(ToolDefinition.name)).all())


@tools_router.get("/audit", response_model=list[ToolAuditView])
def list_tool_audit(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[ToolAuditLog]:
    return list(
        db.scalars(
            select(ToolAuditLog)
            .where(
                ToolAuditLog.tenant_id == user.tenant_id,
                ToolAuditLog.user_id == user.id,
            )
            .order_by(ToolAuditLog.created_at.desc())
            .limit(200)
        ).all()
    )


@tools_router.post("/{tool_id}/execute", response_model=ToolExecutionResponse)
def execute_tool(
    tool_id: str,
    payload: ToolExecutionRequest,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> ToolExecutionResponse:
    tool = scoped_tool(db, tool_id, user)
    server = scoped_server(db, tool.server_id, user)
    runtime: ToolRuntime = request.app.state.tool_runtime
    request_id = request.state.request_id
    try:
        outcome = runtime.execute(
            server,
            tool,
            payload.arguments,
            user.tenant_id,
            user.id,
            requested_timeout=payload.timeout_seconds,
        )
    except ToolExecutionError as exc:
        db.add(
            ToolAuditLog(
                tenant_id=user.tenant_id,
                user_id=user.id,
                server_id=server.id,
                tool_id=tool.id,
                request_id=request_id,
                outcome="failed",
                attempts=exc.attempts,
                duration_ms=exc.duration_ms,
                error_code=exc.code,
            )
        )
        db.commit()
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    db.add(
        ToolAuditLog(
            tenant_id=user.tenant_id,
            user_id=user.id,
            server_id=server.id,
            tool_id=tool.id,
            request_id=request_id,
            outcome="completed",
            attempts=outcome.attempts,
            duration_ms=outcome.duration_ms,
        )
    )
    db.commit()
    return ToolExecutionResponse(
        tool_id=tool.id,
        request_id=request_id,
        status="completed",
        output=outcome.output,
        attempts=outcome.attempts,
        duration_ms=outcome.duration_ms,
    )


def _normalize_remote_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        name = tool.get("name")
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema")
        output_schema = tool.get("outputSchema")
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 160
            or name in names
            or not isinstance(description, str)
            or not isinstance(input_schema, dict)
            or (output_schema is not None and not isinstance(output_schema, dict))
        ):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="MCP server returned invalid tool metadata",
            )
        names.add(name)
        normalized.append(
            {
                "name": name,
                "description": description[:500],
                "input_schema": input_schema,
                "output_schema": output_schema,
                "permission": "mcp:invoke",
                "handler": "mcp.remote",
                "status": "experimental",
            }
        )
    return normalized
