from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from apex_api.config import Settings
from apex_api.database import Database
from apex_api.mcp_client import McpClientError, StreamableHttpMcpClient
from apex_api.models import McpServer, ToolDefinition
from apex_api.rag_store import retrieve_chunks

EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="apex-tool")


class ToolExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        attempts: int = 0,
        duration_ms: float = 0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.attempts = attempts
        self.duration_ms = duration_ms


@dataclass(frozen=True)
class ToolOutcome:
    output: dict[str, Any]
    attempts: int
    duration_ms: float


@dataclass(frozen=True)
class ToolContext:
    database: Database
    settings: Settings
    tenant_id: str
    user_id: str
    server: McpServer
    tool: ToolDefinition


BUILTIN_TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "apex.system.status",
        "description": "Report verified local APEX runtime capability labels.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "output_schema": {
            "type": "object",
            "required": ["platform", "local_model", "trained_apex_model"],
            "properties": {
                "platform": {"type": "string"},
                "local_model": {"type": "string"},
                "trained_apex_model": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "permission": "system:read",
        "handler": "builtin.system_status",
        "status": "implemented",
    },
    {
        "name": "apex.text.statistics",
        "description": "Compute deterministic character, word, and line counts.",
        "input_schema": {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string", "maxLength": 100000}},
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "required": ["characters", "words", "lines"],
            "properties": {
                "characters": {"type": "integer", "minimum": 0},
                "words": {"type": "integer", "minimum": 0},
                "lines": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "permission": "text:analyze",
        "handler": "builtin.text_statistics",
        "status": "implemented",
    },
    {
        "name": "apex.rag.search",
        "description": "Search only the authenticated user's indexed RAG documents.",
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 8000},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "document_id": {"type": ["string", "null"], "maxLength": 36},
            },
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "required": ["results"],
            "properties": {"results": {"type": "array", "maxItems": 20}},
            "additionalProperties": False,
        },
        "permission": "rag:read",
        "handler": "builtin.rag_search",
        "status": "implemented",
    },
)


class ToolRuntime:
    def __init__(self, database: Database, settings: Settings) -> None:
        self.database = database
        self.settings = settings
        self.handlers: dict[str, Callable[[ToolContext, dict[str, Any]], dict[str, Any]]] = {
            "builtin.system_status": self._system_status,
            "builtin.text_statistics": self._text_statistics,
            "builtin.rag_search": self._rag_search,
            "mcp.remote": self._remote_tool,
        }

    def execute(
        self,
        server: McpServer,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        tenant_id: str,
        user_id: str,
        requested_timeout: float | None = None,
    ) -> ToolOutcome:
        started = time.perf_counter()
        if not server.enabled or not tool.enabled:
            raise ToolExecutionError(
                "Tool is disabled", code="tool_disabled", status_code=409
            )
        if tool.required_permission not in server.permissions:
            raise ToolExecutionError(
                "Tool permission is not granted",
                code="permission_denied",
                status_code=403,
            )
        validate_json_schema(arguments, tool.input_schema)
        handler = self.handlers.get(tool.handler)
        if handler is None:
            raise ToolExecutionError(
                "Tool handler is unavailable",
                code="handler_unavailable",
                status_code=409,
            )

        timeout = min(
            requested_timeout or self.settings.tool_default_timeout_seconds,
            server.timeout_seconds,
            30.0,
        )
        retries = server.max_retries if tool.handler == "mcp.remote" else 0
        context = ToolContext(
            database=self.database,
            settings=self.settings,
            tenant_id=tenant_id,
            user_id=user_id,
            server=server,
            tool=tool,
        )
        last_error: Exception | None = None
        for attempt in range(1, retries + 2):
            future = EXECUTOR.submit(handler, context, arguments)
            try:
                output = future.result(timeout=timeout)
                schema_target = (
                    output.get("structuredContent")
                    if tool.handler == "mcp.remote"
                    and isinstance(output.get("structuredContent"), dict)
                    else output
                )
                if tool.output_schema is not None:
                    validate_json_schema(schema_target, tool.output_schema, label="output")
                encoded = json.dumps(output, ensure_ascii=False).encode("utf-8")
                if len(encoded) > self.settings.mcp_max_response_bytes:
                    raise ToolExecutionError(
                        "Tool output exceeded the configured limit",
                        code="output_too_large",
                        status_code=502,
                    )
                return ToolOutcome(
                    output=output,
                    attempts=attempt,
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                )
            except FutureTimeout as exc:
                future.cancel()
                last_error = exc
            except ToolExecutionError:
                raise
            except McpClientError as exc:
                last_error = exc
            except Exception as exc:
                raise ToolExecutionError(
                    "Tool execution failed",
                    code="execution_failed",
                    status_code=500,
                    attempts=attempt,
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                ) from exc

        code = "tool_timeout" if isinstance(last_error, FutureTimeout) else "mcp_unavailable"
        status_code = 504 if code == "tool_timeout" else 502
        raise ToolExecutionError(
            "Tool execution timed out" if code == "tool_timeout" else "MCP tool request failed",
            code=code,
            status_code=status_code,
            attempts=retries + 1,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        ) from last_error

    @staticmethod
    def _system_status(
        _context: ToolContext, _arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "platform": "Implemented",
            "local_model": "Experimental deterministic development provider",
            "trained_apex_model": False,
        }

    @staticmethod
    def _text_statistics(
        _context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        value = arguments["text"]
        return {
            "characters": len(value),
            "words": len(value.split()),
            "lines": len(value.splitlines()) or 1,
        }

    @staticmethod
    def _rag_search(
        context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        with context.database.session_factory() as db:
            ranked = retrieve_chunks(
                db,
                context.tenant_id,
                context.user_id,
                arguments["query"],
                dimensions=context.settings.rag_embedding_dimensions,
                top_k=arguments.get("top_k", 5),
                document_id=arguments.get("document_id"),
            )
        return {
            "results": [
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "source_name": chunk.source_name,
                    "position": chunk.position,
                    "content": chunk.content,
                    "score": chunk.score,
                }
                for chunk in ranked
            ]
        }

    @staticmethod
    def _remote_tool(
        context: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if context.server.endpoint is None:
            raise ToolExecutionError(
                "MCP endpoint is missing",
                code="invalid_server",
                status_code=409,
            )
        client = StreamableHttpMcpClient(
            context.server.endpoint,
            context.settings,
            timeout_seconds=context.server.timeout_seconds,
            credential_reference=context.server.credential_reference,
        )
        return client.call_tool(context.tool.name, arguments)


def validate_json_schema(value: Any, schema: dict[str, Any], *, label: str = "input") -> None:
    if not isinstance(schema, dict):
        raise ToolExecutionError(
            f"Tool {label} schema is invalid",
            code="invalid_schema",
            status_code=409,
        )
    try:
        _validate_value(value, schema, "$")
    except ValueError as exc:
        raise ToolExecutionError(
            f"Tool {label} validation failed: {exc}",
            code=f"invalid_{label}",
            status_code=422 if label == "input" else 502,
        ) from exc


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        candidates = [candidate for candidate in expected if isinstance(candidate, str)]
        if not any(_matches_type(value, candidate) for candidate in candidates):
            raise ValueError(f"{path} has the wrong type")
    elif isinstance(expected, str) and not _matches_type(value, expected):
        raise ValueError(f"{path} must be {expected}")

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not an allowed value")
    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} is missing {', '.join(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = set(value).difference(properties)
            if extras:
                raise ValueError(f"{path} contains unsupported properties")
        for key, child in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                _validate_value(child, child_schema, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) > schema.get("maxItems", len(value)):
            raise ValueError(f"{path} contains too many items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, child in enumerate(value):
                _validate_value(child, item_schema, f"{path}[{index}]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path} is too short")
        if len(value) > schema.get("maxLength", len(value)):
            raise ValueError(f"{path} is too long")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", value):
            raise ValueError(f"{path} is below the minimum")
        if value > schema.get("maximum", value):
            raise ValueError(f"{path} exceeds the maximum")


def _matches_type(value: Any, expected: str) -> bool:
    mapping = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    checker = mapping.get(expected)
    return True if checker is None else checker(value)
