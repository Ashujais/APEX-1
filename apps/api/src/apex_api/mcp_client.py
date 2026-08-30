from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from apex_api.config import Settings

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
CREDENTIAL_REFERENCE_PATTERN = re.compile(r"^[A-Z0-9_]{1,64}$")


class McpClientError(RuntimeError):
    pass


class McpProtocolError(McpClientError):
    pass


@dataclass(frozen=True)
class DiscoveredMcpServer:
    capabilities: dict[str, Any]
    server_info: dict[str, Any]
    tools: list[dict[str, Any]]


def validate_mcp_endpoint(endpoint: str, settings: Settings) -> str:
    parsed = urlsplit(endpoint)
    hostname = (parsed.hostname or "").lower()
    if not hostname or parsed.username or parsed.password or parsed.fragment:
        raise McpClientError(
            "MCP endpoint must be an absolute URL without credentials or fragments"
        )
    local_allowed = settings.env.lower() != "production" and hostname in LOCAL_HOSTS
    remote_allowed = parsed.scheme == "https" and hostname in settings.mcp_allowed_host_set
    if not (
        (local_allowed and parsed.scheme in {"http", "https"})
        or remote_allowed
    ):
        raise McpClientError(
            "MCP endpoint host is not allowed; configure APEX_MCP_ALLOWED_HOSTS"
        )
    return endpoint


class StreamableHttpMcpClient:
    """Bounded MCP 2026-07-28 Streamable HTTP JSON client."""

    def __init__(
        self,
        endpoint: str,
        settings: Settings,
        *,
        timeout_seconds: float,
        credential_reference: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.endpoint = validate_mcp_endpoint(endpoint, settings)
        self.timeout_seconds = timeout_seconds
        self.credential_reference = credential_reference
        self.transport = transport
        self.request_id = 0

    def discover(self) -> DiscoveredMcpServer:
        discovery = self._request("server/discover", {})
        supported = discovery.get("supportedVersions")
        if not isinstance(supported, list) or self.settings.mcp_protocol_version not in supported:
            raise McpProtocolError("MCP server does not support the configured protocol version")
        capabilities = discovery.get("capabilities")
        if not isinstance(capabilities, dict):
            raise McpProtocolError("MCP discovery response omitted capabilities")
        if "tools" not in capabilities:
            return DiscoveredMcpServer(
                capabilities=capabilities,
                server_info=_server_info(discovery),
                tools=[],
            )

        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(10):
            params: dict[str, Any] = {}
            if cursor is not None:
                params["cursor"] = cursor
            page = self._request("tools/list", params)
            page_tools = page.get("tools")
            if not isinstance(page_tools, list):
                raise McpProtocolError("MCP tools/list response omitted tools")
            for tool in page_tools:
                if not isinstance(tool, dict):
                    raise McpProtocolError("MCP tools/list returned an invalid tool")
                tools.append(tool)
                if len(tools) > self.settings.mcp_max_discovered_tools:
                    raise McpProtocolError("MCP server returned too many tools")
            next_cursor = page.get("nextCursor")
            if next_cursor is None:
                break
            if not isinstance(next_cursor, str) or not next_cursor:
                raise McpProtocolError("MCP server returned an invalid pagination cursor")
            cursor = next_cursor
        else:
            raise McpProtocolError("MCP tools/list pagination exceeded ten pages")
        return DiscoveredMcpServer(
            capabilities=capabilities,
            server_info=_server_info(discovery),
            tools=tools,
        )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            name=name,
        )
        if not isinstance(result.get("content", []), list):
            raise McpProtocolError("MCP tool result content must be a list")
        return result

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        name: str | None = None,
    ) -> dict[str, Any]:
        self.request_id += 1
        request_params = dict(params)
        request_params["_meta"] = {
            "io.modelcontextprotocol/protocolVersion": self.settings.mcp_protocol_version,
            "io.modelcontextprotocol/clientInfo": {
                "name": "apex-1",
                "version": "0.1.0",
            },
            "io.modelcontextprotocol/clientCapabilities": {},
        }
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.settings.mcp_protocol_version,
            "Mcp-Method": method,
        }
        if name is not None:
            headers["Mcp-Name"] = name
        headers.update(self._authentication_headers())
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": request_params,
        }
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                trust_env=False,
                transport=self.transport,
            ) as client:
                response = client.post(self.endpoint, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise McpClientError("MCP server request failed") from exc
        if len(response.content) > self.settings.mcp_max_response_bytes:
            raise McpProtocolError("MCP server response exceeded the configured limit")
        body = _parse_response(response)
        if body.get("jsonrpc") != "2.0" or body.get("id") != self.request_id:
            raise McpProtocolError("MCP server returned a mismatched JSON-RPC response")
        if "error" in body:
            raise McpProtocolError("MCP server returned a JSON-RPC error")
        result = body.get("result")
        if not isinstance(result, dict):
            raise McpProtocolError("MCP server response omitted a result object")
        return result

    def _authentication_headers(self) -> dict[str, str]:
        if self.credential_reference is None:
            return {}
        if not CREDENTIAL_REFERENCE_PATTERN.fullmatch(self.credential_reference):
            raise McpClientError("MCP credential reference is invalid")
        environment_name = f"APEX_MCP_CREDENTIAL_{self.credential_reference}"
        credential = os.getenv(environment_name)
        if not credential:
            raise McpClientError("Configured MCP credential reference is unavailable")
        return {"Authorization": f"Bearer {credential}"}


def _parse_response(response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "").lower()
    try:
        if "text/event-stream" in content_type:
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    payload = json.loads(line.removeprefix("data:").strip())
                    if isinstance(payload, dict):
                        return payload
            raise McpProtocolError("MCP event stream did not contain a JSON response")
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise McpProtocolError("MCP server returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise McpProtocolError("MCP server response must be a JSON object")
    return payload


def _server_info(discovery: dict[str, Any]) -> dict[str, Any]:
    metadata = discovery.get("_meta")
    if not isinstance(metadata, dict):
        return {}
    info = metadata.get("io.modelcontextprotocol/serverInfo")
    return info if isinstance(info, dict) else {}
