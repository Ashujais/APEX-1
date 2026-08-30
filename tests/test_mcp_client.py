from __future__ import annotations

import httpx
from pydantic import SecretStr

from apex_api.config import Settings
from apex_api.mcp_client import StreamableHttpMcpClient


def test_current_mcp_discovery_and_tool_call(monkeypatch) -> None:
    monkeypatch.setenv("APEX_MCP_CREDENTIAL_TEST_TOKEN", "not-logged-test-token")
    calls: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        calls.append((payload["method"], request.headers.get("Mcp-Name")))
        assert request.headers["MCP-Protocol-Version"] == "2026-07-28"
        assert request.headers["Authorization"] == "Bearer not-logged-test-token"
        if payload["method"] == "server/discover":
            result = {
                "supportedVersions": ["2026-07-28"],
                "capabilities": {"tools": {}},
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "fixture",
                        "version": "1",
                    }
                },
            }
        elif payload["method"] == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "fixture.echo",
                        "description": "Echo a value",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                    }
                ]
            }
        else:
            assert payload["params"]["arguments"] == {"value": "hello"}
            result = {"content": [{"type": "text", "text": "hello"}]}
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
        )

    settings = Settings(
        env="test",
        secret_key=SecretStr("test-secret-that-is-long-and-random-enough"),
    )
    client = StreamableHttpMcpClient(
        "http://127.0.0.1:8123/mcp",
        settings,
        timeout_seconds=1,
        credential_reference="TEST_TOKEN",
        transport=httpx.MockTransport(handler),
    )
    discovery = client.discover()
    assert discovery.server_info["name"] == "fixture"
    assert discovery.tools[0]["name"] == "fixture.echo"
    result = client.call_tool("fixture.echo", {"value": "hello"})
    assert result["content"][0]["text"] == "hello"
    assert calls == [
        ("server/discover", None),
        ("tools/list", None),
        ("tools/call", "fixture.echo"),
    ]
