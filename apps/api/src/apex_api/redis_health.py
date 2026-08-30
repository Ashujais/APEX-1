from __future__ import annotations

import socket
import ssl
from urllib.parse import unquote, urlsplit


def redis_ping(url: str, timeout_seconds: float) -> bool:
    """Perform a bounded RESP PING without logging or returning connection details."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"redis", "rediss"} or parsed.hostname is None:
            return False
        port = parsed.port or (6380 if parsed.scheme == "rediss" else 6379)
        commands: list[list[str]] = []
        password = unquote(parsed.password) if parsed.password is not None else None
        username = unquote(parsed.username) if parsed.username is not None else None
        if password is not None:
            commands.append(["AUTH", username, password] if username else ["AUTH", password])
        database_text = parsed.path.lstrip("/") or "0"
        database = int(database_text)
        if database < 0:
            return False
        if database:
            commands.append(["SELECT", str(database)])
        commands.append(["PING"])

        with socket.create_connection(
            (parsed.hostname, port), timeout=timeout_seconds
        ) as connection:
            connection.settimeout(timeout_seconds)
            stream: socket.socket
            if parsed.scheme == "rediss":
                stream = ssl.create_default_context().wrap_socket(
                    connection, server_hostname=parsed.hostname
                )
            else:
                stream = connection
            with stream.makefile("rwb", buffering=0) as wire:
                responses = []
                for command in commands:
                    wire.write(_encode_command(command))
                    responses.append(wire.readline(1024))
        return responses[-1] == b"+PONG\r\n" and all(
            response and not response.startswith(b"-") for response in responses
        )
    except (OSError, ValueError):
        return False


def _encode_command(parts: list[str]) -> bytes:
    encoded = [part.encode("utf-8") for part in parts]
    payload = [f"*{len(encoded)}\r\n".encode()]
    for part in encoded:
        payload.extend((f"${len(part)}\r\n".encode(), part, b"\r\n"))
    return b"".join(payload)
