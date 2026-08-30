from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from apex.tooling import ModelTurn, ToolResult, ToolSpec


@dataclass(frozen=True)
class ProviderDescriptor:
    id: str
    name: str
    status: str
    description: str
    modalities: tuple[str, ...]
    capabilities: tuple[str, ...]


class ChatProvider(Protocol):
    descriptor: ProviderDescriptor

    def complete(
        self,
        prompt: str,
        tools: tuple[ToolSpec, ...] = (),
        tool_results: tuple[ToolResult, ...] = (),
    ) -> ModelTurn: ...

    def stream(self, prompt: str) -> Iterable[str]: ...
