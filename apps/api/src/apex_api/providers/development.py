from __future__ import annotations

import re
from collections.abc import Iterable

from apex.tooling import ModelTurn, ToolResult, ToolSpec
from apex_api.providers.base import ProviderDescriptor


class DevelopmentResponder:
    """A transparent deterministic responder for platform integration testing."""

    descriptor = ProviderDescriptor(
        id="apex-dev",
        name="APEX Dev",
        status="experimental",
        description=(
            "Deterministic responder for verifying auth, storage, and streaming; not an LLM."
        ),
        modalities=("text",),
        capabilities=("streaming", "platform-testing"),
    )

    def complete(
        self,
        prompt: str,
        tools: tuple[ToolSpec, ...] = (),
        tool_results: tuple[ToolResult, ...] = (),
    ) -> ModelTurn:
        normalized = re.sub(r"\s+", " ", prompt).strip()
        return ModelTurn(
            content=(
                "APEX Dev received the request and verified the chat pipeline end to end. "
                "This response is deterministic platform-test output, not a trained model result. "
                f"Request summary: {normalized[:240]}"
            )
        )

    def stream(self, prompt: str) -> Iterable[str]:
        response = self.complete(prompt).content
        words = response.split(" ")
        for index, word in enumerate(words):
            yield word + ("" if index == len(words) - 1 else " ")
