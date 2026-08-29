from __future__ import annotations

import re
from collections.abc import Iterable

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

    def stream(self, prompt: str) -> Iterable[str]:
        normalized = re.sub(r"\s+", " ", prompt).strip()
        response = (
            "APEX Dev received the request and verified the chat pipeline end to end. "
            "This response is deterministic platform-test output, not a trained model result. "
            f"Request summary: {normalized[:240]}"
        )
        words = response.split(" ")
        for index, word in enumerate(words):
            yield word + ("" if index == len(words) - 1 else " ")
