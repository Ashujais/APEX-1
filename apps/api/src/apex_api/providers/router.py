from __future__ import annotations

from apex_api.providers.base import ChatProvider, ProviderDescriptor
from apex_api.providers.development import DevelopmentResponder


class ModelRouter:
    def __init__(self, providers: list[ChatProvider] | None = None) -> None:
        configured = providers or [DevelopmentResponder()]
        self._providers = {provider.descriptor.id: provider for provider in configured}

    def get(self, model_id: str) -> ChatProvider:
        try:
            return self._providers[model_id]
        except KeyError as exc:
            raise LookupError(f"Model {model_id!r} is not configured") from exc

    def list(self) -> list[ProviderDescriptor]:
        return [provider.descriptor for provider in self._providers.values()]
