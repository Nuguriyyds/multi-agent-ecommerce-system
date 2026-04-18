from __future__ import annotations

import logging

from app.v3.models import CapabilityDescriptor, CapabilityKind

from .providers import SubAgentProvider, ToolProvider

RegistryProvider = ToolProvider | SubAgentProvider

_KIND_ALIASES = {
    "specialist": CapabilityKind.sub_agent,
    "subagent": CapabilityKind.sub_agent,
    "sub-agent": CapabilityKind.sub_agent,
    "tool": CapabilityKind.tool,
    "mcp": CapabilityKind.mcp_tool,
}


class CapabilityRegistryError(Exception):
    """Base error for capability registry operations."""


class CapabilityAlreadyRegistered(CapabilityRegistryError):
    """Raised when a capability name is registered more than once."""


class CapabilityNotFound(CapabilityRegistryError):
    """Raised when a capability cannot be found in the registry."""


class CapabilityRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, RegistryProvider] = {}
        self._logger = logging.getLogger(__name__)

    def register(self, provider: RegistryProvider) -> CapabilityDescriptor:
        name = provider.name
        if name in self._providers:
            self._logger.warning("Duplicate capability registration rejected: %s", name)
            raise CapabilityAlreadyRegistered(name)

        self._providers[name] = provider
        self._logger.info("Registered capability %s (%s)", name, provider.descriptor.kind.value)
        return provider.descriptor.model_copy(deep=True)

    def get(self, name: str) -> RegistryProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise CapabilityNotFound(name) from exc

    def list(self, kind: CapabilityKind | str | None = None) -> list[CapabilityDescriptor]:
        normalized_kind = self._normalize_kind(kind)
        descriptors = [
            provider.descriptor
            for provider in self._providers.values()
            if normalized_kind is None or provider.descriptor.kind == normalized_kind
        ]
        return [descriptor.model_copy(deep=True) for descriptor in descriptors]

    @staticmethod
    def _normalize_kind(kind: CapabilityKind | str | None) -> CapabilityKind | None:
        if kind is None:
            return None
        if isinstance(kind, CapabilityKind):
            return kind

        normalized = kind.strip().lower()
        if normalized in _KIND_ALIASES:
            return _KIND_ALIASES[normalized]

        try:
            return CapabilityKind(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported capability kind filter: {kind}") from exc

