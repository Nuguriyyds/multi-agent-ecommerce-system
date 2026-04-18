from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.v3.models import (
    CapabilityDescriptor,
    CapabilityKind,
    Observation,
    SpecialistBrief,
    SpecialistObservation,
)


class ProviderKindMismatch(ValueError):
    """Raised when a provider is initialized with the wrong capability kind."""


class CapabilityProvider(ABC):
    allowed_kinds: frozenset[CapabilityKind] = frozenset()

    def __init__(self, descriptor: CapabilityDescriptor) -> None:
        self._descriptor = descriptor
        self._validate_kind(descriptor.kind)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    @property
    def name(self) -> str:
        return self._descriptor.name

    def _validate_kind(self, kind: CapabilityKind) -> None:
        if self.allowed_kinds and kind not in self.allowed_kinds:
            expected = ", ".join(sorted(item.value for item in self.allowed_kinds))
            raise ProviderKindMismatch(f"{type(self).__name__} requires kind in {{{expected}}}, got {kind.value}")


class ToolProvider(CapabilityProvider, ABC):
    allowed_kinds = frozenset({CapabilityKind.tool, CapabilityKind.mcp_tool})

    @abstractmethod
    async def invoke(self, args: dict[str, Any]) -> Observation:
        """Invoke a tool capability and return a structured observation."""


class SubAgentProvider(CapabilityProvider, ABC):
    allowed_kinds = frozenset({CapabilityKind.sub_agent})

    @abstractmethod
    async def invoke(self, brief: SpecialistBrief) -> SpecialistObservation:
        """Invoke a sub-agent capability and return a specialist observation."""


class MCPProvider(ToolProvider, ABC):
    allowed_kinds = frozenset({CapabilityKind.mcp_tool})

