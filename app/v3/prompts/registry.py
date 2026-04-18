from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.v3.models import PromptLayer

_ASSEMBLY_ORDER = (
    PromptLayer.platform,
    PromptLayer.scenario,
    PromptLayer.role,
    PromptLayer.task_brief,
)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    """Immutable prompt record stored in the registry."""

    layer: PromptLayer
    name: str
    version: str
    text: str


class PromptRegistryError(Exception):
    """Base error for prompt registry operations."""


class PromptAlreadyRegistered(PromptRegistryError):
    """Raised when a prompt layer/name/version triple is registered twice."""


class PromptNotFound(PromptRegistryError):
    """Raised when a prompt cannot be found in the registry."""


class PromptSelectionAmbiguous(PromptRegistryError):
    """Raised when assembly cannot infer a single prompt for a layer."""


class PromptRegistry:
    """Versioned prompt registry with deterministic layered assembly."""

    def __init__(self) -> None:
        self._prompts: dict[PromptLayer, dict[str, dict[str, PromptDefinition]]] = {
            layer: {} for layer in PromptLayer
        }
        self._latest_versions: dict[tuple[PromptLayer, str], str] = {}
        self._logger = logging.getLogger(__name__)

    def register(
        self,
        layer: PromptLayer | str,
        name: str,
        version: str | int,
        text: str,
    ) -> PromptDefinition:
        normalized_layer = self._normalize_layer(layer)
        normalized_name = self._normalize_name(name)
        normalized_version = self._normalize_version(version)
        normalized_text = self._normalize_text(text)

        versions = self._prompts[normalized_layer].setdefault(normalized_name, {})
        if normalized_version in versions:
            self._logger.warning(
                "Duplicate prompt registration rejected: %s/%s@%s",
                normalized_layer.value,
                normalized_name,
                normalized_version,
            )
            raise PromptAlreadyRegistered(
                f"{normalized_layer.value}:{normalized_name}:{normalized_version}"
            )

        definition = PromptDefinition(
            layer=normalized_layer,
            name=normalized_name,
            version=normalized_version,
            text=normalized_text,
        )
        versions[normalized_version] = definition
        self._latest_versions[(normalized_layer, normalized_name)] = normalized_version

        self._logger.info(
            "Registered prompt %s/%s@%s",
            normalized_layer.value,
            normalized_name,
            normalized_version,
        )
        return definition

    def get(
        self,
        layer: PromptLayer | str,
        name: str,
        *,
        version: str | int | None = None,
    ) -> PromptDefinition:
        normalized_layer = self._normalize_layer(layer)
        normalized_name = self._normalize_name(name)
        versions = self._prompts[normalized_layer].get(normalized_name)

        if versions is None:
            raise PromptNotFound(f"{normalized_layer.value}:{normalized_name}")

        resolved_version = self._resolve_version(
            normalized_layer,
            normalized_name,
            version=version,
        )
        try:
            return versions[resolved_version]
        except KeyError as exc:
            raise PromptNotFound(
                f"{normalized_layer.value}:{normalized_name}:{resolved_version}"
            ) from exc

    def assemble(self, context: Mapping[str | PromptLayer, Any] | None = None) -> str:
        assembly_context = context or {}
        sections: list[str] = []

        for layer in _ASSEMBLY_ORDER:
            definition = self._resolve_assembly_prompt(layer, assembly_context)
            if definition is not None:
                sections.append(definition.text)

        current_turn_context = self._lookup_context_value(
            assembly_context,
            "current_turn_context",
        )
        if current_turn_context is not _MISSING:
            context_text = str(current_turn_context)
            if context_text.strip():
                sections.append(context_text)

        return "\n\n".join(sections)

    def _resolve_assembly_prompt(
        self,
        layer: PromptLayer,
        context: Mapping[str | PromptLayer, Any],
    ) -> PromptDefinition | None:
        selection = self._lookup_context_value(context, layer)
        if selection is _MISSING:
            return self._default_prompt_for_layer(layer)
        if selection is None:
            return None

        if isinstance(selection, str):
            return self.get(layer, selection)

        if isinstance(selection, tuple) and len(selection) == 2:
            name, version = selection
            return self.get(layer, str(name), version=version)

        if isinstance(selection, Mapping):
            if "name" not in selection:
                raise ValueError(f"Prompt selection for {layer.value} must include 'name'")
            return self.get(
                layer,
                str(selection["name"]),
                version=selection.get("version"),
            )

        raise TypeError(
            f"Unsupported prompt selection type for {layer.value}: {type(selection).__name__}"
        )

    def _default_prompt_for_layer(self, layer: PromptLayer) -> PromptDefinition | None:
        names = self._prompts[layer]
        if not names:
            return None
        if len(names) > 1:
            registered_names = ", ".join(sorted(names))
            raise PromptSelectionAmbiguous(
                f"Layer {layer.value!r} has multiple prompt names registered: {registered_names}"
            )

        prompt_name = next(iter(names))
        return self.get(layer, prompt_name)

    def _resolve_version(
        self,
        layer: PromptLayer,
        name: str,
        *,
        version: str | int | None,
    ) -> str:
        if version is None:
            latest_version = self._latest_versions.get((layer, name))
            if latest_version is None:
                raise PromptNotFound(f"{layer.value}:{name}")
            return latest_version
        return self._normalize_version(version)

    @staticmethod
    def _lookup_context_value(
        context: Mapping[str | PromptLayer, Any],
        key: str | PromptLayer,
    ) -> Any:
        if key in context:
            return context[key]
        if isinstance(key, PromptLayer) and key.value in context:
            return context[key.value]
        return _MISSING

    @staticmethod
    def _normalize_layer(layer: PromptLayer | str) -> PromptLayer:
        if isinstance(layer, PromptLayer):
            return layer
        return PromptLayer(layer)

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Prompt name must not be blank")
        return normalized_name

    @staticmethod
    def _normalize_version(version: str | int) -> str:
        normalized_version = str(version).strip()
        if not normalized_version:
            raise ValueError("Prompt version must not be blank")
        return normalized_version

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text.strip():
            raise ValueError("Prompt text must not be blank")
        return text
