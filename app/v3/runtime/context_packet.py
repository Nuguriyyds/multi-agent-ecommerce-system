from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from app.v3.models import CompressionPolicy, ContextPacket, SessionState, TurnTaskBoard

_MISSING = object()
_WORKING_MEMORY_SECTION_KEYS = {
    "active_constraints",
    "confirmed_preferences",
    "current_candidates",
    "comparison_dimensions",
    "unanswered_clarifications",
    "memory_conflicts",
}


class ContextPacketBuilder:
    def __init__(self, compression_policy: CompressionPolicy | None = None) -> None:
        self._compression_policy = compression_policy.model_copy(deep=True) if compression_policy else CompressionPolicy()

    def compress(
        self,
        session: SessionState,
        task_board: TurnTaskBoard,
        *,
        latest_user_message: str = "",
        compression_policy: CompressionPolicy | None = None,
    ) -> ContextPacket:
        session_working_memory = self._sanitize_value(session.session_working_memory)
        durable_user_memory = self._sanitize_value(session.durable_user_memory)
        policy = compression_policy.model_copy(deep=True) if compression_policy else self._compression_policy.model_copy(deep=True)

        confirmed_preferences = self._as_dict(session_working_memory.get("confirmed_preferences"))
        confirmed_preferences.update(durable_user_memory)

        active_constraints = self._derive_active_constraints(session_working_memory, durable_user_memory)
        active_constraints.update(self._as_dict(session_working_memory.get("active_constraints")))

        return ContextPacket(
            session_id=session.session_id,
            latest_user_message=latest_user_message,
            active_constraints=active_constraints,
            session_working_memory=session_working_memory,
            durable_user_memory=durable_user_memory,
            confirmed_preferences=confirmed_preferences,
            current_candidates=self._as_list(session_working_memory.get("current_candidates")),
            comparison_dimensions=self._as_string_list(session_working_memory.get("comparison_dimensions")),
            unanswered_clarifications=self._as_list(session_working_memory.get("unanswered_clarifications")),
            memory_conflicts=self._as_list(session_working_memory.get("memory_conflicts")),
            recent_observation_ids=self._recent_observation_ids(task_board),
            compression_policy=policy,
        )

    def _derive_active_constraints(
        self,
        session_working_memory: dict[str, Any],
        durable_user_memory: dict[str, Any],
    ) -> dict[str, Any]:
        derived: dict[str, Any] = {}
        for key, value in session_working_memory.items():
            if key in _WORKING_MEMORY_SECTION_KEYS:
                continue
            derived[key] = copy.deepcopy(value)
        return derived

    def _sanitize_value(self, value: Any) -> Any:
        sanitized = self._sanitize_node(value)
        if sanitized is _MISSING:
            return {} if isinstance(value, Mapping) else []
        return sanitized

    def _sanitize_node(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            source = value.get("source")
            if isinstance(source, str) and source.lower() == "inferred":
                return _MISSING

            sanitized_mapping: dict[str, Any] = {}
            for key, child in value.items():
                if key == "inferred":
                    continue

                sanitized_child = self._sanitize_node(child)
                if sanitized_child is _MISSING:
                    continue
                sanitized_mapping[key] = sanitized_child
            return sanitized_mapping

        if isinstance(value, list):
            sanitized_items: list[Any] = []
            for item in value:
                sanitized_item = self._sanitize_node(item)
                if sanitized_item is _MISSING:
                    continue
                sanitized_items.append(sanitized_item)
            return sanitized_items

        return copy.deepcopy(value)

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return copy.deepcopy(value)
        return {}

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return copy.deepcopy(value)
        return []

    @staticmethod
    def _as_string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _recent_observation_ids(task_board: TurnTaskBoard) -> list[str]:
        observation_ids: list[str] = []
        seen_ids: set[str] = set()
        for task in task_board.tasks:
            for invocation in task.invocations:
                if invocation.observation_id is None or invocation.observation_id in seen_ids:
                    continue
                seen_ids.add(invocation.observation_id)
                observation_ids.append(invocation.observation_id)
        return observation_ids


__all__ = ["ContextPacketBuilder"]
