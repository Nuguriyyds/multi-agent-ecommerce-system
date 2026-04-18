from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from app.v3.hooks import HookBus
from app.v3.models import MemoryEntry, MemoryLayer, MemoryWriteDecision

from .write_decision import emit_memory_write_hook, evaluate_memory_write


class DurableMemory:
    """User-scoped durable memory gated by MemoryWriteDecision."""

    def __init__(self, hook_bus: HookBus | None = None) -> None:
        self._store: dict[str, MemoryEntry] = {}
        self._hook_bus = hook_bus
        self._logger = logging.getLogger(__name__)

    async def write(
        self,
        user_id: str,
        entry: MemoryEntry,
        *,
        session_id: str | None = None,
        trace_id: str | None = None,
        turn_number: int | None = None,
    ) -> MemoryWriteDecision:
        stored_entry = entry.model_copy(update={"layer": MemoryLayer.durable_user}, deep=True)
        decision = evaluate_memory_write(stored_entry, target_layer=MemoryLayer.durable_user)

        if decision.decision == "allow":
            self._store[self._storage_key(user_id, stored_entry.key)] = stored_entry
            self._logger.info(
                "Durable memory write allowed: user=%s key=%s source=%s",
                user_id,
                stored_entry.key,
                stored_entry.source.value,
            )
        else:
            self._logger.warning(
                "Durable memory write denied: user=%s key=%s source=%s reason=%s",
                user_id,
                stored_entry.key,
                stored_entry.source.value,
                decision.reason,
            )

        await emit_memory_write_hook(
            self._hook_bus,
            entry=stored_entry,
            decision=decision,
            session_id=session_id,
            user_id=user_id,
            trace_id=trace_id,
            turn_number=turn_number,
        )
        return decision

    def get_view(self, user_id: str, keys: list[str] | None = None) -> Mapping[str, Any]:
        if keys is None:
            entries = self._entries_for_user(user_id)
            snapshot = {entry.key: copy.deepcopy(entry.value) for entry in entries.values()}
        else:
            snapshot = {
                key: copy.deepcopy(entry.value)
                for key in keys
                if (entry := self._store.get(self._storage_key(user_id, key))) is not None
            }
        return MappingProxyType(snapshot)

    def get_entry(self, user_id: str, key: str) -> MemoryEntry | None:
        entry = self._store.get(self._storage_key(user_id, key))
        if entry is None:
            return None
        return entry.model_copy(deep=True)

    def _entries_for_user(self, user_id: str) -> dict[str, MemoryEntry]:
        prefix = f"{user_id}:"
        return {
            entry.key: entry.model_copy(deep=True)
            for storage_key, entry in self._store.items()
            if storage_key.startswith(prefix)
        }

    @staticmethod
    def _storage_key(user_id: str, key: str) -> str:
        return f"{user_id}:{key}"
