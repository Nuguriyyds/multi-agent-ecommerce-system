from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from app.v3.hooks import HookBus
from app.v3.models import MemoryEntry, MemoryLayer, MemoryWriteDecision

from .write_decision import emit_memory_write_hook, evaluate_memory_write


class SessionMemory:
    """Session-scoped working memory with immutable read snapshots."""

    def __init__(self, hook_bus: HookBus | None = None) -> None:
        self._store: dict[str, dict[str, MemoryEntry]] = {}
        self._hook_bus = hook_bus
        self._logger = logging.getLogger(__name__)

    async def write(
        self,
        session_id: str,
        entry: MemoryEntry,
        *,
        trace_id: str | None = None,
        turn_number: int | None = None,
    ) -> MemoryWriteDecision:
        stored_entry = entry.model_copy(update={"layer": MemoryLayer.session_working}, deep=True)
        decision = evaluate_memory_write(stored_entry, target_layer=MemoryLayer.session_working)
        self._store.setdefault(session_id, {})[stored_entry.key] = stored_entry

        self._logger.info(
            "Session memory write allowed: session=%s key=%s source=%s",
            session_id,
            stored_entry.key,
            stored_entry.source.value,
        )
        await emit_memory_write_hook(
            self._hook_bus,
            entry=stored_entry,
            decision=decision,
            session_id=session_id,
            trace_id=trace_id,
            turn_number=turn_number,
        )
        return decision

    def get_view(self, session_id: str, keys: list[str] | None = None) -> Mapping[str, Any]:
        session_entries = self._store.get(session_id, {})
        snapshot = self._build_snapshot(session_entries, keys=keys)
        return MappingProxyType(snapshot)

    def get_entry(self, session_id: str, key: str) -> MemoryEntry | None:
        entry = self._store.get(session_id, {}).get(key)
        if entry is None:
            return None
        return entry.model_copy(deep=True)

    @staticmethod
    def _build_snapshot(entries: dict[str, MemoryEntry], *, keys: list[str] | None = None) -> dict[str, Any]:
        if keys is None:
            iterator = entries.items()
        else:
            iterator = ((key, entries[key]) for key in keys if key in entries)

        return {key: copy.deepcopy(entry.value) for key, entry in iterator}
