from __future__ import annotations

from app.v3.models import TraceRecord


class TraceStore:
    def __init__(self) -> None:
        self._store: dict[tuple[str, int], TraceRecord] = {}

    def save(self, trace: TraceRecord) -> None:
        key = (trace.session_id, trace.turn_number)
        self._store[key] = trace.model_copy(deep=True)

    def get(self, session_id: str, turn_number: int) -> TraceRecord | None:
        trace = self._store.get((session_id, turn_number))
        if trace is None:
            return None
        return trace.model_copy(deep=True)


__all__ = ["TraceStore"]
