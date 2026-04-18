from __future__ import annotations

from typing import Any

from app.v3.hooks import HookBus
from app.v3.models import HookEvent, HookPoint, HookResult, MemoryEntry, MemoryLayer, MemoryWriteDecision


def evaluate_memory_write(
    entry: MemoryEntry,
    *,
    target_layer: MemoryLayer = MemoryLayer.durable_user,
) -> MemoryWriteDecision:
    return MemoryWriteDecision.evaluate(entry, target_layer=target_layer)


def build_memory_write_payload(
    entry: MemoryEntry,
    decision: MemoryWriteDecision,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "memory_key": entry.key,
        "decision": decision.decision,
        "target_layer": decision.target_layer.value,
        "reason": decision.reason,
        "source": entry.source.value,
        "status": entry.status.value,
        "written": decision.decision in {"allow", "replace", "revoke"},
    }
    if user_id is not None:
        payload["user_id"] = user_id
    return payload


async def emit_memory_write_hook(
    hook_bus: HookBus | None,
    *,
    entry: MemoryEntry,
    decision: MemoryWriteDecision,
    session_id: str | None = None,
    user_id: str | None = None,
    trace_id: str | None = None,
    turn_number: int | None = None,
) -> list[HookResult]:
    if hook_bus is None:
        return []

    event = HookEvent(
        hook_point=HookPoint.memory_write,
        session_id=session_id,
        trace_id=trace_id,
        turn_number=turn_number,
        payload=build_memory_write_payload(entry, decision, user_id=user_id),
    )
    return await hook_bus.emit(HookPoint.memory_write, event)


__all__ = ["build_memory_write_payload", "emit_memory_write_hook", "evaluate_memory_write"]
