from __future__ import annotations

from typing import Any

import pytest

from app.v3.hooks import HookBus
from app.v3.memory import DurableMemory, SessionMemory
from app.v3.models import HookEvent, HookPoint, HookResult, MemoryEntry, MemoryLayer, MemorySource


def make_entry(key: str, value: Any, source: MemorySource) -> MemoryEntry:
    return MemoryEntry(key=key, value=value, source=source)


@pytest.mark.asyncio
async def test_durable_memory_allows_user_confirmed_budget_and_emits_hook() -> None:
    bus = HookBus()
    observed_events: list[HookEvent] = []

    async def recorder(event: HookEvent) -> HookResult:
        observed_events.append(event.model_copy(deep=True))
        return HookResult(handler_name="recorder")

    bus.register(HookPoint.memory_write, recorder)
    memory = DurableMemory(hook_bus=bus)

    decision = await memory.write(
        "user-1",
        make_entry(
            "budget",
            {"max": 3000, "currency": "CNY"},
            MemorySource.user_confirmed,
        ),
        session_id="session-1",
        trace_id="trace-1",
        turn_number=1,
    )

    assert decision.decision == "allow"
    assert decision.target_layer == MemoryLayer.durable_user
    assert dict(memory.get_view("user-1")) == {"budget": {"max": 3000, "currency": "CNY"}}
    assert len(observed_events) == 1
    assert observed_events[0].hook_point == HookPoint.memory_write
    assert observed_events[0].session_id == "session-1"
    assert observed_events[0].payload["memory_key"] == "budget"
    assert observed_events[0].payload["decision"] == "allow"
    assert observed_events[0].payload["target_layer"] == MemoryLayer.durable_user.value
    assert observed_events[0].payload["user_id"] == "user-1"
    assert observed_events[0].payload["written"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [MemorySource.inferred, MemorySource.tool_fact], ids=lambda item: item.value)
async def test_durable_memory_denies_non_user_confirmed_sources(source: MemorySource) -> None:
    bus = HookBus()
    observed_events: list[HookEvent] = []

    async def recorder(event: HookEvent) -> HookResult:
        observed_events.append(event.model_copy(deep=True))
        return HookResult(handler_name="recorder")

    bus.register(HookPoint.memory_write, recorder)
    memory = DurableMemory(hook_bus=bus)

    decision = await memory.write(
        "user-1",
        make_entry("brand_preference", "Sony", source),
        session_id="session-1",
        trace_id="trace-1",
        turn_number=1,
    )

    assert decision.decision == "deny"
    assert decision.target_layer == MemoryLayer.durable_user
    assert decision.reason
    assert dict(memory.get_view("user-1")) == {}
    assert len(observed_events) == 1
    assert observed_events[0].payload["decision"] == "deny"
    assert observed_events[0].payload["source"] == source.value
    assert observed_events[0].payload["written"] is False


@pytest.mark.asyncio
async def test_session_memory_isolates_sessions_and_emits_hooks() -> None:
    bus = HookBus()
    observed_events: list[HookEvent] = []

    async def recorder(event: HookEvent) -> HookResult:
        observed_events.append(event.model_copy(deep=True))
        return HookResult(handler_name="recorder")

    bus.register(HookPoint.memory_write, recorder)
    memory = SessionMemory(hook_bus=bus)

    first_decision = await memory.write(
        "session-1",
        make_entry("budget", {"max": 3000}, MemorySource.user_confirmed),
        trace_id="trace-1",
        turn_number=1,
    )
    second_decision = await memory.write(
        "session-2",
        make_entry("budget", {"max": 1500}, MemorySource.inferred),
        trace_id="trace-2",
        turn_number=1,
    )

    assert first_decision.decision == "allow"
    assert second_decision.decision == "allow"
    assert dict(memory.get_view("session-1")) == {"budget": {"max": 3000}}
    assert dict(memory.get_view("session-2")) == {"budget": {"max": 1500}}
    assert dict(memory.get_view("missing-session")) == {}
    assert len(observed_events) == 2
    assert [event.payload["target_layer"] for event in observed_events] == [
        MemoryLayer.session_working.value,
        MemoryLayer.session_working.value,
    ]


@pytest.mark.asyncio
async def test_session_memory_view_is_read_only_snapshot() -> None:
    memory = SessionMemory()
    await memory.write(
        "session-1",
        make_entry("budget", {"max": 3000, "currency": "CNY"}, MemorySource.user_confirmed),
    )

    view = memory.get_view("session-1")

    with pytest.raises(TypeError):
        view["brand"] = "Sony"  # type: ignore[index]

    nested_budget = view["budget"]
    nested_budget["max"] = 1200

    assert memory.get_view("session-1")["budget"]["max"] == 3000
