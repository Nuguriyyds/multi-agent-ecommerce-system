from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.v3.models import (
    AgentDecision,
    CallToolAction,
    CapabilityDescriptor,
    CapabilityKind,
    InvocationRecord,
    Observation,
    ReplyToUserAction,
    SessionState,
    TaskStatus,
    TurnTask,
)
from app.v3.registry import CapabilityRegistry, ToolProvider
from app.v3.runtime import ContextPacketBuilder, SerialExecutor, TraceStore, TurnTaskBoard


class MockToolProvider(ToolProvider):
    def __init__(self) -> None:
        super().__init__(
            CapabilityDescriptor(
                name="catalog_search",
                kind=CapabilityKind.tool,
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                output_schema={"type": "object"},
                permission_tag="catalog.read",
            )
        )
        self.calls: list[dict[str, object]] = []

    async def invoke(self, args: dict[str, object]) -> Observation:
        self.calls.append(dict(args))
        index = len(self.calls)
        return Observation(
            observation_id=f"obs-{index}",
            source="catalog_search",
            summary=f"Mock catalog result {index}",
            payload={"query": args["query"], "rank": index},
            evidence_source="tool:catalog_search",
        )


class DecisionSequence:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self._decisions = decisions
        self.calls = 0

    async def __call__(self, _) -> AgentDecision:
        decision = self._decisions[self.calls]
        self.calls += 1
        return decision


def make_board_with_dependencies() -> TurnTaskBoard:
    done_task = TurnTask(
        task_id="task-done",
        name="extract_budget",
        status=TaskStatus.done,
        description="Budget already extracted.",
    )
    ready_task = TurnTask(
        task_id="task-ready",
        name="search_catalog",
        status=TaskStatus.pending,
        description="Search the catalog next.",
    )
    blocked_task = TurnTask(
        task_id="task-blocked",
        name="compare_shortlist",
        status=TaskStatus.blocked,
        depends_on=["task-ready"],
        blocked_reason="waiting_on_dependencies",
        description="Compare after search.",
    )
    return TurnTaskBoard.create([done_task, ready_task, blocked_task])


def make_context_session() -> SessionState:
    return SessionState(
        session_id="session-ctx",
        user_id="user-1",
        session_working_memory={
            "active_constraints": {"category": "headphones", "budget_max": 3000},
            "current_candidates": [
                {"sku": "sku-1", "name": "Sony WH-1000XM5"},
                {"sku": "sku-shadow", "name": "Shadow Candidate", "source": "inferred"},
            ],
            "unanswered_clarifications": [
                {"slot": "wear_style", "question": "你更偏好头戴还是入耳？"}
            ],
            "comparison_dimensions": ["price", "battery"],
            "memory_conflicts": [{"key": "brand_preference", "values": ["Sony", "Bose"]}],
            "scene": "commute",
            "inferred": {"brand_preference": "Sony"},
        },
        durable_user_memory={
            "budget": {"max": 3000, "currency": "CNY"},
            "brand_preference": {"value": "Sony", "source": "inferred"},
        },
    )


def test_turn_task_board_runs_ready_then_unblocks_blocked_tasks() -> None:
    board = make_board_with_dependencies()

    assert board.completed_task_ids == ["task-done"]
    assert board.ready_task_ids == ["task-ready"]
    assert board.blocked_task_ids == ["task-blocked"]

    first = board.next_ready()
    assert first is not None
    assert first.task_id == "task-ready"
    assert first.status == TaskStatus.running

    board.mark_done(first.task_id)
    assert board.completed_task_ids == ["task-done", "task-ready"]
    assert board.ready_task_ids == ["task-blocked"]

    second = board.next_ready()
    assert second is not None
    assert second.task_id == "task-blocked"
    assert second.status == TaskStatus.running

    board.mark_done(second.task_id)
    assert board.completed_task_ids == ["task-done", "task-ready", "task-blocked"]
    assert board.ready_task_ids == []
    assert board.blocked_task_ids == []


def test_context_packet_builder_strips_inferred_fields_and_matches_golden_fixture() -> None:
    session = make_context_session()
    board = TurnTaskBoard.create(
        [
            TurnTask(
                task_id="task-1",
                name="search_catalog",
                status=TaskStatus.done,
                invocations=[
                    InvocationRecord(
                        invocation_id="inv-1",
                        task_id="task-1",
                        capability_name="catalog_search",
                        capability_kind=CapabilityKind.tool,
                        status="succeeded",
                        arguments={"query": "anc"},
                        observation_id="obs-1",
                    )
                ],
            )
        ]
    )
    builder = ContextPacketBuilder()

    packet = builder.compress(
        session,
        board,
        latest_user_message="我想看 3000 元内的降噪耳机。",
    )

    payload = packet.model_dump(mode="json")
    golden_path = Path("tests/v3/fixtures/context_packet_golden.json")
    expected = json.loads(golden_path.read_text(encoding="utf-8"))

    assert payload == expected
    assert "inferred" not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_serial_executor_records_trace_and_stores_turn_trace() -> None:
    registry = CapabilityRegistry()
    tool_provider = MockToolProvider()
    registry.register(tool_provider)
    trace_store = TraceStore()
    decision_sequence = DecisionSequence(
        [
            AgentDecision(
                action=CallToolAction(
                    capability_name="catalog_search",
                    arguments={"query": "anc headphones"},
                ),
                rationale="Need candidate facts before any recommendation.",
                next_task_label="search_catalog",
                continue_loop=True,
            ),
            AgentDecision(
                action=CallToolAction(
                    capability_name="catalog_search",
                    arguments={"query": "battery life"},
                ),
                rationale="Need one more tool-backed fact.",
                next_task_label="refresh_candidates",
                continue_loop=True,
            ),
            AgentDecision(
                action=ReplyToUserAction(
                    message="Sony WH-1000XM5 is a strong fit for this budget.",
                    observation_ids=["obs-1", "obs-2"],
                ),
                rationale="The two catalog observations are sufficient for a constrained reply.",
                next_task_label="reply_to_user",
                continue_loop=False,
            ),
        ]
    )
    executor = SerialExecutor(
        decision_provider=decision_sequence,
        registry=registry,
        trace_store=trace_store,
        initial_node="candidate_search",
    )
    session = SessionState(
        session_id="session-exec",
        user_id="user-1",
        session_working_memory={"active_constraints": {"category": "headphones"}},
        durable_user_memory={"budget": {"max": 3000}},
    )

    result = await executor.run_turn(session, "我想要一款 3000 元左右的降噪耳机。")

    assert result.status == "reply"
    assert result.message == "Sony WH-1000XM5 is a strong fit for this budget."
    assert result.completed_steps == 3
    assert tool_provider.calls == [{"query": "anc headphones"}, {"query": "battery life"}]

    trace = trace_store.get("session-exec", 1)
    assert trace is not None
    assert trace.trace_id == result.trace_id
    assert trace.terminal_state == "reply"
    assert len(trace.decisions) == 3
    assert [record.task_name for record in trace.task_records] == [
        "search_catalog",
        "refresh_candidates",
        "reply_to_user",
    ]
    assert len(trace.invocations) == 2
    assert [invocation.observation_id for invocation in trace.invocations] == ["obs-1", "obs-2"]
    assert [observation.observation_id for observation in trace.observations] == ["obs-1", "obs-2"]
