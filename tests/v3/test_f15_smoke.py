from __future__ import annotations

import importlib
from uuid import UUID

import pytest

from tests.v3.smoke.helpers import HAPPY_TOOL_OBSERVATION_ID, create_smoke_client, parse_captured_logs

catalog_search_module = importlib.import_module("app.v3.tools.catalog_search")

_HAPPY_START = "帮我看看 3000 左右的降噪耳机"
_HAPPY_FOLLOW_UP = "通勤用，不要 Beats"
_FALLBACK_MESSAGE = "就这个了，帮我下单"


@pytest.mark.asyncio
async def test_smoke_logging_is_json_and_trace_endpoint_stays_readable(monkeypatch) -> None:
    monkeypatch.setattr(
        catalog_search_module,
        "uuid4",
        lambda: UUID("11111111-1111-1111-1111-111111111111"),
    )
    app, client = await create_smoke_client(
        {
            _HAPPY_START: {
                "action": {
                    "kind": "ask_clarification",
                    "question": "好的，3000 左右的降噪耳机。你主要在什么场景用？有没有特别不想要的品牌？",
                    "missing_slots": ["scene", "exclusions"],
                },
                "rationale": "Need scene and exclusions before searching.",
                "next_task_label": "clarify_constraints",
                "continue_loop": False,
            },
            "通勤用，不要 beats": [
                {
                    "action": {
                        "kind": "call_tool",
                        "capability_name": "catalog_search",
                        "arguments": {
                            "query": "3000 左右 通勤 降噪耳机",
                            "filters": {
                                "category": "earphones",
                                "scene": "commute",
                                "price_min": 2500,
                                "price_max": 3500,
                                "exclude_brands": ["Beats"],
                                "limit": 4,
                            },
                        },
                    },
                    "rationale": "The user has enough confirmed constraints to search the catalog.",
                    "next_task_label": "search_candidates",
                    "continue_loop": True,
                },
                {
                    "action": {
                        "kind": "reply_to_user",
                        "message": (
                            "3000 左右通勤降噪耳机里，Sony WH-1000XM5 最均衡，"
                            "Bose QuietComfort Ultra 更偏舒适佩戴。要不要我再继续对比音质差异？"
                        ),
                        "observation_ids": [HAPPY_TOOL_OBSERVATION_ID],
                    },
                    "rationale": "The catalog result is enough to produce the first recommendation reply.",
                    "next_task_label": "reply_to_user",
                    "continue_loop": False,
                },
            ],
            "帮我下单": {
                "action": {
                    "kind": "fallback",
                    "reason": "business_scope_violation",
                    "user_message": (
                        "目前我只能帮你做导购咨询。下单需要你到电商平台直接购买，"
                        "如果你想继续比较商品我可以接着帮你看。"
                    ),
                },
                "rationale": "Checkout is outside the V3.0 shopping-guidance boundary.",
                "next_task_label": "fallback",
                "continue_loop": False,
            },
        }
    )
    try:
        create_response = await client.post("/api/v3/sessions")
        session_id = create_response.json()["session_id"]

        turn_1 = await client.post(
            f"/api/v3/sessions/{session_id}/messages",
            json={"message": _HAPPY_START},
        )
        turn_2 = await client.post(
            f"/api/v3/sessions/{session_id}/messages",
            json={"message": _HAPPY_FOLLOW_UP},
        )
        turn_3 = await client.post(
            f"/api/v3/sessions/{session_id}/messages",
            json={"message": _FALLBACK_MESSAGE},
        )
        trace_reply = await client.get(f"/api/v3/sessions/{session_id}/turns/2/trace")
        trace_fallback = await client.get(f"/api/v3/sessions/{session_id}/turns/3/trace")
        logs = parse_captured_logs(app)
    finally:
        await client.aclose()
        await app.state.v3_main_agent.llm_client.aclose()

    assert turn_1.json()["status"] == "clarification"
    assert turn_2.json()["status"] == "reply"
    assert turn_3.json()["status"] == "fallback"

    assert logs
    required_keys = {"timestamp", "level", "trace_id", "session_id", "turn_number", "event", "payload"}
    assert all(required_keys.issubset(entry.keys()) for entry in logs)

    reply_trace_id = turn_2.json()["trace_id"]
    reply_logs = [entry for entry in logs if entry["trace_id"] == reply_trace_id]
    reply_events = {entry["event"] for entry in reply_logs}
    assert {"turn.started", "invocation.started", "invocation.succeeded", "turn.finished"} <= reply_events

    hook_points = {
        entry["payload"]["hook_point"]
        for entry in reply_logs
        if entry["event"] == "hook.emit"
    }
    assert {"turn_start", "decision", "task", "invocation", "turn_end"} <= hook_points

    reply_trace_body = trace_reply.json()
    fallback_trace_body = trace_fallback.json()

    assert trace_reply.status_code == 200
    assert reply_trace_body["decisions"]
    assert reply_trace_body["invocations"]
    assert reply_trace_body["fallback_reason"] is None
    assert reply_trace_body["observations"][0]["observation_id"] == HAPPY_TOOL_OBSERVATION_ID

    assert trace_fallback.status_code == 200
    assert fallback_trace_body["decisions"]
    assert fallback_trace_body["invocations"] == []
    assert fallback_trace_body["fallback_reason"] == "runtime:business_scope_violation"
