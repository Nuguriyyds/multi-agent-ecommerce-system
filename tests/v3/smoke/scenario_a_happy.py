from __future__ import annotations

import importlib
from uuid import UUID

import pytest

from tests.v3.smoke.helpers import HAPPY_TOOL_OBSERVATION_ID, create_smoke_client

catalog_search_module = importlib.import_module("app.v3.tools.catalog_search")

_TURN_1_MESSAGE = "帮我看看 3000 左右的降噪耳机"
_TURN_2_MESSAGE = "通勤用，不要 Beats"


@pytest.mark.asyncio
async def test_scenario_a_happy_path_reaches_recommendation_in_two_turns(monkeypatch) -> None:
    monkeypatch.setattr(
        catalog_search_module,
        "uuid4",
        lambda: UUID("11111111-1111-1111-1111-111111111111"),
    )
    app, client = await create_smoke_client(
        {
            _TURN_1_MESSAGE: {
                "action": {
                    "kind": "ask_clarification",
                    "question": "好的，3000 左右的降噪耳机。你主要在什么场景用？有没有特别不想要的品牌？",
                    "missing_slots": ["scene", "exclusions"],
                },
                "rationale": "Need scene and brand exclusions before the search can continue safely.",
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
                    "rationale": "One catalog-backed observation is enough for the first recommendation reply.",
                    "next_task_label": "reply_to_user",
                    "continue_loop": False,
                },
            ],
        }
    )
    try:
        create_response = await client.post("/api/v3/sessions")
        session_id = create_response.json()["session_id"]

        turn_1 = await client.post(
            f"/api/v3/sessions/{session_id}/messages",
            json={"message": _TURN_1_MESSAGE},
        )
        turn_2 = await client.post(
            f"/api/v3/sessions/{session_id}/messages",
            json={"message": _TURN_2_MESSAGE},
        )
        trace_response = await client.get(f"/api/v3/sessions/{session_id}/turns/2/trace")
    finally:
        await client.aclose()
        await app.state.v3_main_agent.llm_client.aclose()

    turn_1_body = turn_1.json()
    turn_2_body = turn_2.json()
    trace_body = trace_response.json()

    assert turn_1.status_code == 200
    assert turn_1_body["status"] == "clarification"
    assert turn_1_body["completed_steps"] == 1

    assert turn_2.status_code == 200
    assert turn_2_body["status"] == "reply"
    assert turn_2_body["completed_steps"] == 2
    assert "Sony WH-1000XM5" in turn_2_body["message"]
    assert "Bose QuietComfort Ultra" in turn_2_body["message"]

    assert trace_response.status_code == 200
    assert trace_body["terminal_state"] == "reply"
    assert [item["action"]["kind"] for item in trace_body["decisions"]] == [
        "call_tool",
        "reply_to_user",
    ]
    assert trace_body["invocations"][0]["capability_name"] == "catalog_search"
    assert trace_body["observations"][0]["observation_id"] == HAPPY_TOOL_OBSERVATION_ID
