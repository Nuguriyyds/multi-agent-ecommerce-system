from __future__ import annotations

import pytest

from tests.v3.smoke.helpers import create_smoke_client


@pytest.mark.asyncio
async def test_scenario_b_stays_in_multi_turn_clarification_until_direction_is_confirmed() -> None:
    app, client = await create_smoke_client(
        {
            "帮我选个礼物": {
                "action": {
                    "kind": "ask_clarification",
                    "question": "好的，帮你选礼物。先了解一下：送给谁？大概什么预算？",
                    "missing_slots": ["recipient", "budget"],
                },
                "rationale": "No searchable constraints are available yet.",
                "next_task_label": "clarify_gift_context",
                "continue_loop": False,
            },
            "送女朋友的，生日礼物": {
                "action": {
                    "kind": "ask_clarification",
                    "question": "生日礼物，明白了。预算大概多少？有没有她比较喜欢的品类方向？",
                    "missing_slots": ["budget", "category"],
                },
                "rationale": "The recipient and occasion are clear, but budget and category are still missing.",
                "next_task_label": "clarify_budget_and_category",
                "continue_loop": False,
            },
            "1000-2000 吧，她喜欢听歌": {
                "action": {
                    "kind": "ask_clarification",
                    "question": "1000-2000，她爱听歌。你是想送耳机之类的数码产品，还是其他方向？",
                    "missing_slots": ["category_confirmation"],
                },
                "rationale": "Earphones are only an inferred direction and still need explicit confirmation.",
                "next_task_label": "confirm_category_direction",
                "continue_loop": False,
            },
        }
    )
    try:
        create_response = await client.post("/api/v3/sessions")
        session_id = create_response.json()["session_id"]

        turn_1 = await client.post(
            f"/api/v3/sessions/{session_id}/messages",
            json={"message": "帮我选个礼物"},
        )
        turn_2 = await client.post(
            f"/api/v3/sessions/{session_id}/messages",
            json={"message": "送女朋友的，生日礼物"},
        )
        turn_3 = await client.post(
            f"/api/v3/sessions/{session_id}/messages",
            json={"message": "1000-2000 吧，她喜欢听歌"},
        )
        trace_response = await client.get(f"/api/v3/sessions/{session_id}/turns/3/trace")
    finally:
        await client.aclose()
        await app.state.v3_main_agent.llm_client.aclose()

    assert turn_1.status_code == 200
    assert turn_2.status_code == 200
    assert turn_3.status_code == 200

    for response in (turn_1, turn_2, turn_3):
        body = response.json()
        assert body["status"] == "clarification"
        assert body["completed_steps"] == 1

    trace_body = trace_response.json()
    assert trace_response.status_code == 200
    assert trace_body["terminal_state"] == "clarification"
    assert [item["action"]["kind"] for item in trace_body["decisions"]] == ["ask_clarification"]
    assert trace_body["invocations"] == []
