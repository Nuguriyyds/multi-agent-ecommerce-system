from __future__ import annotations

import pytest

from tests.v3.smoke.helpers import create_smoke_client


@pytest.mark.asyncio
async def test_scenario_c_falls_back_for_checkout_request() -> None:
    app, client = await create_smoke_client(
        {
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
            }
        }
    )
    try:
        create_response = await client.post("/api/v3/sessions")
        session_id = create_response.json()["session_id"]

        response = await client.post(
            f"/api/v3/sessions/{session_id}/messages",
            json={"message": "就这个了，帮我下单"},
        )
        trace_response = await client.get(f"/api/v3/sessions/{session_id}/turns/1/trace")
    finally:
        await client.aclose()
        await app.state.v3_main_agent.llm_client.aclose()

    body = response.json()
    trace_body = trace_response.json()

    assert response.status_code == 200
    assert body["status"] == "fallback"
    assert "电商平台" in body["message"]

    assert trace_response.status_code == 200
    assert trace_body["terminal_state"] == "fallback"
    assert trace_body["fallback_reason"] == "runtime:business_scope_violation"
    assert [item["action"]["kind"] for item in trace_body["decisions"]] == ["fallback"]
    assert trace_body["invocations"] == []
