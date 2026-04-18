from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.v3.config import Settings

HAPPY_TOOL_OBSERVATION_ID = "obs-111111111111"


def configure_mock_responses(app, mock_responses: Mapping[str, object]) -> None:
    llm_client = app.state.v3_main_agent.llm_client
    llm_client._mock_responses = {  # noqa: SLF001 - deterministic smoke setup
        key: llm_client._normalize_sequence(value)  # noqa: SLF001
        for key, value in mock_responses.items()
    }
    llm_client._mock_cursors.clear()  # noqa: SLF001
    llm_client.prompt_history.clear()
    llm_client.scenario_history.clear()


async def create_smoke_client(mock_responses: Mapping[str, object]):
    app = create_app(Settings(openai_api_key="", app_debug=False))
    configure_mock_responses(app, mock_responses)
    client = AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )
    return app, client


def parse_captured_logs(app) -> list[dict[str, Any]]:
    return [json.loads(line) for line in app.state.v3_log_capture.lines]
