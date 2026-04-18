from __future__ import annotations

import pytest

from app.v3.models import PromptLayer
from app.v3.prompts import PromptRegistry, PromptSelectionAmbiguous


def test_prompt_registry_assembles_layers_in_fixed_order() -> None:
    registry = PromptRegistry()
    registry.register(PromptLayer.role, "main_agent", "v1", "ROLE")
    registry.register(PromptLayer.task_brief, "budget_pick", "v1", "TASK")
    registry.register(PromptLayer.platform, "core", "v1", "PLATFORM")
    registry.register(PromptLayer.scenario, "shopping", "v1", "SCENARIO")

    assembled = registry.assemble()

    assert assembled == "PLATFORM\n\nSCENARIO\n\nROLE\n\nTASK"


def test_prompt_registry_defaults_to_latest_version_for_same_name() -> None:
    registry = PromptRegistry()
    registry.register(PromptLayer.role, "main_agent", "v1", "ROLE_V1")
    registry.register(PromptLayer.role, "main_agent", "v2", "ROLE_V2")

    prompt = registry.get(PromptLayer.role, "main_agent")

    assert prompt.version == "v2"
    assert prompt.text == "ROLE_V2"


def test_prompt_registry_can_fetch_exact_name_and_version() -> None:
    registry = PromptRegistry()
    registry.register(PromptLayer.role, "main_agent", "v1", "ROLE_V1")
    registry.register(PromptLayer.role, "main_agent", "v2", "ROLE_V2")

    prompt = registry.get(PromptLayer.role, "main_agent", version="v1")

    assert prompt.version == "v1"
    assert prompt.text == "ROLE_V1"


def test_prompt_registry_requires_explicit_selection_when_layer_has_multiple_names() -> None:
    registry = PromptRegistry()
    registry.register(PromptLayer.platform, "core", "v1", "PLATFORM")
    registry.register(PromptLayer.role, "main_agent", "v1", "MAIN_ROLE")
    registry.register(PromptLayer.role, "comparison_specialist", "v1", "COMPARE_ROLE")

    with pytest.raises(PromptSelectionAmbiguous):
        registry.assemble()

    assembled = registry.assemble({"role": ("comparison_specialist", "v1")})

    assert assembled == "PLATFORM\n\nCOMPARE_ROLE"
