"""BaseAgent 测试 — 覆盖正常执行、超时降级、重试后成功、连续失败返回默认结果。"""

from __future__ import annotations

import asyncio

import pytest

from agents.base import BaseAgent
from models.agent_result import AgentResult


# ── 测试用具体 Agent ──────────────────────────────────────────────────


class SuccessAgent(BaseAgent):
    """始终成功返回结果。"""

    async def execute(self, input_data: dict) -> dict:
        return {"result": "ok", "input": input_data}

    def default_result(self, input_data: dict) -> dict:
        return {"result": "default"}


class TimeoutAgent(BaseAgent):
    """execute 永远超时。"""

    async def execute(self, input_data: dict) -> dict:
        await asyncio.sleep(999)
        return {"result": "should_not_reach"}

    def default_result(self, input_data: dict) -> dict:
        return {"result": "timeout_fallback"}


class FailThenSucceedAgent(BaseAgent):
    """前 N 次抛异常，之后成功。"""

    def __init__(self, fail_times: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.fail_times = fail_times
        self.call_count = 0

    async def execute(self, input_data: dict) -> dict:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise RuntimeError(f"Simulated failure #{self.call_count}")
        return {"result": "recovered", "attempt": self.call_count}

    def default_result(self, input_data: dict) -> dict:
        return {"result": "fail_fallback"}


class AlwaysFailAgent(BaseAgent):
    """永远失败。"""

    async def execute(self, input_data: dict) -> dict:
        raise RuntimeError("Permanent failure")

    def default_result(self, input_data: dict) -> dict:
        return {"result": "permanent_fallback"}


# ── 测试用例 ──────────────────────────────────────────────────────────


class TestNormalExecution:
    """正常执行：execute 成功，返回正确数据。"""

    @pytest.mark.asyncio
    async def test_success_returns_data(self):
        agent = SuccessAgent(name="test-success", timeout=5.0)
        result = await agent.run({"user_id": "u1"})

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.degraded is False
        assert result.data == {"result": "ok", "input": {"user_id": "u1"}}
        assert result.agent_name == "test-success"
        assert result.attempts == 1
        assert result.error == ""
        assert result.latency_ms > 0


class TestTimeoutDegradation:
    """超时触发降级：execute 超时后直接返回 default_result，不再重试。"""

    @pytest.mark.asyncio
    async def test_timeout_returns_default(self):
        agent = TimeoutAgent(name="test-timeout", timeout=0.05, max_retries=3)
        result = await agent.run({})

        assert result.success is False
        assert result.degraded is True
        assert result.data == {"result": "timeout_fallback"}
        assert "Timeout" in result.error
        # 超时不重试，只尝试 1 次
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_timeout_latency_within_bounds(self):
        agent = TimeoutAgent(name="test-timeout-latency", timeout=0.05)
        result = await agent.run({})

        # 耗时应在超时阈值附近，不会远超
        assert result.latency_ms < 500


class TestRetryThenSuccess:
    """失败重试后成功：前几次失败，在重试次数内恢复。"""

    @pytest.mark.asyncio
    async def test_fail_once_then_succeed(self):
        agent = FailThenSucceedAgent(
            fail_times=1,
            name="test-retry",
            timeout=5.0,
            max_retries=2,
            retry_base_delay=0.01,  # 测试中缩短退避
        )
        result = await agent.run({})

        assert result.success is True
        assert result.degraded is False
        assert result.data["result"] == "recovered"
        assert result.attempts == 2
        assert agent.call_count == 2

    @pytest.mark.asyncio
    async def test_fail_twice_then_succeed(self):
        agent = FailThenSucceedAgent(
            fail_times=2,
            name="test-retry-2",
            timeout=5.0,
            max_retries=2,
            retry_base_delay=0.01,
        )
        result = await agent.run({})

        assert result.success is True
        assert result.data["result"] == "recovered"
        assert result.attempts == 3
        assert agent.call_count == 3


class TestConsecutiveFailuresFallback:
    """连续失败返回默认结果：重试全部耗尽后降级。"""

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        agent = AlwaysFailAgent(
            name="test-always-fail",
            timeout=5.0,
            max_retries=2,
            retry_base_delay=0.01,
        )
        result = await agent.run({})

        assert result.success is False
        assert result.degraded is True
        assert result.data == {"result": "permanent_fallback"}
        assert result.attempts == 3  # 1 initial + 2 retries
        assert "RuntimeError" in result.error

    @pytest.mark.asyncio
    async def test_zero_retries_fails_immediately(self):
        agent = AlwaysFailAgent(
            name="test-no-retry",
            timeout=5.0,
            max_retries=0,
            retry_base_delay=0.01,
        )
        result = await agent.run({})

        assert result.success is False
        assert result.degraded is True
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_fail_exceeds_retry_count(self):
        """失败次数刚好超过重试次数 → 降级。"""
        agent = FailThenSucceedAgent(
            fail_times=3,  # 会失败 3 次
            name="test-exceed",
            timeout=5.0,
            max_retries=2,  # 只允许重试 2 次（共 3 次尝试）
            retry_base_delay=0.01,
        )
        result = await agent.run({})

        # 第 3 次尝试仍然失败 → 降级
        assert result.success is False
        assert result.degraded is True
        assert result.data == {"result": "fail_fallback"}


class TestAgentConfiguration:
    """Agent 配置和属性测试。"""

    def test_default_timeout_from_settings(self):
        agent = SuccessAgent(name="test-defaults")
        # 默认超时来自 settings.agent_timeout (10.0)
        assert agent.timeout == 10.0
        assert agent.max_retries == 2
        assert agent.retry_base_delay == 0.5

    def test_custom_timeout(self):
        agent = SuccessAgent(name="test-custom", timeout=3.0, max_retries=5)
        assert agent.timeout == 3.0
        assert agent.max_retries == 5

    @pytest.mark.asyncio
    async def test_agent_name_in_result(self):
        agent = SuccessAgent(name="my-agent")
        result = await agent.run({})
        assert result.agent_name == "my-agent"
