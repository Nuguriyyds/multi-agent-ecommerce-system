from __future__ import annotations

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Agent 执行结果的统一包装。"""

    success: bool = Field(description="是否成功执行（未降级）")
    data: dict = Field(default_factory=dict, description="业务数据")
    degraded: bool = Field(default=False, description="是否走了降级/兜底逻辑")
    agent_name: str = Field(default="", description="Agent 名称")
    attempts: int = Field(default=1, description="实际尝试次数")
    error: str = Field(default="", description="最后一次失败的错误信息（成功时为空）")
    latency_ms: float = Field(default=0.0, description="总耗时（毫秒）")
