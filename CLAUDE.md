# 智能电商推荐系统 — 技术约束

## 项目概述

按 Anthropic Harness 方法论构建的电商推荐系统。每次会话用 `/code` 启动 Coding Agent，只做一个 feature。

状态文件：`feature_list.json`（特性清单）、`claude-progress.txt`（进度交接）。

## 已建立的模式（F01-F02，勿破坏）

- 所有 Agent 继承 `agents/base.py:BaseAgent`，实现 `execute()` 和 `default_result()`
- Agent 返回值统一用 `models/agent_result.py:AgentResult`（Pydantic 模型）
- 超时 → 立即降级不重试；普通异常 → 指数退避重试
- 配置统一走 `config/settings.py:get_settings()`，环境变量前缀 `ECOM_`

## 技术约束（所有后续 feature 必须遵守）

### 并发
- 并行用 `asyncio.gather()`，不用线程
- 所有 Agent 的 `execute()` 必须是 async

### 数据传递
- Agent 之间通过 Pydantic model 传数据，不用裸 dict
- 在 `models/` 下定义所有数据结构（UserProfile、Product、RecommendResult 等）

### 外部依赖
- 首期所有外部服务（Redis、Milvus、MySQL）用 mock/内存实现
- Mock 实现放在对应 service 文件里，通过配置开关切换（后续可替换为真实服务）
- LLM 调用通过 OpenAI 兼容接口（langchain-openai 的 ChatOpenAI），API key 为空时走 mock

### 测试
- 每个 Agent 必须有对应的 `tests/test_xxx.py`
- 测试不依赖外部服务（LLM、Redis），全部用 mock
- 用 `pytest-asyncio` 测试 async 函数

### 日志
- 用标准库 `logging`（不引入 structlog，保持依赖简单）
- 关键操作必须有日志：Agent 启动、成功、失败、降级

### 不要做的事
- 不要添加 feature_list.json 中未列出的功能
- 不要引入未在 requirements.txt 中声明的依赖（需要新依赖时先加到 requirements.txt）
- 不要修改已通过验收的 feature 的代码（除非新 feature 有明确需要）
