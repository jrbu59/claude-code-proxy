# Azure GPT-5 Reasoning Integration Notes

## 背景
- 目标：让 Claude Proxy 在 Azure GPT-5 reasoning 模式下支持工具调用（如 TodoWrite、Task）。
- 当前实现：对 Claude 请求做转换，调用 Azure Responses API，识别 `function_call` / `tool_call`，执行本地工具并回传结果，再将最终消息转换回 Claude SSE。

## 已完成工作
1. **工具执行骨架**
   - 在 `src/core/tool_executor.py` 添加了轻量执行器，识别 `TodoWrite`、`Task` 等并返回 JSON 占位结果，保留 per-request 状态。

2. **Responses API Loop**
   - `src/core/client.py` 内 `_execute_responses_flow` 支持循环调用，识别 `function_call`，执行工具，准备 `tool_outputs`。
   - 增加 `_submit_responses_tool_outputs`，尝试两个端点：`/responses/{id}/submit_tool_outputs` 和 `/responses/{id}/tool_outputs`，捕获 404。
   - 流式输出 `_responses_stream_from_chat` 模拟 OpenAI SSE：先推 tool delta，再推文本，最后 finish reason。

3. **调试辅助**
   - 记录 DEBUG 日志到 `logs/claude-proxy.log`，并根据响应 ID dump 原始 payload 至 `logs/responses/<request_id>_iter#.json`。

## 目前限制
- Azure Responses API 在 2025-11 测试时未开放提交工具输出的端点，两条候选路径都返回 404。
- 因此 reasoning + tool_call 流程在 Azure 上会卡在提交阶段，最终抛出 500。
- 普通 Chat Completions 流程不受影响。

## 后续建议
1. 关注 Azure 官方文档，确认何时提供 tool outputs 提交支持。
2. 与 Azure 支持团队或管理员确认是否有私有 API/配置。
3. 其他模型（非 Azure）如开放真实 tool loop，可复用当前骨架在 `_submit_responses_tool_outputs` 中添加对应端点。
4. 若短期内需要 reasoning，可考虑禁用工具调用或改为 OpenAI 公共端点测试。

## 相关文件
- `src/core/client.py`
- `src/core/tool_executor.py`
- `src/core/logging.py`
- `src/core/config.py`
- 调试产物：`logs/claude-proxy.log`、`logs/responses/*.json`
- 环境变量：
  - `REASONING_CHAT_PREFIXES`：聊天模式 thinking 的模型前缀（如 `glm`）。
  - `REASONING_CHAT_MAX_OUTPUT_TOKENS`：聊天 reasoning 输出上限（默认 16384）。
  - `PROVIDER_MAX_OUTPUT_TOKENS`：针对 DashScope 等 provider 的通用输出上限（选填）。
