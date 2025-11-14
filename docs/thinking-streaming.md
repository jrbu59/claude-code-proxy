# 思考模式兼容设计说明

## 背景

Kimi（DashScope）、GLM 等上游模型已经支持思考/推理模式，会通过 `reasoning_content` 等字段返回思考过程。为了让 Claude CLI 在使用代理时完整展示这类信息，代理需要在 **请求阶段** 告知上游开启推理，同时在 **响应阶段** 将推理增量转换为 Claude 的 SSE 协议。

## 请求侧配置

### 环境变量

在 `.env.<provider>` 中通过以下环境变量控制思考模式：

- `REASONING_CHAT_PREFIXES`：匹配使用 Chat Completions API 并支持思考模式的模型前缀，例如 `kimi`、`glm`。对于当前不支持 `enable_thinking` 的提供商（如 Azure OpenAI），可通过设为空或不匹配的前缀自动跳过思考参数，避免 400 错误。
- `REASONING_DEFAULT_ENABLED`：设为 `true` 时，对匹配到的模型自动开启思考模式。
- `REASONING_CHAT_MAX_OUTPUT_TOKENS`、`PROVIDER_MAX_OUTPUT_TOKENS`：限制推理场景的最大输出，避免超出上游限制。

以 `.env.kimi` 为例：

```bash
REASONING_CHAT_PREFIXES=kimi
REASONING_DEFAULT_ENABLED=true
REASONING_CHAT_MAX_OUTPUT_TOKENS=16384
PROVIDER_MAX_OUTPUT_TOKENS=32768
```

### 请求转换逻辑

`src/conversion/request_converter.py` 中的 `apply_reasoning_parameters` 会：

1. 判断请求模型名是否匹配 `REASONING_CHAT_PREFIXES` 或 `REASONING_MODEL_PREFIXES`，并确认该提供商支持思考模式。对于 Azure OpenAI 等不支持 `enable_thinking` 的提供商，会自动跳过相关参数。
2. 若开启推理，在 Chat Completions 模式下写入 `extra_body.enable_thinking = True`，并在可用时附带 `reasoning_effort`、`reasoning_verbosity`，同时根据配置裁剪 `max_tokens`。
3. 当请求包含工具定义（Claude 的函数调用）时，为兼容 DeepSeek 等模型在思考模式下不支持 function calling 的限制，代理会自动关闭思考模式并记录日志。
4. 若未来接入 Responses API 推理模型，可通过同一函数扩展。

## 响应侧转换

### 流式处理

`src/conversion/response_converter.py` 对 OpenAI 流式响应做如下处理：

1. 引入 `_merge_reasoning_fields`，统一从 `reasoning_content`、`reasoning`、`thinking` 等字段提取推理文本。
2. 通过 `_ReasoningBlockState` 在 Claude SSE 协议中生成新的 `thinking` 内容块：
   - `content_block_start (type=thinking)`
   - 多个 `content_block_delta (type=thinking_delta)`
   - `content_block_stop`
3. 与正文、工具调用、usage 等事件并行输出，确保 CLI 能逐步显示思考过程。
4. 断线/取消时仍会清理未关闭的 `thinking` 块并输出错误事件；若上游未返回 usage，代理会基于输出字符数估算 token，保证 Claude 不再显示 `nan`。

### 缓冲与非流式回放

`OpenAIClient._generate_buffered_stream_chunks` 同步支持思考文本：

- 当上游以非流式方式返回 `reasoning_content` 时，会将其拼接在第一条 `delta` 中，字段名保持为 `reasoning_content`，下游复用同一抽象即可。

## 扩展指引

为新的上游模型接入思考模式时，建议按以下步骤操作：

1. 在 `.env.<provider>` 中追加前缀、默认开关与最大输出配置。
2. 修改 `apply_reasoning_parameters`，确保能根据模型名正确设置 `extra_body` 或 Responses API 的 `reasoning` 字段。
3. 如返回结构与现有字段不同，在 `_merge_reasoning_fields` 中补充新的字段名。
4. 若模型使用自定义工具或补充字段，可在 `_handle_openai_delta` 内扩展处理逻辑。
5. 新增或更新测试，覆盖流式与缓冲两条路径，例如 `tests/test_reasoning_stream.py`、`tests/test_buffered_stream.py`。

## 调试与验证

1. 将 `LOG_LEVEL` 设为 `DEBUG`，观察 `logs/claude-proxy.log`，确认请求日志包含 `extra_json: {'enable_thinking': True}`。
2. 关注日志中的 `SSE outbound` 行，应出现 `content_block_start`（thinking）、`thinking_delta` 等事件。
3. 使用 `curl` 或 `.venv/bin/python` 模拟请求，确保返回流中既有推理文本也有最终回答。
4. 运行 `.venv/bin/pytest`，保证自检通过。

## 工具策略与配置建议

思考模式本身不依赖 Claude 工具，但部分模型对工具调用的兼容度有限。现阶段的做法是通过环境变量（如 `TOOL_GROUPS_DISABLED_DASHSCOPE`）按提供商禁用敏感工具组。建议：

1. 保留 `tasking`、`playwright_mcp` 等仍不兼容的组；
2. 根据需求选择性开放 `filesystem`、`search`、`productivity` 等与推理无冲突的能力；
3. 如需为单一模型开启更多工具，可在对应 `.env` 中新增覆盖变量（例如 `TOOL_GROUPS_DISABLED_KIMI=...`）。

后续若上游模型的推理字段发生变化或增加新事件类型，只需在 `_merge_reasoning_fields` 与 `_ReasoningBlockState` 中扩展即可，不影响现有流程。
