# Repository Guidelines

## 项目结构与模块组织
- `src/` 核心代码：
  - `api/`（FastAPI 路由）、`conversion/`（Claude→OpenAI 转换）、`core/`（配置、客户端、日志）、`models/`（数据模型）、`main.py`（入口）。
- `tests/` 测试（如 `tests/test_*.py`、`test_cancellation.py`）。
- `.env`、`.env.example` 运行时配置；请勿提交密钥。
- 文档与资源：`README.md`、`QUICKSTART.md`、`BINARY_PACKAGING.md`、`demo.png`。

## 构建、测试与开发命令
- 安装依赖：`uv sync` 或 `pip install -r requirements.txt`。
- 本地运行：`uv run claude-code-proxy` 或 `python start_proxy.py`。
- Docker：`docker compose up -d`。
- 测试：`pytest -q` 或 `pytest tests`。
- 格式化/整理：`black .`、`isort .`。
- 类型检查：`mypy src`。

## 代码风格与命名约定
- Python ≥3.9；4 空格缩进；最大行长 100（Black）。
- 强制类型注解；避免无类型定义（mypy 配置启用）。
- 命名：模块/函数用 snake_case，类用 CamelCase，环境变量用 UPPER_SNAKE。
- 导入保持有序（isort，Black profile）；使用显式包路径（如 `src.core.config`）。

## 测试准则
- 框架：`pytest`、`pytest-asyncio`；HTTP 测试使用 `httpx`。
- 文件命名：`tests/test_*.py`；测试函数以 `test_*` 命名。
- 重点覆盖 `src/api/endpoints.py`、`src/conversion/*`、`src/core/*` 的配置与映射。
- 测试应快速、隔离；避免真实网络调用，尽量 mock OpenAI 客户端。

## 提交与合并请求规范
- 采用 Conventional Commits：`feat:`、`fix:`、`docs:`、`chore:`，可带作用域（如 `docs(binaries)`、`fix[config]`）。
- 提交信息用祈使句，简洁说明影响，并关联 Issue。
- PR 内容：变更说明、复现与验证步骤、受影响的环境变量（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`CUSTOM_HEADER_*`）、必要截图/日志。

## 安全与配置提示
- 通过 `.env` 配置；严禁提交任何密钥。
- 必需：`OPENAI_API_KEY`。可选：`ANTHROPIC_API_KEY` 用于客户端校验。
- 自定义请求头：设置 `CUSTOM_HEADER_*`（自动转换为 HTTP 头；详见 README）。
- 建议本地使用 `LOG_LEVEL=info`，避免在日志中输出敏感信息。
