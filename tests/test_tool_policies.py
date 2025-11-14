from src.core.config import Config
from src.core.constants import Constants
from src.core.tool_policies import sanitize_tool_payload


def build_request():
    return {
        "model": "glm-4.6",
        "messages": [
            {
                "role": Constants.ROLE_ASSISTANT,
                "content": "Thinking...",
                "tool_calls": [
                    {
                        "id": "call_task",
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": "Task",
                        },
                    },
                    {
                        "id": "call_playwright",
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": "mcp__playwright__browser_click",
                        },
                    },
                ],
            },
            {
                "role": Constants.ROLE_TOOL,
                "tool_call_id": "call_task",
                "content": "task output",
            },
            {
                "role": Constants.ROLE_TOOL,
                "tool_call_id": "call_playwright",
                "content": "click output",
            },
        ],
        "tool_choice": {
            "type": Constants.TOOL_FUNCTION,
            Constants.TOOL_FUNCTION: {"name": "Task"},
        },
        "tools": [
            {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {
                    "name": "Task",
                    "description": "Launch agent",
                    "parameters": {},
                },
            },
            {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {
                    "name": "mcp__playwright__browser_click",
                    "description": "Click element",
                    "parameters": {},
                },
            },
        ],
    }


def test_sanitize_tool_payload_removes_disabled_groups():
    request = build_request()

    sanitize_tool_payload(request, ["tasking"])

    # Task tool stripped
    remaining_tools = request["tools"]
    assert len(remaining_tools) == 1
    assert (
        remaining_tools[0][Constants.TOOL_FUNCTION]["name"]
        == "mcp__playwright__browser_click"
    )

    # tool_choice reset to auto because preferred tool removed
    assert request["tool_choice"] == "auto"

    # assistant message should keep only playwright call
    assistant = request["messages"][0]
    assert len(assistant["tool_calls"]) == 1
    assert (
        assistant["tool_calls"][0][Constants.TOOL_FUNCTION]["name"]
        == "mcp__playwright__browser_click"
    )

    # tool message referencing removed call gone
    tool_ids = [m.get("tool_call_id") for m in request["messages"] if m["role"] == "tool"]
    assert tool_ids == ["call_playwright"]


def test_sanitize_tool_payload_removes_all_tools():
    request = build_request()
    sanitize_tool_payload(
        request,
        ["tasking", "filesystem", "shell", "search", "productivity", "playwright_mcp"],
    )

    assert "tools" not in request
    assert "tool_choice" not in request

    assistant = request["messages"][0]
    assert "tool_calls" not in assistant
    assert all(m["role"] != "tool" for m in request["messages"][1:])


def test_config_provider_specific_tool_groups(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TOOL_GROUPS_DISABLED", "tasking")
    monkeypatch.setenv("TOOL_GROUPS_DISABLED_DASHSCOPE", "playwright_mcp")

    cfg = Config()
    try:
        assert set(cfg.get_disabled_tool_groups("dashscope")) == {
            "tasking",
            "playwright_mcp",
        }
        assert set(cfg.get_disabled_tool_groups("openai")) == {"tasking"}
        assert cfg.resolve_provider_key("https://dashscope.aliyuncs.com/compatible-mode/v1") == "dashscope"
    finally:
        # Prevent side effects on other tests by clearing env overrides
        monkeypatch.delenv("TOOL_GROUPS_DISABLED", raising=False)
        monkeypatch.delenv("TOOL_GROUPS_DISABLED_DASHSCOPE", raising=False)
