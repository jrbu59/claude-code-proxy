from src.conversion.request_converter import convert_claude_to_openai
from src.core.config import config
from src.core.model_manager import model_manager
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage, ClaudeTool


def test_thinking_disabled_when_tools_present(monkeypatch):
    monkeypatch.setattr(config, "reasoning_default_enabled", True)
    monkeypatch.setattr(config, "reasoning_chat_prefixes", ("deepseek",))

    request = ClaudeMessagesRequest(
        model="deepseek-v3",
        max_tokens=1024,
        messages=[ClaudeMessage(role="user", content="hi")],
        stream=True,
        tools=[
            ClaudeTool(
                name="Read",
                description="",
                input_schema={"type": "object", "properties": {}},
            )
        ],
    )

    bundle = convert_claude_to_openai(request, model_manager)
    payload = bundle.payload

    assert (
        "extra_body" not in payload
        or not payload.get("extra_body", {}).get("enable_thinking")
    ), "thinking should be disabled when tools are present"
