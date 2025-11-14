"""Tests for streaming reasoning integration."""

import json
import logging
import pytest

from src.conversion.response_converter import convert_openai_streaming_to_claude
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage


@pytest.mark.asyncio
async def test_streaming_emits_thinking_blocks():
    async def fake_stream():
        first = {
            "choices": [
                {
                    "delta": {"reasoning_content": "先分析"},
                    "finish_reason": None,
                }
            ]
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}"
        second = {
            "choices": [
                {
                    "delta": {"content": "给出答案"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
        yield f"data: {json.dumps(second, ensure_ascii=False)}"
        yield "data: [DONE]"

    request = ClaudeMessagesRequest(
        model="claude-haiku-test",
        max_tokens=32,
        messages=[ClaudeMessage(role="user", content="你好")],
        stream=True,
    )

    events = []
    logger = logging.getLogger("test_reasoning_stream")
    async for event in convert_openai_streaming_to_claude(fake_stream(), request, logger):
        events.append(event)

    assert any(
        '"type": "thinking"' in event or 'thinking_delta' in event for event in events
    ), "expected thinking block events in SSE stream"
    assert any('"usage"' in event for event in events if 'message_delta' in event)
    assert any(
        '"stop_reason": "end_turn"' in event or 'message_stop' in event for event in events
    )
