"""Unit tests for buffered streaming fallback logic."""

import json
import asyncio

import httpx
from openai._exceptions import BadRequestError

from src.core.config import config
from src.core.client import OpenAIClient


def test_should_buffer_stream_model_defaults():
    # Default configuration should capture GLM models.
    assert config.should_buffer_stream_model("glm-4.5")
    assert not config.should_buffer_stream_model("gpt-4o")


def test_generate_buffered_stream_chunks_with_text_only():
    client = object.__new__(OpenAIClient)

    completion = {
        "choices": [
            {
                "message": {
                    "content": "你好，世界",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    chunks = list(client._generate_buffered_stream_chunks(completion))

    assert chunks[0].startswith("data: ")
    delta_payload = json.loads(chunks[0][len("data: "):])
    assert delta_payload["choices"][0]["delta"]["content"] == "你好，世界"

    final_payload = json.loads(chunks[1][len("data: "):])
    assert final_payload["choices"][0]["finish_reason"] == "stop"
    assert final_payload["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}

    assert chunks[2] == "data: [DONE]"


def test_generate_buffered_stream_chunks_with_tool_calls():
    client = object.__new__(OpenAIClient)

    completion = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_docs",
                                "arguments": '{"query": "test"}'
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }

    chunks = list(client._generate_buffered_stream_chunks(completion))

    serialized = json.loads(chunks[0][len("data: "):])
    tool_delta = serialized["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["index"] == 0
    assert tool_delta["id"] == "call_1"
    assert tool_delta["function"]["name"] == "search_docs"
    assert tool_delta["function"]["arguments"] == '{"query": "test"}'

    finish = json.loads(chunks[1][len("data: "):])
    assert finish["choices"][0]["finish_reason"] == "tool_calls"

    assert chunks[2] == "data: [DONE]"


def test_generate_buffered_stream_chunks_with_reasoning():
    client = object.__new__(OpenAIClient)

    completion = {
        "choices": [
            {
                "message": {
                    "content": "最终回答",
                    "reasoning_content": [
                        {"text": "先思考"},
                        {"text": "再总结"},
                    ],
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }

    chunks = list(client._generate_buffered_stream_chunks(completion))
    first_payload = json.loads(chunks[0][len("data: "):])
    delta = first_payload["choices"][0]["delta"]
    assert delta["content"] == "最终回答"
    assert delta["reasoning_content"] == "先思考再总结"

    finish_payload = json.loads(chunks[1][len("data: "):])
    assert finish_payload["choices"][0]["finish_reason"] == "stop"


def test_buffered_stream_fallback(monkeypatch):
    # Force buffering regardless of model name.
    monkeypatch.setattr(config, "should_buffer_stream_model", lambda model: True)

    client = OpenAIClient(api_key="sk-test", base_url="https://example.com")

    request_payload = {"model": "glm-4.5", "messages": [], "max_tokens": 10}
    bundle = {"payload": request_payload, "api_mode": "chat_completions"}

    class FakeChunk:
        def __init__(self, payload):
            self._payload = payload

        def model_dump(self):
            return self._payload

    async def fake_create(**kwargs):
        if not kwargs.get("stream"):
            response = httpx.Response(
                400,
                request=httpx.Request("POST", "https://example.com/chat/completions"),
            )
            body = {
                "error": {
                    "message": "This model only support stream mode, please enable the stream parameter to access the model. "
                }
            }
            raise BadRequestError(
                "This model only support stream mode",
                response=response,
                body=body,
            )

        async def generator():
            yield FakeChunk(
                {
                    "id": "chatcmpl-test",
                    "choices": [{"delta": {"content": "buffered "}, "finish_reason": None}],
                }
            )
            yield FakeChunk(
                {"choices": [{"delta": {"content": "response"}, "finish_reason": None}]}
            )
            yield FakeChunk(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )

        return generator()

    monkeypatch.setattr(
        client.client.chat.completions, "create", fake_create, raising=False
    )

    result = asyncio.run(client.create_chat_completion(bundle))

    assert result["choices"][0]["message"]["content"] == "buffered response"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 1, "completion_tokens": 1}
    assert result["id"] == "chatcmpl-test"
