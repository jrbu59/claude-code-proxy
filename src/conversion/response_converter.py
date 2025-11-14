import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import HTTPException, Request

from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest


def _build_sse_event(logger, event_type: str, payload: Dict[str, Any]) -> str:
    event = (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )
    logger.debug("SSE outbound: %s", event.strip())
    return event


def _extract_reasoning_texts(raw: Any) -> Iterable[str]:
    """Yield plain-text reasoning fragments from diverse provider payloads."""

    if raw is None:
        return []

    def _inner(value: Any) -> Iterable[str]:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                yield text
        elif isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str) and text.strip():
                yield text
            # Some providers nest reasoning text under other keys
            for key in ("content", "reasoning", "messages"):
                nested = value.get(key)
                if isinstance(nested, (list, tuple)):
                    for item in nested:
                        yield from _inner(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from _inner(item)

    return list(_inner(raw))


def _merge_reasoning_fields(delta: Dict[str, Any]) -> Iterable[str]:
    """Collect reasoning text from known delta field names."""

    allowed_fields = ("reasoning_content", "reasoning", "thought", "thinking")
    for field in allowed_fields:
        if field in delta and delta[field] is not None:
            texts = _extract_reasoning_texts(delta[field])
            if texts:
                for text in texts:
                    yield text


@dataclass
class _ReasoningBlockState:
    logger: Any
    allocate_index: Any
    block_index: Optional[int] = None
    active: bool = False

    def emit_delta(self, text: str) -> List[str]:
        text = text or ""
        if not text.strip():
            return []

        events: List[str] = []
        if not self.active:
            self.block_index = int(self.allocate_index())
            start_payload = {
                "type": Constants.EVENT_CONTENT_BLOCK_START,
                "index": self.block_index,
                "content_block": {
                    "type": Constants.CONTENT_THINKING,
                    "text": "",
                    "thinking": {"type": "text", "text": ""},
                },
            }
            events.append(
                _build_sse_event(
                    self.logger,
                    Constants.EVENT_CONTENT_BLOCK_START,
                    start_payload,
                )
            )
            self.active = True

        delta_payload = {
            "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
            "index": self.block_index,
            "delta": {
                "type": Constants.DELTA_THINKING,
                "text": text,
                "thinking": {"type": "text_delta", "text": text},
            },
        }
        events.append(_build_sse_event(self.logger, Constants.EVENT_CONTENT_BLOCK_DELTA, delta_payload))
        return events

    def close(self) -> List[str]:
        if not self.active or self.block_index is None:
            return []
        payload = {
            "type": Constants.EVENT_CONTENT_BLOCK_STOP,
            "index": self.block_index,
        }
        self.active = False
        block_idx = self.block_index
        self.block_index = None
        return [_build_sse_event(self.logger, Constants.EVENT_CONTENT_BLOCK_STOP, payload)]


class _ContentIndexAllocator:
    def __init__(self, start: int = 1):
        self._next = start

    def __call__(self) -> int:
        current = self._next
        self._next += 1
        return current


def _get_tool_call_state(store: Dict[int, Dict[str, Any]], index: int) -> Dict[str, Any]:
    if index not in store:
        store[index] = {
            "id": None,
            "name": None,
            "args_buffer": "",
            "json_sent": False,
            "claude_index": None,
            "started": False,
        }
    return store[index]


def _handle_openai_delta(
    logger,
    delta: Dict[str, Any],
    text_block_index: int,
    reasoning_state: _ReasoningBlockState,
    current_tool_calls: Dict[int, Dict[str, Any]],
    allocator: _ContentIndexAllocator,
) -> Tuple[List[str], int]:
    events: List[str] = []
    char_increment = 0

    for reasoning_text in _merge_reasoning_fields(delta):
        char_increment += len(reasoning_text)
        events.extend(reasoning_state.emit_delta(reasoning_text))

    if delta.get("content") is not None:
        char_increment += len(delta["content"] or "")
        events.append(
            _build_sse_event(
                logger,
                Constants.EVENT_CONTENT_BLOCK_DELTA,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                    "index": text_block_index,
                    "delta": {
                        "type": Constants.DELTA_TEXT,
                        "text": delta["content"],
                    },
                },
            )
        )

    for tc_delta in delta.get("tool_calls", []) or []:
        tc_index = tc_delta.get("index", 0)
        tool_call = _get_tool_call_state(current_tool_calls, tc_index)

        if tc_delta.get("id"):
            tool_call["id"] = tc_delta["id"]

        function_data = tc_delta.get(Constants.TOOL_FUNCTION, {}) or {}
        if function_data.get("name"):
            tool_call["name"] = function_data["name"]

        if (
            tool_call.get("id")
            and tool_call.get("name")
            and not tool_call.get("started")
        ):
            claude_index = allocator()
            tool_call["claude_index"] = claude_index
            tool_call["started"] = True
            events.append(
                _build_sse_event(
                    logger,
                    Constants.EVENT_CONTENT_BLOCK_START,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_START,
                        "index": claude_index,
                        "content_block": {
                            "type": Constants.CONTENT_TOOL_USE,
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "input": {},
                        },
                    },
                )
            )

        if (
            "arguments" in function_data
            and tool_call.get("started")
            and function_data["arguments"] is not None
        ):
            tool_call["args_buffer"] += function_data["arguments"]
            try:
                json.loads(tool_call["args_buffer"])
                if not tool_call.get("json_sent"):
                    events.append(
                        _build_sse_event(
                            logger,
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": tool_call["claude_index"],
                                "delta": {
                                    "type": Constants.DELTA_INPUT_JSON,
                                    "partial_json": tool_call["args_buffer"],
                                },
                            },
                        )
                    )
                    tool_call["json_sent"] = True
            except json.JSONDecodeError:
                pass

    return events, char_increment


def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    # Build Claude content blocks
    content_blocks = []

    # Add text content
    text_content = message.get("content")
    if text_content is not None:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    # Add tool calls
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            try:
                arguments = json.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    # Map finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)

    # Build Claude response
    claude_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get(
                "completion_tokens", 0
            ),
        },
    }

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger
):
    """Convert OpenAI streaming response to Claude streaming format."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    logger.debug(
        "Starting streaming conversion for model=%s, request_id=%s",
        original_request.model,
        message_id,
    )

    # Send initial SSE events
    yield _build_sse_event(
        logger,
        Constants.EVENT_MESSAGE_START,
        {
            "type": Constants.EVENT_MESSAGE_START,
            "message": {
                "id": message_id,
                "type": "message",
                "role": Constants.ROLE_ASSISTANT,
                "model": original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    yield _build_sse_event(
        logger,
        Constants.EVENT_CONTENT_BLOCK_START,
        {
            "type": Constants.EVENT_CONTENT_BLOCK_START,
            "index": 0,
            "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
        },
    )

    yield _build_sse_event(
        logger,
        Constants.EVENT_PING,
        {"type": Constants.EVENT_PING},
    )

    text_block_index = 0
    allocator = _ContentIndexAllocator(start=1)
    reasoning_state = _ReasoningBlockState(logger=logger, allocate_index=allocator)
    current_tool_calls: Dict[int, Dict[str, Any]] = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data: Optional[Dict[str, int]] = None
    generated_chars = 0

    try:
        async for line in openai_stream:
            logger.debug("Upstream raw line: %r", line)
            if not line.strip() or not line.startswith("data: "):
                continue

            chunk_data = line[6:]
            if chunk_data.strip() == "[DONE]":
                break

            try:
                logger.debug("Upstream SSE chunk: %s", chunk_data)
                chunk = json.loads(chunk_data)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse SSE chunk: %s", chunk_data)
                logger.warning("Reasoning conversion parse error: %s", exc)
                continue

            usage = chunk.get("usage") or {}
            if usage:
                prompt_details = usage.get("prompt_tokens_details", {}) or {}
                usage_data = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_details.get("cached_tokens", 0),
                }

            choices = chunk.get("choices", []) or []
            if not choices:
                continue

            choice = choices[0] or {}
            delta = choice.get("delta", {}) or {}
            finish_reason = choice.get("finish_reason")

            events, char_increment = _handle_openai_delta(
                logger,
                delta,
                text_block_index,
                reasoning_state,
                current_tool_calls,
                allocator,
            )
            generated_chars += char_increment
            for event in events:
                yield event

            if finish_reason:
                if finish_reason == "length":
                    final_stop_reason = Constants.STOP_MAX_TOKENS
                elif finish_reason in ("tool_calls", "function_call"):
                    final_stop_reason = Constants.STOP_TOOL_USE
                elif finish_reason == "stop":
                    final_stop_reason = Constants.STOP_END_TURN
                else:
                    final_stop_reason = Constants.STOP_END_TURN

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"Streaming error: {exc}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(exc)}"},
        }
        yield _build_sse_event(logger, "error", error_event)
        return

    if usage_data is None:
        approx_tokens = max(1, (generated_chars + 3) // 4) if generated_chars else 1
        usage_data = {
            "input_tokens": approx_tokens,
            "output_tokens": approx_tokens,
            "cache_read_input_tokens": 0,
        }

    for event in reasoning_state.close():
        yield event

    yield _build_sse_event(
        logger,
        Constants.EVENT_CONTENT_BLOCK_STOP,
        {
            "type": Constants.EVENT_CONTENT_BLOCK_STOP,
            "index": text_block_index,
        },
    )

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield _build_sse_event(
                logger,
                Constants.EVENT_CONTENT_BLOCK_STOP,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                    "index": tool_data["claude_index"],
                },
            )

    delta_payload = {
        "type": Constants.EVENT_MESSAGE_DELTA,
        "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
    }
    if usage_data is not None:
        delta_payload["usage"] = usage_data

    yield _build_sse_event(
        logger,
        Constants.EVENT_MESSAGE_DELTA,
        delta_payload,
    )
    yield _build_sse_event(
        logger,
        Constants.EVENT_MESSAGE_STOP,
        {"type": Constants.EVENT_MESSAGE_STOP},
    )



async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Request,
    openai_client,
    request_id: str,
):
    """Convert OpenAI streaming response to Claude streaming format with cancellation support."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield _build_sse_event(
        logger,
        Constants.EVENT_MESSAGE_START,
        {
            "type": Constants.EVENT_MESSAGE_START,
            "message": {
                "id": message_id,
                "type": "message",
                "role": Constants.ROLE_ASSISTANT,
                "model": original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    yield _build_sse_event(
        logger,
        Constants.EVENT_CONTENT_BLOCK_START,
        {
            "type": Constants.EVENT_CONTENT_BLOCK_START,
            "index": 0,
            "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
        },
    )

    yield _build_sse_event(
        logger,
        Constants.EVENT_PING,
        {"type": Constants.EVENT_PING},
    )

    text_block_index = 0
    allocator = _ContentIndexAllocator(start=1)
    reasoning_state = _ReasoningBlockState(logger=logger, allocate_index=allocator)
    current_tool_calls: Dict[int, Dict[str, Any]] = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data: Optional[Dict[str, int]] = None
    generated_chars = 0

    try:
        async for line in openai_stream:
            if await http_request.is_disconnected():
                logger.info(f"Client disconnected, cancelling request {request_id}")
                openai_client.cancel_request(request_id)
                break

            logger.debug("Upstream raw line: %r", line)
            if not line.strip() or not line.startswith("data: "):
                continue

            chunk_data = line[6:]
            if chunk_data.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(chunk_data)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse chunk: %s", chunk_data)
                logger.warning("Reasoning conversion parse error: %s", exc)
                continue

            usage = chunk.get("usage") or {}
            if usage:
                prompt_details = usage.get("prompt_tokens_details", {}) or {}
                usage_data = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_details.get("cached_tokens", 0),
                }

            choices = chunk.get("choices", []) or []
            if not choices:
                continue

            choice = choices[0] or {}
            delta = choice.get("delta", {}) or {}
            finish_reason = choice.get("finish_reason")

            events, char_increment = _handle_openai_delta(
                logger,
                delta,
                text_block_index,
                reasoning_state,
                current_tool_calls,
                allocator,
            )
            generated_chars += char_increment
            for event in events:
                yield event

            if finish_reason:
                if finish_reason == "length":
                    final_stop_reason = Constants.STOP_MAX_TOKENS
                elif finish_reason in ("tool_calls", "function_call"):
                    final_stop_reason = Constants.STOP_TOOL_USE
                elif finish_reason == "stop":
                    final_stop_reason = Constants.STOP_END_TURN
                else:
                    final_stop_reason = Constants.STOP_END_TURN
                break

    except HTTPException as exc:
        if exc.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            error_event = {
                "type": "error",
                "error": {
                    "type": "cancelled",
                    "message": "Request was cancelled by client",
                },
            }
            yield _build_sse_event(logger, "error", error_event)
            return

        friendly_message = openai_client.classify_openai_error(exc.detail)
        logger.error(f"Streaming HTTPException ({exc.status_code}): {friendly_message}")
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": friendly_message},
        }
        yield _build_sse_event(logger, "error", error_event)
        return

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"Streaming error: {exc}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(exc)}"},
        }
        yield _build_sse_event(logger, "error", error_event)
        return

    if usage_data is None:
        approx_tokens = max(1, (generated_chars + 3) // 4) if generated_chars else 1
        usage_data = {
            "input_tokens": approx_tokens,
            "output_tokens": approx_tokens,
            "cache_read_input_tokens": 0,
        }

    for event in reasoning_state.close():
        yield event

    yield _build_sse_event(
        logger,
        Constants.EVENT_CONTENT_BLOCK_STOP,
        {
            "type": Constants.EVENT_CONTENT_BLOCK_STOP,
            "index": text_block_index,
        },
    )

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield _build_sse_event(
                logger,
                Constants.EVENT_CONTENT_BLOCK_STOP,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                    "index": tool_data["claude_index"],
                },
            )

    delta_payload = {
        "type": Constants.EVENT_MESSAGE_DELTA,
        "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
    }
    if usage_data is not None:
        delta_payload["usage"] = usage_data

    yield _build_sse_event(
        logger,
        Constants.EVENT_MESSAGE_DELTA,
        delta_payload,
    )

    yield _build_sse_event(
        logger,
        Constants.EVENT_MESSAGE_STOP,
        {"type": Constants.EVENT_MESSAGE_STOP},
    )
