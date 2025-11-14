import json
import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal, Tuple
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage, ClaudeThinkingConfig
from src.core.config import config
import logging
from src.core.tool_policies import sanitize_tool_payload

logger = logging.getLogger(__name__)

TOOL_CALL_ID_MAX_LENGTH = 40


def supports_reasoning_model(model_name: str) -> bool:
    """Check whether the target model should receive reasoning parameters."""
    if not model_name:
        return False
    normalized = model_name.lower()
    return any(
        normalized.startswith(prefix.lower()) for prefix in config.reasoning_model_prefixes
    )


def supports_chat_reasoning_model(model_name: str) -> bool:
    """Check whether the target model uses chat completions for reasoning."""
    if not model_name:
        return False
    normalized = model_name.lower()
    return any(
        normalized.startswith(prefix.lower()) for prefix in config.reasoning_chat_prefixes
    )


def normalize_tool_call_id(raw_id: str, tool_id_map: Dict[str, str]) -> str:
    """Normalize Claude tool IDs to satisfy Azure's length restriction."""
    if not raw_id:
        raw_id = uuid.uuid4().hex

    if raw_id in tool_id_map:
        return tool_id_map[raw_id]

    hashed = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
    normalized = hashed[:TOOL_CALL_ID_MAX_LENGTH]
    tool_id_map[raw_id] = normalized
    return normalized


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> "OpenAIRequestBundle":
    """Convert Claude API request format to OpenAI format."""

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Convert messages
    openai_messages = []
    tool_id_map: Dict[str, str] = {}

    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            openai_messages.append(
                {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
            )

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg, tool_id_map)
            openai_messages.append(openai_message)

            # Check if next message contains tool results
            if i + 1 < len(claude_request.messages) and not config.disable_tools:
                next_msg = claude_request.messages[i + 1]
                if (
                    next_msg.role == Constants.ROLE_USER
                    and isinstance(next_msg.content, list)
                    and any(
                        block.type == Constants.CONTENT_TOOL_RESULT
                        for block in next_msg.content
                        if hasattr(block, "type")
                    )
                ):
                    # Process tool results
                    i += 1  # Skip to tool result message
                    tool_results = convert_claude_tool_results(next_msg, tool_id_map)
                    openai_messages.extend(tool_results)

        i += 1

    # Build OpenAI request
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": min(
            max(claude_request.max_tokens, config.min_tokens_limit),
            config.max_tokens_limit,
        ),
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }
    logger.debug(
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools (optional disable via config)
    if claude_request.tools and not config.disable_tools:
        openai_tools = []
        for tool in claude_request.tools:
            if tool.name and tool.name.strip():
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice and not config.disable_tools:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    # If tools are disabled, scrub assistant tool_calls from prior messages to avoid provider errors
    if config.disable_tools:
        for m in openai_messages:
            if m.get("role") == Constants.ROLE_ASSISTANT and "tool_calls" in m:
                m.pop("tool_calls", None)
        openai_request.pop("tools", None)
        openai_request.pop("tool_choice", None)

    provider_key = config.resolve_provider_key()
    disabled_groups = config.get_disabled_tool_groups(provider_key)
    if disabled_groups:
        logger.debug(
            "Disabling tool groups for provider '%s': %s",
            provider_key,
            ", ".join(disabled_groups),
        )
        sanitize_tool_payload(openai_request, disabled_groups)

    openai_request = apply_provider_output_limits(openai_request)

    api_mode, payload = apply_reasoning_parameters(
        openai_request,
        claude_request.thinking,
        openai_model,
    )

    return OpenAIRequestBundle(payload=payload, api_mode=api_mode)


def apply_reasoning_parameters(
    openai_request: Dict[str, Any],
    thinking_config: Optional[ClaudeThinkingConfig],
    model_name: str,
) -> Tuple[str, Dict[str, Any]]:
    """Translate Claude thinking config to request payload and determine API mode."""
    enable_reasoning = False
    effort: Optional[str] = None
    verbosity: Optional[str] = None

    if thinking_config:
        enable_reasoning = thinking_config.enabled
        effort = thinking_config.effort or config.reasoning_effort
        verbosity = thinking_config.verbosity or config.reasoning_verbosity
    elif config.reasoning_default_enabled:
        enable_reasoning = True
        effort = config.reasoning_effort
        verbosity = config.reasoning_verbosity

    if not enable_reasoning:
        return "chat_completions", openai_request

    chat_reasoning = supports_chat_reasoning_model(model_name)
    responses_reasoning = supports_reasoning_model(model_name)

    if not (chat_reasoning or responses_reasoning):
        logger.debug(
            "Skipping reasoning parameters: model '%s' is not in reasoning prefixes %s",
            model_name,
            config.reasoning_model_prefixes,
        )
        return "chat_completions", openai_request

    if chat_reasoning:
        chat_request = dict(openai_request)
        extra_body = dict(chat_request.get("extra_body") or {})
        extra_body.setdefault("enable_thinking", True)
        chat_request["extra_body"] = extra_body
        max_tokens = chat_request.get("max_tokens")
        limit = max(1, config.reasoning_chat_max_output)
        if max_tokens is None or max_tokens > limit:
            chat_request["max_tokens"] = limit
        return "chat_completions", chat_request

    # Skip reasoning when tool interactions exist (not yet supported via Responses API)
    if contains_tool_messages(openai_request["messages"]):
        logger.info(
            "Reasoning requested but tool interactions present; falling back to chat completions."
        )
        return "chat_completions", openai_request

    responses_payload = convert_to_responses_payload(
        openai_request,
        effort=effort,
        verbosity=verbosity,
    )
    return "responses", responses_payload


def convert_to_responses_payload(
    openai_request: Dict[str, Any],
    effort: Optional[str],
    verbosity: Optional[str],
) -> Dict[str, Any]:
    """Convert a chat-completions payload into a Responses API payload."""
    messages = openai_request.get("messages", [])
    responses_messages = convert_messages_to_responses_format(messages)

    payload: Dict[str, Any] = {
        "model": openai_request["model"],
        "input": responses_messages,
    }

    if openai_request.get("temperature") is not None:
        payload["temperature"] = openai_request["temperature"]

    max_tokens = openai_request.get("max_tokens")
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens

    if "top_p" in openai_request and openai_request["top_p"] is not None:
        payload["top_p"] = openai_request["top_p"]

    if "stop" in openai_request:
        payload["stop"] = openai_request["stop"]

    # Carry over tools if present (Responses API supports same structure)
    if "tools" in openai_request:
        payload["tools"] = adapt_tools_for_responses(openai_request["tools"])
    if "tool_choice" in openai_request:
        adapted_choice = adapt_tool_choice_for_responses(openai_request["tool_choice"])
        if adapted_choice is not None:
            payload["tool_choice"] = adapted_choice

    reasoning_payload: Dict[str, Any] = {}
    if effort:
        reasoning_payload["effort"] = effort
    if reasoning_payload:
        payload["reasoning"] = reasoning_payload

    if verbosity:
        payload["text"] = {"verbosity": verbosity}

    return payload


def adapt_tools_for_responses(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Chat Completions tool schema to Responses API schema."""
    adapted: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == Constants.TOOL_FUNCTION:
            fn = tool.get(Constants.TOOL_FUNCTION, {})
            name = fn.get("name")
            if not name:
                continue
            converted = {"type": "function", "name": name}
            if fn.get("description"):
                converted["description"] = fn["description"]
            if fn.get("parameters"):
                converted["parameters"] = fn["parameters"]
            adapted.append(converted)
        else:
            adapted.append(tool)
    return adapted


def adapt_tool_choice_for_responses(choice: Any) -> Optional[Any]:
    """Normalize tool_choice structure for Responses API."""
    if choice is None:
        return None
    if isinstance(choice, str):
        return choice  # e.g., "auto" or "none"
    if not isinstance(choice, dict):
        return None

    choice_type = choice.get("type")
    if choice_type == Constants.TOOL_FUNCTION:
        fn = choice.get(Constants.TOOL_FUNCTION, {})
        name = fn.get("name")
        if not name:
            return None
        converted = {"type": "function", "name": name}
        if fn.get("parameters"):
            converted["parameters"] = fn["parameters"]
        return converted

    return choice

def convert_messages_to_responses_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adapt chat-completion messages into Responses API message blocks."""
    responses_messages: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if not role:
            continue

        content = message.get("content")
        responses_content: List[Dict[str, Any]] = []

        def text_type_for_role(msg_role: str) -> str:
            return (
                "output_text"
                if msg_role == Constants.ROLE_ASSISTANT
                else "input_text"
            )

        if isinstance(content, str):
            responses_content.append(
                {"type": text_type_for_role(role), "text": content}
            )
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    responses_content.append(
                        {
                            "type": text_type_for_role(role),
                            "text": block.get("text", ""),
                        }
                    )
                elif block_type == "image_url":
                    responses_content.append(
                        {
                            "type": "input_image",
                            "image_url": block.get("image_url", {}),
                        }
                    )
        elif content is None:
            pass
        else:
            responses_content.append(
                {"type": text_type_for_role(role), "text": str(content)}
            )

        message_entry: Dict[str, Any] = {"role": role}
        if responses_content:
            message_entry["content"] = responses_content
        if "tool_calls" in message:
            message_entry["tool_calls"] = message["tool_calls"]
        if "tool_call_id" in message:
            message_entry["tool_call_id"] = message["tool_call_id"]
        responses_messages.append(message_entry)
    return responses_messages


def contains_tool_messages(messages: List[Dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") == Constants.ROLE_TOOL:
            return True
        if message.get("role") == Constants.ROLE_ASSISTANT and message.get("tool_calls"):
            return True
    return False


def apply_provider_output_limits(openai_request: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp output tokens for providers with stricter ceilings."""
    max_tokens = openai_request.get("max_tokens")
    if not isinstance(max_tokens, int):
        return openai_request

    base_url = (config.openai_base_url or "").lower()
    limit = None

    if "dashscope.aliyuncs.com" in base_url:
        limit = config.provider_max_output_tokens or config.reasoning_chat_max_output or 16384

    if limit and max_tokens > limit:
        logger.debug(
            "Clamping max_tokens from %s to %s based on provider limits",
            max_tokens,
            limit,
        )
        openai_request["max_tokens"] = limit

    return openai_request


@dataclass
class OpenAIRequestBundle:
    payload: Dict[str, Any]
    api_mode: Literal["chat_completions", "responses"] = "chat_completions"


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": block.text})
        elif block.type == Constants.CONTENT_IMAGE:
            # Convert Claude image format to OpenAI format
            if (
                isinstance(block.source, dict)
                and block.source.get("type") == "base64"
                and "media_type" in block.source
                and "data" in block.source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                        },
                    }
                )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(
    msg: ClaudeMessage, tool_id_map: Dict[str, str]
) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": None}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            text_parts.append(block.text)
        elif block.type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": normalize_tool_call_id(block.id, tool_id_map),
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )
        elif block.type == Constants.CONTENT_THINKING:
            # Skip thinking metadata when forwarding to upstream models.
            continue

    openai_message = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = None

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(
    msg: ClaudeMessage, tool_id_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            if block.type == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(block.content)
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": normalize_tool_call_id(
                            block.tool_use_id, tool_id_map
                        ),
                        "content": content,
                    }
                )

    return tool_messages


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item, ensure_ascii=False))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"
