import asyncio
import contextlib
import hashlib
import json
import os
import time
from dataclasses import dataclass
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any, List, Iterable
import httpx
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai._exceptions import (
    APIError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)

from src.core.constants import Constants
from src.core.tool_executor import ToolExecutor
from src.core.logging import logger
from src.core.config import config


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: Any

class AsyncRateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._next_time = 0.0
        self._lock: Optional[asyncio.Lock] = None

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = loop.time()
            self._next_time = now + self.min_interval


class OpenAIClient:
    """Async OpenAI client with cancellation support."""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 90, api_version: Optional[str] = None, custom_headers: Optional[Dict[str, str]] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}
        # Detect Azure usage either via explicit api_version or azure endpoint URL
        self.is_azure = bool(api_version) or (
            isinstance(base_url, str) and ("azure.com" in base_url or "openai.azure" in base_url)
        )
        
        # Prepare default headers
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "claude-proxy/1.0.0"
        }
        
        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}
        
        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                default_headers=all_headers,
                max_retries=0,
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=all_headers,
                max_retries=0,
            )
        self.active_requests: Dict[str, asyncio.Event] = {}
        self.tool_executor = ToolExecutor()
        self.buffer_rate_limiter = AsyncRateLimiter(
            config.buffer_stream_rate_limit_seconds
        )
        self._conversation_cache: Dict[str, Dict[str, Any]] = {}
        self._conversation_cache_ttl = config.buffer_stream_cache_ttl_seconds
    
    def _adapt_request_for_provider(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize request parameters for specific providers (e.g., Azure)."""
        # Work on a shallow copy to avoid side effects
        req = dict(request)
        
        # Azure's Chat Completions for some models require `max_completion_tokens`
        if self.is_azure:
            if "max_tokens" in req:
                # Only translate when the azure-specific parameter isn't already set
                if "max_completion_tokens" not in req:
                    req["max_completion_tokens"] = req.pop("max_tokens")
                else:
                    # If both are present, drop the generic to avoid Azure 400s
                    req.pop("max_tokens", None)
        
        return req
    
    async def _await_with_cancellation(
        self, task: asyncio.Task, cancel_event: Optional[asyncio.Event]
    ):
        if cancel_event is None:
            return await task

        cancel_task = asyncio.create_task(cancel_event.wait())
        try:
            done, pending = await asyncio.wait(
                [task, cancel_task], return_when=asyncio.FIRST_COMPLETED
            )

            if cancel_task in done:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise HTTPException(status_code=499, detail="Request cancelled by client")

            return await task
        finally:
            cancel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_task

    def _cache_key(self, request: Dict[str, Any]) -> str:
        relevant = {
            "model": request.get("model"),
            "messages": request.get("messages"),
            "temperature": request.get("temperature"),
            "max_tokens": request.get("max_tokens"),
            "stream": request.get("stream"),
        }
        serialized = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _maybe_get_cached_response(
        self, request: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        ttl = self._conversation_cache_ttl
        if ttl <= 0:
            return None
        cache_key = self._cache_key(request)
        entry = self._conversation_cache.get(cache_key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > ttl:
            self._conversation_cache.pop(cache_key, None)
            return None
        logger.info(
            "Serving cached response for model=%s", request.get("model")
        )
        return entry["response"]

    def _store_cached_response(self, request: Dict[str, Any], response: Dict[str, Any]):
        ttl = self._conversation_cache_ttl
        if ttl <= 0:
            return
        cache_key = self._cache_key(request)
        self._conversation_cache[cache_key] = {
            "timestamp": time.time(),
            "response": response,
        }

    def _store_failed_request(self, request: Dict[str, Any]):
        if self._conversation_cache_ttl <= 0:
            return
        cache_key = self._cache_key(request)
        self._conversation_cache.pop(cache_key, None)

    async def _call_chat_completions_with_retry(
        self,
        request_payload: Dict[str, Any],
        cancel_event: Optional[asyncio.Event] = None,
    ):
        attempt = 0
        last_rate_error: Optional[Exception] = None
        total_attempts = max(0, config.max_retries) + 1
        model_name = request_payload.get("model")

        while attempt < total_attempts:
            await self._apply_rate_limit(model_name)
            try:
                completion_task = asyncio.create_task(
                    self.client.chat.completions.create(**request_payload)
                )
                return await self._await_with_cancellation(
                    completion_task, cancel_event
                )
            except RateLimitError as e:
                last_rate_error = e
                attempt += 1
                if attempt >= total_attempts:
                    break
                delay = config.retry_backoff_seconds * attempt
                logger.warning(
                    "Rate limit encountered (attempt %d/%d). Retrying in %.3fs",
                    attempt,
                    total_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
            except APIError as e:
                status_code = getattr(e, "status_code", None)
                if status_code == 429 and attempt < total_attempts - 1:
                    attempt += 1
                    delay = config.retry_backoff_seconds * attempt
                    logger.warning(
                        "API 429 encountered (attempt %d/%d). Retrying in %.3fs",
                        attempt,
                        total_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise HTTPException(
                    status_code=status_code or 500,
                    detail=self.classify_openai_error(str(e)),
                )

        if last_rate_error is not None:
            raise HTTPException(
                status_code=429,
                detail=self.classify_openai_error(str(last_rate_error)),
            )

        raise HTTPException(
            status_code=429,
            detail="Exceeded retry attempts after rate limiting",
        )

    async def _apply_rate_limit(self, model_name: Optional[str]) -> None:
        if not model_name:
            return
        if not config.should_buffer_stream_model(model_name):
            return
        await self.buffer_rate_limiter.wait()

    async def create_chat_completion(self, request: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        try:
            payload, api_mode = self._extract_request_bundle(request)

            if api_mode == "responses":
                completion_task = asyncio.create_task(
                    self._execute_responses_flow(payload, request_id=request_id)
                )
                completion = await self._await_with_cancellation(
                    completion_task, cancel_event if request_id else None
                )
                return (
                    completion
                    if isinstance(completion, dict)
                    else completion.model_dump()
                )

            normalized_request = self._adapt_request_for_provider(payload)
            cached_response = self._maybe_get_cached_response(normalized_request)
            if cached_response is not None:
                return cached_response
            model_name = normalized_request.get("model")
            if config.should_buffer_stream_model(model_name):
                logger.debug(
                    "Direct streaming fallback for non-stream request (model=%s)",
                    model_name,
                )
                result = await self._request_via_streaming_fallback(
                    normalized_request,
                    request_id,
                    cancel_event if request_id else None,
                )
                self._store_cached_response(normalized_request, result)
                return result
            try:
                completion = await self._call_chat_completions_with_retry(
                    normalized_request,
                    cancel_event if request_id else None,
                )
                result = (
                    completion
                    if isinstance(completion, dict)
                    else completion.model_dump()
                )
                self._store_cached_response(normalized_request, result)
                return result
            except BadRequestError as e:
                if self._requires_streaming_fallback(e, normalized_request):
                    logger.info(
                        "Retrying non-streaming request via streaming fallback for model=%s",
                        normalized_request.get("model"),
                    )
                    result = await self._request_via_streaming_fallback(
                        normalized_request,
                        request_id,
                        cancel_event if request_id else None,
                    )
                    self._store_cached_response(normalized_request, result)
                    return result
                self._store_failed_request(normalized_request)
                raise

        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def create_chat_completion_stream(
        self, request: Dict[str, Any], request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            payload, api_mode = self._extract_request_bundle(request)

            if api_mode == "responses":
                chat_response = await self._execute_responses_flow(
                    payload, request_id=request_id
                )
                async for chunk in self._responses_stream_from_chat(chat_response):
                    if request_id and request_id in self.active_requests:
                        if self.active_requests[request_id].is_set():
                            raise HTTPException(
                                status_code=499, detail="Request cancelled by client"
                            )
                    yield chunk
            else:
                normalized_request = self._adapt_request_for_provider(payload)
                model_name = normalized_request.get("model", "")
                stream_options = dict(normalized_request.get("stream_options") or {})
                stream_options.setdefault("include_usage", True)
                normalized_request["stream_options"] = stream_options
                normalized_request["stream"] = True

                force_buffer = config.should_buffer_stream_model(model_name)
                cached_response = self._maybe_get_cached_response(normalized_request)
                if cached_response is not None:
                    async def cached_generator():
                        for chunk in self._generate_buffered_stream_chunks(
                            cached_response
                        ):
                            yield chunk

                    async for chunk in cached_generator():
                        yield chunk
                    return

                if force_buffer:
                    buffered_request = dict(normalized_request)
                    buffered_request.pop("stream", None)
                    buffered_request.pop("stream_options", None)

                    logger.debug(
                        "Using buffered streaming fallback for model=%s", model_name
                    )

                    try:
                        completion = await self._call_chat_completions_with_retry(
                            buffered_request,
                            cancel_event if request_id else None,
                        )

                        completion_dict = (
                            completion
                            if isinstance(completion, dict)
                            else completion.model_dump()
                        )
                        self._store_cached_response(buffered_request, completion_dict)
                    except BadRequestError as e:
                        if self._requires_streaming_fallback(e, buffered_request):
                            logger.info(
                                "Buffered streaming fallback hitting streaming-only constraint; reissuing as streaming for model=%s",
                                model_name,
                            )
                            completion_dict = await self._request_via_streaming_fallback(
                                buffered_request,
                                request_id,
                                cancel_event if request_id else None,
                            )
                            self._store_cached_response(
                                buffered_request, completion_dict
                            )
                        else:
                            self._store_failed_request(buffered_request)
                            raise

                    async def buffered_generator():
                        for chunk in self._generate_buffered_stream_chunks(
                            completion_dict
                        ):
                            if request_id and request_id in self.active_requests:
                                if self.active_requests[request_id].is_set():
                                    raise HTTPException(
                                        status_code=499,
                                        detail="Request cancelled by client",
                                    )
                            yield chunk

                    async for chunk in buffered_generator():
                        yield chunk
                    return

                streaming_completion = await self._call_chat_completions_with_retry(
                    normalized_request,
                    cancel_event if request_id else None,
                )

                try:
                    async for chunk in streaming_completion:
                        if request_id and request_id in self.active_requests:
                            if self.active_requests[request_id].is_set():
                                raise HTTPException(
                                    status_code=499, detail="Request cancelled by client"
                                )

                        chunk_dict = chunk.model_dump()
                        chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                        yield f"data: {chunk_json}"
                finally:
                    if hasattr(streaming_completion, "aclose"):
                        with contextlib.suppress(Exception):
                            await streaming_completion.aclose()

                yield "data: [DONE]"

        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()
        
        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or "country, region, or territory not supported" in error_str:
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."
        
        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."
        
        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
        
        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
        
        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."

        # Tool calls / functions unsupported by provider or model
        if (
            "tool call is not supported" in error_str
            or "tools are not supported" in error_str
            or "tool_calls are not supported" in error_str
            or "function call is not supported" in error_str
            or ("invalid_parameter_error" in error_str and "tool" in error_str)
        ):
            return (
                "Tool calls are not supported by the upstream provider/model. "
                "Remove tools/tool_choice from the request or switch to a model that supports tools."
            )

        # Default: return original message
        return str(error_detail)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False

    def _extract_request_bundle(self, request: Dict[str, Any]):
        """Support both legacy dict requests and structured bundles."""
        api_mode = "chat_completions"
        payload = request

        if hasattr(request, "payload") and hasattr(request, "api_mode"):
            payload = request.payload
            api_mode = getattr(request, "api_mode", "chat_completions")
        elif isinstance(request, dict) and "payload" in request and "api_mode" in request:
            payload = request["payload"]
            api_mode = request["api_mode"]

        return payload, api_mode

    def _requires_streaming_fallback(
        self, error: BadRequestError, request: Dict[str, Any]
    ) -> bool:
        if request.get("stream"):
            return False

        message_parts = [str(error)]
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            try:
                message_parts.append(json.dumps(body, ensure_ascii=False))
            except Exception:
                message_parts.append(str(body))
        elif body is not None:
            message_parts.append(str(body))

        combined = " ".join(message_parts).lower()
        if "only support stream mode" in combined:
            return True
        return False

    async def _request_via_streaming_fallback(
        self,
        request: Dict[str, Any],
        request_id: Optional[str],
        cancel_event: Optional[asyncio.Event],
    ) -> Dict[str, Any]:
        streaming_request = dict(request)
        streaming_request["stream"] = True
        stream_options = dict(streaming_request.get("stream_options") or {})
        stream_options["include_usage"] = True
        streaming_request["stream_options"] = stream_options

        streaming_completion = await self._call_chat_completions_with_retry(
            streaming_request,
            cancel_event,
        )

        try:
            return await self._collect_streaming_completion(
                streaming_completion, request_id, cancel_event
            )
        finally:
            if hasattr(streaming_completion, "aclose"):
                with contextlib.suppress(Exception):
                    await streaming_completion.aclose()

    async def _collect_streaming_completion(
        self,
        streaming_completion,
        request_id: Optional[str],
        cancel_event: Optional[asyncio.Event],
    ) -> Dict[str, Any]:
        content_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        response_id: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None

        async for chunk in streaming_completion:
            if request_id and request_id in self.active_requests:
                if self.active_requests[request_id].is_set():
                    raise HTTPException(status_code=499, detail="Request cancelled by client")
            if cancel_event and cancel_event.is_set():
                raise HTTPException(status_code=499, detail="Request cancelled by client")

            chunk_dict = chunk.model_dump()
            logger.debug(
                "Streaming fallback chunk (request_id=%s): %s",
                request_id,
                json.dumps(chunk_dict, ensure_ascii=False),
            )
            if response_id is None:
                response_id = chunk_dict.get("id")
            if chunk_dict.get("usage"):
                usage = chunk_dict["usage"]

            for choice in chunk_dict.get("choices", []):
                delta = choice.get("delta", {}) or {}

                if delta.get("content") is not None:
                    content_parts.append(delta["content"])

                if "tool_calls" in delta:
                    for call_delta in delta["tool_calls"] or []:
                        index = call_delta.get("index", 0)
                        entry = tool_calls.setdefault(
                            index,
                            {
                                "id": None,
                                "type": Constants.TOOL_FUNCTION,
                                Constants.TOOL_FUNCTION: {"name": None, "arguments": ""},
                            },
                        )

                        if call_delta.get("id"):
                            entry["id"] = call_delta["id"]
                        if call_delta.get("type"):
                            entry["type"] = call_delta["type"]

                        function_delta = call_delta.get(Constants.TOOL_FUNCTION, {}) or {}
                        if function_delta.get("name"):
                            entry[Constants.TOOL_FUNCTION]["name"] = function_delta["name"]
                        if (
                            "arguments" in function_delta
                            and function_delta["arguments"] is not None
                        ):
                            entry[Constants.TOOL_FUNCTION]["arguments"] += function_delta[
                                "arguments"
                            ]

                finish = choice.get("finish_reason")
                if finish:
                    finish_reason = finish

        tool_calls_list: List[Dict[str, Any]] = []
        for index in sorted(tool_calls.keys()):
            entry = tool_calls[index]
            function_data = entry[Constants.TOOL_FUNCTION]
            tool_calls_list.append(
                {
                    "id": entry.get("id"),
                    "type": entry.get("type", Constants.TOOL_FUNCTION),
                    Constants.TOOL_FUNCTION: {
                        "name": function_data.get("name") or "",
                        "arguments": function_data.get("arguments", ""),
                    },
                }
            )

        message: Dict[str, Any] = {"role": "assistant"}
        if content_parts:
            message["content"] = "".join(content_parts)
        else:
            message["content"] = None

        if tool_calls_list:
            message["tool_calls"] = tool_calls_list

        return {
            "id": response_id or "",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _generate_buffered_stream_chunks(self, completion: Dict[str, Any]):
        """Convert a non-streaming completion into SSE-like streaming chunks."""

        choices = completion.get("choices", [])
        if not choices:
            logger.warning("Buffered completion missing choices payload")
            yield "data: [DONE]"
            return

        choice = choices[0] or {}
        message = choice.get("message", {}) or {}
        finish_reason = choice.get("finish_reason") or "stop"

        delta_payload: Dict[str, Any] = {}

        def _extract_reasoning_text(raw: Any) -> Optional[str]:
            if raw is None:
                return None

            def _collect(value: Any) -> Iterable[str]:
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
                    for key in ("content", "reasoning", "messages"):
                        nested = value.get(key)
                        if isinstance(nested, (list, tuple)):
                            for item in nested:
                                yield from _collect(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        yield from _collect(item)

            parts = list(_collect(raw))
            if not parts:
                return None
            return "".join(parts)

        content = message.get("content")
        if content is not None:
            delta_payload["content"] = content

        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking")
        )
        reasoning_text = _extract_reasoning_text(reasoning)
        if reasoning_text:
            delta_payload["reasoning_content"] = reasoning_text

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            serialized_calls = []
            for idx, tool_call in enumerate(tool_calls):
                serialized_calls.append(
                    {
                        "index": idx,
                        "id": tool_call.get("id"),
                        "type": tool_call.get("type"),
                        Constants.TOOL_FUNCTION: tool_call.get(
                            Constants.TOOL_FUNCTION, {}
                        ),
                    }
                )
            delta_payload["tool_calls"] = serialized_calls

        if delta_payload:
            first_chunk = {
                "choices": [
                    {
                        "delta": delta_payload,
                        "finish_reason": None,
                    }
                ]
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}"

        final_chunk: Dict[str, Any] = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ]
        }

        usage = completion.get("usage")
        if usage:
            final_chunk["usage"] = usage

        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}"
        yield "data: [DONE]"

    def _convert_responses_result_to_chat_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Responses API result to look like a Chat Completion."""
        output_text_segments: List[str] = []
        reasoning_segments: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        finish_reason = "stop"
        usage = response.get("usage", {})

        def extract_text_from_block(block: Dict[str, Any]) -> str:
            if not isinstance(block, dict):
                return str(block)
            if "text" in block and isinstance(block["text"], str):
                return block["text"]
            # Some reasoning blocks have nested items
            nested_texts = []
            for key in ("content", "reasoning", "messages"):
                nested = block.get(key)
                if isinstance(nested, list):
                    for item in nested:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            nested_texts.append(item["text"])
            return "\n".join(nested_texts)

        outputs = response.get("output", [])
        tool_call_counter = 0
        for output in outputs:
            finish_reason = output.get("finish_reason", finish_reason)
            for part in output.get("content", []):
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "output_text":
                    output_text_segments.append(part.get("text", "") or "")
                elif part_type == "reasoning":
                    reasoning_text = extract_text_from_block(part)
                    if reasoning_text:
                        reasoning_segments.append(reasoning_text)
                elif part_type in ("tool_call", "function_call"):
                    function_block = part.get("function", {})
                    tool_call_id = part.get("id") or part.get("tool_call_id")
                    if not tool_call_id:
                        tool_call_counter += 1
                        tool_call_id = f"tool_call_{tool_call_counter}"
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": Constants.TOOL_FUNCTION,
                            Constants.TOOL_FUNCTION: {
                                "name": function_block.get("name", ""),
                                "arguments": (
                                    function_block.get("arguments")
                                    if isinstance(function_block.get("arguments"), str)
                                    else json.dumps(function_block.get("arguments", {}), ensure_ascii=False)
                                ),
                            },
                        }
                    )
                else:
                    text_value = extract_text_from_block(part)
                    if text_value:
                        output_text_segments.append(text_value)

        output_text = response.get("output_text")
        if output_text is None and output_text_segments:
            output_text = "".join(output_text_segments)

        if output_text is None and reasoning_segments and not tool_calls:
            output_text = "\n\n".join(reasoning_segments)

        logger.debug(
            "Responses API parsed output (text_segments=%d, reasoning_segments=%d, tool_calls=%d)",
            len(output_text_segments),
            len(reasoning_segments),
            len(tool_calls),
        )

        chat_response = {
            "id": response.get("id", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text if output_text is not None else None,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
            },
        }

        return chat_response

    async def _execute_responses_flow(
        self, payload: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute Responses API call, handling tool call loops until completion."""
        responses_payload = dict(payload)
        responses_payload.pop("stream", None)

        tag = request_id or "unknown"
        logger.debug(
            "Responses API invocation (request_id=%s, model=%s, has_tools=%s)",
            tag,
            responses_payload.get("model"),
            bool(responses_payload.get("tools")),
        )

        response = await self.client.responses.create(**responses_payload)
        safety_counter = 0

        while True:
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else dict(response)
            )

            self._dump_responses_payload(response_dict, tag, safety_counter)
            self._log_responses_output_debug(response_dict, tag)

            tool_calls = self._extract_responses_tool_calls(response_dict)
            if tool_calls:
                safety_counter += 1
                if safety_counter > 10:
                    raise HTTPException(
                        status_code=500,
                        detail="Exceeded maximum tool iteration depth for Responses API",
                    )

                tool_outputs = []
                for call in tool_calls:
                    logger.info(
                        "Executing tool '%s' (request_id=%s, call_id=%s)",
                        call.name,
                        tag,
                        call.call_id,
                    )
                    logger.debug(
                        "Tool arguments for '%s' (request_id=%s): %s",
                        call.name,
                        tag,
                        call.arguments,
                    )
                    output = await self.tool_executor.execute(
                        call.name, call.arguments, request_id=request_id
                    )
                    logger.debug(
                        "Tool '%s' output (request_id=%s, call_id=%s): %s",
                        call.name,
                        tag,
                        call.call_id,
                        output,
                    )
                    tool_outputs.append(
                        {
                            "tool_call_id": call.call_id,
                            "output": output,
                        }
                    )

                if not tool_outputs:
                    raise HTTPException(
                        status_code=500,
                        detail="Tool execution produced no outputs for Responses API",
                    )

                response_id = response_dict.get("id")
                if not response_id:
                    raise HTTPException(
                        status_code=500,
                        detail="Responses API reply missing identifier for tool submission",
                    )

                logger.info(
                    "Submitting %d tool outputs to Responses API (request_id=%s, response_id=%s)",
                    len(tool_outputs),
                    tag,
                    response_id,
                )
                response = await self._submit_responses_tool_outputs(
                    response_id=response_id,
                    tool_outputs=tool_outputs,
                )
                continue

            chat_response = self._convert_responses_result_to_chat_format(response_dict)
            logger.debug(
                "Responses API completed (request_id=%s, finish_reason=%s)",
                tag,
                chat_response.get("choices", [{}])[0].get("finish_reason"),
            )
            return chat_response

    def _extract_responses_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool call requests from a Responses API response."""
        tool_calls: List[ToolCall] = []

        for item in response.get("output", []):
            item_type = item.get("type")
            # Some responses may surface tool calls directly on the item.
            if item_type in ("tool_call", "function_call"):
                maybe_call = self._parse_tool_call_dict(item)
                if maybe_call:
                    tool_calls.append(maybe_call)
                continue

            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") not in ("tool_call", "function_call"):
                    continue
                maybe_call = self._parse_tool_call_dict(content)
                if maybe_call:
                    tool_calls.append(maybe_call)

        return tool_calls

    def _parse_tool_call_dict(self, data: Dict[str, Any]) -> Optional[ToolCall]:
        """Normalise tool call dict into ToolCall data."""
        call_id = (
            data.get("id")
            or data.get("tool_call_id")
            or data.get("call_id")
        )
        name = data.get("name") or data.get("tool_name")
        if not call_id or not name:
            return None

        arguments = (
            data.get("arguments")
            or data.get("input")
            or data.get("payload")
            or {}
        )

        if isinstance(arguments, str):
            arguments = self._parse_arguments_string(arguments)

        return ToolCall(call_id=call_id, name=name, arguments=arguments)

    def _parse_arguments_string(self, raw: str) -> Any:
        """Attempt to parse tool arguments represented as a string."""
        try:
            return json.loads(raw)
        except Exception:
            # Fallback to returning the raw string when parsing fails.
            return raw

    def _log_responses_output_debug(self, response: Dict[str, Any], tag: str) -> None:
        """Emit trimmed debug information about Responses output structure."""
        output = response.get("output")
        if not output:
            logger.debug(
                "Responses output empty (request_id=%s, available_keys=%s)",
                tag,
                list(response.keys()),
            )
            return

        preview = []
        for idx, item in enumerate(output[:2]):
            if isinstance(item, dict):
                preview.append(
                    {
                        "index": idx,
                        "type": item.get("type"),
                        "keys": list(item.keys()),
                    }
                )
            else:
                preview.append({"index": idx, "repr": repr(item)})

        logger.debug(
            "Responses output preview (request_id=%s): %s ... len=%d",
            tag,
            preview,
            len(output),
        )

    def _dump_responses_payload(
        self, response: Dict[str, Any], tag: str, iteration: int
    ) -> None:
        """Persist a snapshot of the raw Responses payload for debugging."""
        try:
            dump_dir = "logs/responses"
            os.makedirs(dump_dir, exist_ok=True)
            filename = os.path.join(
                dump_dir,
                f"{tag.replace('/', '_')}_iter{iteration}.json",
            )
            with open(filename, "w", encoding="utf-8") as fh:
                json.dump(response, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to dump Responses payload (request_id=%s): %s", tag, exc
            )

    async def _submit_responses_tool_outputs(
        self, response_id: str, tool_outputs: List[Dict[str, Any]]
    ):
        """Submit tool outputs for a Responses API conversation using raw POST."""
        last_error: Optional[Exception] = None
        endpoints = [
            f"/responses/{response_id}/submit_tool_outputs",
            f"/responses/{response_id}/tool_outputs",
        ]

        for endpoint in endpoints:
            try:
                result = await self.client.responses._post(
                    endpoint,
                    body={"tool_outputs": tool_outputs},
                    cast_to=dict,
                )
                return result if isinstance(result, dict) else result.model_dump()
            except AttributeError as exc:
                last_error = exc
                break
            except APIError as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code == 404:
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code == 404:
                    continue
                raise

        raise HTTPException(
            status_code=500,
            detail=f"Responses API tool output submission failed: {last_error}",
        )


    async def _responses_stream_from_chat(self, chat_response: Dict[str, Any]):
        """Yield a minimal SSE-compatible stream from a chat completion style response."""
        content = None
        finish_reason = "stop"
        tool_calls = []
        choices = chat_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            finish_reason = choices[0].get("finish_reason", "stop") or "stop"
            tool_calls = message.get("tool_calls") or []

        # Emit tool call deltas first to mimic OpenAI streaming semantics
        for index, tool_call in enumerate(tool_calls):
            function_payload = tool_call.get(Constants.TOOL_FUNCTION, {})
            chunk = {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tool_call.get("id"),
                                    "type": tool_call.get("type"),
                                    "function": {
                                        "name": function_payload.get("name"),
                                        "arguments": function_payload.get("arguments", ""),
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
            chunk_str = json.dumps(chunk, ensure_ascii=False)
            logger.debug("SSE tool chunk: %s", chunk_str)
            yield f"data: {chunk_str}"

        # Emit content chunk if any
        if content:
            chunk = {
                "choices": [
                    {
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ]
            }
            chunk_str = json.dumps(chunk, ensure_ascii=False)
            logger.debug("SSE content chunk: %s", chunk_str)
            yield f"data: {chunk_str}"

        final_chunk = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ]
        }
        chunk_str = json.dumps(final_chunk, ensure_ascii=False)
        logger.debug("SSE final chunk: %s", chunk_str)
        yield f"data: {chunk_str}"
        yield "data: [DONE]"
