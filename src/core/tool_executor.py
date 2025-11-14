import json
import logging
from collections import defaultdict
from typing import Any, Dict, Optional


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""


class ToolExecutor:
    """Execute reasoning tool calls on behalf of the upstream model."""

    def __init__(self) -> None:
        # Maintain lightweight per-request state for tools such as TodoWrite.
        self.todo_state: Dict[str, Any] = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    async def execute(
        self, tool_name: str, arguments: Any, request_id: Optional[str] = None
    ) -> str:
        """Execute a tool call and return a JSON-serialised result."""
        handler_name = f"_handle_{tool_name.lower()}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            self.logger.warning(
                "Received call for unsupported tool '%s' (request_id=%s)",
                tool_name,
                request_id or "unknown",
            )
            return self._fallback_response(tool_name, arguments)

        try:
            result_payload = await handler(arguments, request_id=request_id)
        except ToolExecutionError as exc:
            result_payload = {"status": "error", "message": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            result_payload = {
                "status": "error",
                "message": f"Unexpected error executing {tool_name}: {exc}",
            }

        return json.dumps(result_payload, ensure_ascii=False)

    async def _handle_todowrite(
        self, arguments: Any, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record todo items emitted by the model."""
        todos_snapshot = {
            "arguments": arguments,
            "request_id": request_id,
        }
        if request_id:
            self.todo_state[request_id].append(todos_snapshot)
        return {
            "status": "ok",
            "message": "Todo list recorded",
            "todos": arguments,
        }

    async def _handle_task(
        self, arguments: Any, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Acknowledge task orchestration requests."""
        return {
            "status": "ok",
            "message": "Task acknowledged",
            "task": arguments,
        }

    async def _handle_todofinish(
        self, arguments: Any, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle Todo completion notifications."""
        if request_id and request_id in self.todo_state:
            self.todo_state[request_id].append(
                {"completion": arguments, "request_id": request_id}
            )
        return {
            "status": "ok",
            "message": "Todo completion recorded",
            "completion": arguments,
        }

    def _fallback_response(self, tool_name: str, arguments: Any) -> str:
        """Generic response for tools without explicit handlers."""
        payload = {
            "status": "noop",
            "message": f"Tool '{tool_name}' is not implemented in proxy",
            "arguments": arguments,
        }
        self.logger.debug("Returning fallback payload for tool '%s'", tool_name)
        return json.dumps(payload, ensure_ascii=False)
