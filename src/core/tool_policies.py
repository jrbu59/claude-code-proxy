from __future__ import annotations

from typing import Dict, Iterable, List, MutableMapping, Optional, Set

from src.core.constants import Constants


TOOL_GROUPS: Dict[str, Set[str]] = {
    "tasking": {
        "Task",
        "ExitPlanMode",
        "AskUserQuestion",
        "Skill",
        "SlashCommand",
    },
    "filesystem": {
        "Read",
        "Edit",
        "Write",
        "NotebookEdit",
    },
    "shell": {
        "Bash",
        "BashOutput",
        "KillShell",
    },
    "search": {
        "Glob",
        "Grep",
        "WebSearch",
    },
    "productivity": {
        "WebFetch",
        "TodoWrite",
    },
    "playwright_mcp": {
        "mcp__playwright__browser_close",
        "mcp__playwright__browser_resize",
        "mcp__playwright__browser_console_messages",
        "mcp__playwright__browser_handle_dialog",
        "mcp__playwright__browser_evaluate",
        "mcp__playwright__browser_file_upload",
        "mcp__playwright__browser_fill_form",
        "mcp__playwright__browser_install",
        "mcp__playwright__browser_press_key",
        "mcp__playwright__browser_type",
        "mcp__playwright__browser_navigate",
        "mcp__playwright__browser_navigate_back",
        "mcp__playwright__browser_network_requests",
        "mcp__playwright__browser_take_screenshot",
        "mcp__playwright__browser_snapshot",
        "mcp__playwright__browser_click",
        "mcp__playwright__browser_drag",
        "mcp__playwright__browser_hover",
        "mcp__playwright__browser_select_option",
        "mcp__playwright__browser_tabs",
        "mcp__playwright__browser_wait_for",
    },
    "chrome_devtools_mcp": {
        "mcp__chrome-devtools__click",
        "mcp__chrome-devtools__close_page",
        "mcp__chrome-devtools__drag",
        "mcp__chrome-devtools__emulate",
        "mcp__chrome-devtools__evaluate_script",
        "mcp__chrome-devtools__fill",
        "mcp__chrome-devtools__fill_form",
        "mcp__chrome-devtools__get_console_message",
        "mcp__chrome-devtools__get_network_request",
        "mcp__chrome-devtools__handle_dialog",
        "mcp__chrome-devtools__hover",
        "mcp__chrome-devtools__list_console_messages",
        "mcp__chrome-devtools__list_network_requests",
        "mcp__chrome-devtools__list_pages",
        "mcp__chrome-devtools__navigate_page",
        "mcp__chrome-devtools__new_page",
        "mcp__chrome-devtools__performance_analyze_insight",
        "mcp__chrome-devtools__performance_start_trace",
        "mcp__chrome-devtools__performance_stop_trace",
        "mcp__chrome-devtools__press_key",
        "mcp__chrome-devtools__resize_page",
        "mcp__chrome-devtools__select_page",
        "mcp__chrome-devtools__take_screenshot",
        "mcp__chrome-devtools__take_snapshot",
        "mcp__chrome-devtools__upload_file",
        "mcp__chrome-devtools__wait_for",
    },
    "context7_mcp": {
        "mcp__context7__resolve-library-id",
        "mcp__context7__get-library-docs",
    },
    "fetch_mcp": {
        "mcp__mcp-fetch__imageFetch",
    },
    "resource_mcp": {
        "ListMcpResourcesTool",
        "ReadMcpResourceTool",
    },
}

TOOL_NAME_TO_GROUP: Dict[str, str] = {
    tool_name: group_name
    for group_name, names in TOOL_GROUPS.items()
    for tool_name in names
}


def normalize_group_names(raw_groups: Iterable[str]) -> List[str]:
    """Normalize and deduplicate tool group names."""
    seen: Set[str] = set()
    normalized: List[str] = []
    for group in raw_groups:
        key = group.strip().lower()
        if not key:
            continue
        if key not in TOOL_GROUPS:
            continue
        if key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized


def disabled_tool_names_from_groups(disabled_groups: Iterable[str]) -> Set[str]:
    """Translate group names into the concrete tool names that should be removed."""
    names: Set[str] = set()
    for group in disabled_groups:
        if group in TOOL_GROUPS:
            names.update(TOOL_GROUPS[group])
    return names


def sanitize_tool_payload(
    openai_request: MutableMapping[str, object],
    disabled_groups: Iterable[str],
) -> None:
    """
    Remove tool metadata from an OpenAI request based on the disabled groups.

    This mutates the provided request in place. When all tools are removed,
    tool_choice/tool_calls/tool role messages are stripped to prevent upstream errors.
    """

    disabled_names = disabled_tool_names_from_groups(disabled_groups)
    if not disabled_names:
        return

    tools = openai_request.get("tools")
    if isinstance(tools, list):
        filtered_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                filtered_tools.append(tool)
                continue
            fn = tool.get(Constants.TOOL_FUNCTION, {})
            if not isinstance(fn, dict):
                filtered_tools.append(tool)
                continue
            name = fn.get("name")
            if name in disabled_names:
                continue
            filtered_tools.append(tool)
        if filtered_tools:
            openai_request["tools"] = filtered_tools
        else:
            openai_request.pop("tools", None)

    tool_choice = openai_request.get("tool_choice")
    if isinstance(tool_choice, dict):
        fn = tool_choice.get(Constants.TOOL_FUNCTION, {})
        if isinstance(fn, dict):
            name = fn.get("name")
            if name in disabled_names:
                openai_request["tool_choice"] = "auto"

    removed_tool_call_ids: Set[str] = set()
    messages = openai_request.get("messages")

    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue

            if (
                message.get("role") == Constants.ROLE_ASSISTANT
                and isinstance(message.get("tool_calls"), list)
            ):
                filtered_calls = []
                for call in message["tool_calls"]:
                    if not isinstance(call, dict):
                        filtered_calls.append(call)
                        continue
                    fn = call.get(Constants.TOOL_FUNCTION, {})
                    name = None
                    if isinstance(fn, dict):
                        name = fn.get("name")
                    if name in disabled_names:
                        if call.get("id"):
                            removed_tool_call_ids.add(call["id"])
                        continue
                    filtered_calls.append(call)
                if filtered_calls:
                    message["tool_calls"] = filtered_calls
                else:
                    message.pop("tool_calls", None)

        if removed_tool_call_ids:
            cleaned_messages: List[Dict[str, object]] = []
            for message in messages:
                if (
                    isinstance(message, dict)
                    and message.get("role") == Constants.ROLE_TOOL
                    and message.get("tool_call_id") in removed_tool_call_ids
                ):
                    continue
                cleaned_messages.append(message)
            openai_request["messages"] = cleaned_messages

    if not openai_request.get("tools"):
        openai_request.pop("tool_choice", None)
