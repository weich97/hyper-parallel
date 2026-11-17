# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Pinned repository CLI context, compaction and executable tool contracts."""

from __future__ import annotations

import json
from typing import Any


_CLI_PERMISSIONS = (
    "<permissions instructions>\n"
    "Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is "
    "`danger-full-access`: No filesystem sandboxing - all commands are permitted. Network access is enabled.\n"
    "Approval policy is currently never. Do not provide the `sandbox_permissions` for any reason, "
    "commands will be rejected.\n"
    "</permissions instructions>"
)
_REPOSITORY_PERMISSIONS = (
    "<permissions instructions>\n"
    "This task runs in an isolated candidate Docker container with network=none. "
    "No Docker socket or NPU device is available. Put submitted source edits only in /workspace "
    "and obey the task's narrower file allowlist. Use /tmp only for temporary files. "
    "No permission escalation is available. The controller freezes the final artifact, and a separate "
    "trusted grader checks its file allowlist and evaluates the patch.\n"
    "</permissions instructions>"
)
_CLI_PERMISSION_PROFILE = '<permission_profile type="disabled"><file_system type="unrestricted" />' \
                          '</permission_profile>'
_REPOSITORY_PERMISSION_PROFILE = (
    '<permission_profile type="candidate-container"><file_system type="container-root">'
    '<source_edit_scope>/workspace</source_edit_scope><scratch>/tmp</scratch></file_system>'
    '<network>none</network><docker_socket>unavailable</docker_socket><npu>unavailable</npu>'
    '</permission_profile>'
)
ISOLATED_COMPACT_PROMPT = (
    "HYPER_RL_CODE_AGENT_COMPACTION_V2\n"
    "Write only a compact handoff of at most 160 words in at most six bullets. Do not call tools. "
    "Keep the repository issue and edit constraints, exact relevant source paths and function names, "
    "current verified edits, latest test results, unresolved blockers, and the next concrete action. "
    "Distinguish failed commands and unchanged files from successful edits. "
    "Do not quote source code, patches, command output, or the chronological transcript. "
    "Summarize the current state, not every previous attempt. Finish within this word limit."
)


def _isolated_request_kind(original: dict[str, Any]) -> str:
    """Use the pinned CLI's explicit request kind, never infer it from tools alone."""
    metadata = original.get("client_metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("x-codex-turn-metadata"), str):
        raise ValueError("Repository Codex request omitted pinned CLI metadata")
    try:
        kind = json.loads(metadata["x-codex-turn-metadata"])["request_kind"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Repository Codex request has invalid pinned CLI metadata") from error
    if kind not in {"turn", "compaction"}:
        raise ValueError("Repository Codex request has unsupported pinned CLI kind")
    return kind


def prepare_isolated_request(original: dict[str, Any], transformed: dict[str, Any]) -> str:
    """Validate CLI context and adapt only the request forwarded to the model.

    Args:
        original: Unmodified Responses request retained as episode evidence.
        transformed: Chat request updated in place with the real container limits.

    Returns:
        Validated CLI request kind, either turn or compaction.

    Raises:
        ValueError: CLI context or tool schema differs from the pinned contract.
    """
    request_kind = _isolated_request_kind(original)
    input_items, environment_text = _isolated_context(original)
    messages = transformed["messages"]
    _rewrite_context(messages, environment_text)
    _merge_assistant_messages(messages)
    _prepare_request_tools(original, transformed, input_items, request_kind)
    return request_kind


def _validate_context_message(item: Any, role: str) -> None:
    """Require the pinned message envelope before inspecting its content."""
    if (not isinstance(item, dict) or item.get("type") != "message"
            or item.get("role") != role or not isinstance(item.get("content"), list)):
        label = "developer" if role == "developer" else "environment"
        raise ValueError(f"Repository Codex {label} context differs from pinned CLI format")


def _isolated_context(original: dict[str, Any]) -> tuple[list, str]:
    """Read the two fixed context items without changing episode evidence."""
    input_items = original.get("input")
    if not isinstance(input_items, list) or len(input_items) < 2:
        raise ValueError("Repository Codex request omitted pinned CLI context items")
    developer, environment = input_items[:2]
    _validate_context_message(developer, "developer")
    _validate_context_message(environment, "user")
    _validate_permission_block(developer)
    return input_items, _environment_text(environment)


def _validate_permission_block(developer: dict[str, Any]) -> None:
    """Reject altered permission text before forwarding container-specific limits."""
    permission_blocks = [block for block in developer["content"] if isinstance(block, dict)
                         and isinstance(block.get("text"), str)
                         and ("<permissions instructions>" in block["text"]
                              or "</permissions instructions>" in block["text"])]
    if (len(permission_blocks) != 1 or permission_blocks[0].get("type") != "input_text"
            or permission_blocks[0]["text"] != _CLI_PERMISSIONS):
        raise ValueError("Repository Codex permissions block differs from pinned CLI format")


def _environment_text(environment: dict[str, Any]) -> str:
    """Require the exact environment wrapper and disabled CLI permission profile."""
    environment_blocks = [block for block in environment["content"]
                          if isinstance(block, dict) and block.get("type") == "input_text"]
    if len(environment_blocks) != 1 or not isinstance(environment_blocks[0].get("text"), str):
        raise ValueError("Repository Codex environment block differs from pinned CLI format")
    environment_text = environment_blocks[0]["text"]
    if (not environment_text.startswith("<environment_context>\n")
            or not environment_text.endswith("</environment_context>")
            or environment_text.count("<environment_context>") != 1
            or environment_text.count("</environment_context>") != 1
            or environment_text.count("<permission_profile") != 1
            or environment_text.count("</permission_profile>") != 1
            or environment_text.count("<cwd>/workspace</cwd>") != 1
            or environment_text.count(_CLI_PERMISSION_PROFILE) != 1):
        raise ValueError("Repository Codex environment block differs from pinned CLI format")
    return environment_text


def _rewrite_context(messages: list[dict], environment_text: str) -> None:
    """Replace only the validated permission and environment context."""
    if (not messages or messages[0].get("role") != "system"
            or messages[0]["content"].count(_CLI_PERMISSIONS) != 1):
        raise ValueError("Repository Codex permissions could not be located after translation")
    messages[0]["content"] = messages[0]["content"].replace(_CLI_PERMISSIONS, _REPOSITORY_PERMISSIONS)
    environment_messages = [message for message in messages
                            if message.get("role") == "user" and message.get("content") == environment_text]
    if len(environment_messages) != 1:
        raise ValueError("Repository Codex environment could not be located after translation")
    environment_messages[0]["content"] = environment_text.replace(
        _CLI_PERMISSION_PROFILE, _REPOSITORY_PERMISSION_PROFILE
    )


def _merge_assistant_messages(messages: list[dict]) -> None:
    """Keep commentary and tool items from one turn in one Chat message."""
    # Responses separates commentary and function calls into items from the same
    # assistant turn. A second Chat message would add a spurious end-of-turn token.
    for index in range(len(messages) - 1, 0, -1):
        previous, current = messages[index - 1], messages[index]
        if (previous.get("role") == current.get("role") == "assistant"
                and current.get("tool_calls") and not previous.get("tool_calls")):
            previous["content"] = (previous.get("content") or "") + (current.get("content") or "")
            previous["tool_calls"] = current["tool_calls"]
            if current.get("reasoning"):
                previous["reasoning"] = (previous.get("reasoning") or "") + current["reasoning"]
            del messages[index]


def _prepare_request_tools(original: dict, transformed: dict, input_items: list, request_kind: str) -> None:
    """Validate compaction or the fixed pair of executable tools."""
    tools = transformed.get("tools")
    if request_kind == "compaction":
        last = input_items[-1]
        if (original.get("tools") != [] or tools not in (None, [])
                or not isinstance(last, dict) or last.get("type") != "message"
                or last.get("role") != "user" or last.get("content") != [
                    {"type": "input_text", "text": ISOLATED_COMPACT_PROMPT}]):
            raise ValueError("Repository Codex compaction request differs from pinned CLI contract")
        return
    if (not isinstance(tools, list) or len(tools) != 2
            or {tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)}
            != {"exec_command", "write_stdin"}):
        raise ValueError("Repository Codex tool set differs from the two executable CLI tools")
    for tool in tools:
        function = tool.get("function", {})
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        if function.get("name") == "exec_command":
            if not isinstance(properties, dict):
                raise ValueError("Repository Codex exec_command schema has no properties")
            hidden = {"sandbox_permissions", "justification", "prefix_rule"}
            function["parameters"] = {
                **parameters,
                "properties": {name: value for name, value in properties.items() if name not in hidden},
            }
        _validate_tool_schema(function)


def _validate_tool_schema(function: dict[str, Any]) -> None:
    """Require the pinned CLI's simple closed argument objects before sampling."""
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError("Repository Codex tool parameters must be an object schema")
    properties = parameters.get("properties")
    required = parameters.get("required")
    if (not isinstance(properties, dict) or parameters.get("additionalProperties") is not False
            or not isinstance(required, list) or not all(isinstance(name, str) for name in required)
            or not set(required).issubset(properties)):
        raise ValueError("Repository Codex tool schema differs from pinned closed argument contract")
    if any(not isinstance(value, dict) or value.get("type") not in {"string", "number", "integer", "boolean"}
           for value in properties.values()):
        raise ValueError("Repository Codex tool schema has unsupported argument types")


def tool_argument_failure(response: dict[str, Any], transformed: dict[str, Any]) -> dict | None:
    """Classify parser-verified calls that the forwarded CLI schema cannot execute.

    Args:
        response: Model response whose parser evidence has already been validated.
        transformed: Forwarded Chat request containing the closed tool schemas.

    Returns:
        Trainable model-failure evidence, or None when arguments match the schema.
    """
    tools = {tool["function"]["name"]: tool["function"]["parameters"] for tool in transformed["tools"]}
    calls = response["choices"][0]["message"].get("tool_calls") or []
    for call in calls:
        function = call["function"]
        name = function["name"]
        if name not in tools:
            return {"failure_origin": "model", "failure_reason": "undeclared_tool_name", "trainable": True,
                    "tool_feedback": f"Tool {name!r} is not declared. Nothing was executed. "
                                     "Use exec_command or write_stdin with their declared arguments."}
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            return {"failure_origin": "model", "failure_reason": "invalid_tool_arguments", "trainable": True,
                    "tool_feedback": f"Tool {name!r} requires a JSON argument object. Nothing was executed."}
        schema = tools[name]
        properties = schema["properties"]
        extra = sorted(set(arguments) - set(properties))
        if extra:
            detail = ", ".join(extra)
            feedback = f"Tool {name!r} does not accept argument(s) {detail}. Nothing was executed."
            if name == "exec_command" and "stdin" in extra:
                feedback += " Put input in the cmd shell command using a heredoc, without a separate stdin key."
            return {"failure_origin": "model", "failure_reason": "undeclared_tool_argument", "trainable": True,
                    "tool_feedback": feedback, "tool_name": name, "invalid_arguments": extra}
        missing = sorted(set(schema["required"]) - set(arguments))
        if missing:
            return {"failure_origin": "model", "failure_reason": "missing_tool_argument", "trainable": True,
                    "tool_feedback": f"Tool {name!r} requires argument(s) {', '.join(missing)}. Nothing was executed."}
        for key, value in arguments.items():
            expected = properties[key]["type"]
            if expected == "string":
                valid = isinstance(value, str)
            elif expected == "boolean":
                valid = isinstance(value, bool)
            elif expected == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not valid:
                return {"failure_origin": "model", "failure_reason": "invalid_tool_argument_type",
                        "trainable": True, "tool_feedback": f"Tool {name!r} requires {key!r} to be {expected}. "
                                                          "Nothing was executed."}
    return None
