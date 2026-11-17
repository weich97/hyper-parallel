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
"""Local HTTP and pinned CLI request fixtures shared by gateway contract tests."""

import json
from typing import Any
import urllib.error
import urllib.request


def gateway_http(gateway: Any, method: str, path: str, token: str | None = None,
          body: bytes | None = None) -> tuple[int, dict]:
    """Use real local HTTP, retaining error status and decoded response bodies."""
    headers = {} if token is None else {"Authorization": "Bearer " + token}
    request = urllib.request.Request(
        f"http://{gateway.address[0]}:{gateway.address[1]}{path}",
        method=method, data=body, headers=headers,
    )
    try:
        response = urllib.request.urlopen(request, timeout=3)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.loads(response.read())


def repository_request(prompt: str = "question") -> dict[str, Any]:
    """Model the two context blocks emitted by the pinned candidate Codex CLI."""
    return {
        "instructions": "Repair the repository and run a focused test.",
        "client_metadata": {"x-codex-turn-metadata": json.dumps({"request_kind": "turn"})},
        "tools": [{"type": "function", "name": "exec_command", "description": "Run a shell command.",
                   "parameters": {"type": "object", "properties": {
                       "cmd": {"type": "string"}, "sandbox_permissions": {"type": "string"},
                       "justification": {"type": "string"}, "prefix_rule": {"type": "array"}},
                       "required": ["cmd"], "additionalProperties": False}},
                  {"type": "function", "name": "write_stdin", "description": "Write to a running session.",
                   "parameters": {"type": "object", "properties": {"session_id": {"type": "number"}},
                                  "required": ["session_id"], "additionalProperties": False}}],
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": (
                "<permissions instructions>\n"
                "Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is "
                "`danger-full-access`: No filesystem sandboxing - all commands are permitted. "
                "Network access is enabled.\n"
                "Approval policy is currently never. Do not provide the `sandbox_permissions` for any reason, "
                "commands will be rejected.\n"
                "</permissions instructions>"
            )}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": (
                "<environment_context>\n"
                "  <cwd>/workspace</cwd>\n"
                "  <shell>bash</shell>\n"
                "  <permission_profile type=\"disabled\"><file_system type=\"unrestricted\" />"
                "</permission_profile>\n"
                "</environment_context>"
            )}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
    }
