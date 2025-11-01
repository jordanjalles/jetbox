"""
Server tools behavior - provides background server management.

Extracts server tools from tools.py into a composable AgentBehavior:
- start_server: Start background server process
- stop_server: Stop running server
- check_server: Check server status and logs
- list_servers: List all running servers

Features:
- Process management via orchestrator
- Server state tracking
- Log tailing
- Request/response mechanism via files
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from behaviors.base import AgentBehavior


class ServerToolsBehavior(AgentBehavior):
    """
    Provides server management tools: start_server, stop_server, check_server, list_servers.

    Servers are managed by the orchestrator via request/response files.
    """

    def __init__(
        self,
        workspace_manager=None,
        ledger_file: Path | None = None,
        **kwargs
    ):
        """
        Initialize ServerToolsBehavior.

        Args:
            workspace_manager: WorkspaceManager instance
            ledger_file: Path to ledger file for audit logging
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager
        self.ledger_file = ledger_file

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "server_tools"

    def get_tools(self) -> list[dict[str, Any]]:
        """Return server management tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "start_server",
                    "description": "Start a background server process (e.g., web server). Returns server info.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Command to run (e.g., ['python', '-m', 'http.server', '8000'])"
                            },
                            "name": {
                                "type": "string",
                                "description": "Optional server name (auto-generated if omitted)"
                            }
                        },
                        "required": ["cmd"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "stop_server",
                    "description": "Stop a running background server.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "server_id": {
                                "type": "string",
                                "description": "Server identifier (from start_server or list_servers)"
                            }
                        },
                        "required": ["server_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_server",
                    "description": "Check server status and get recent logs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "server_id": {
                                "type": "string",
                                "description": "Server identifier"
                            },
                            "tail_lines": {
                                "type": "integer",
                                "description": "Number of recent log lines (default 20)"
                            }
                        },
                        "required": ["server_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_servers",
                    "description": "List all running background servers.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Dispatch server management tool calls.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: Additional context (workspace_manager, ledger_file)

        Returns:
            Server operation result dict
        """
        # Allow runtime override
        workspace_manager = kwargs.get('workspace_manager', self.workspace_manager)
        ledger_file = kwargs.get('ledger_file', self.ledger_file)

        if tool_name == "start_server":
            return self._start_server(
                args.get("cmd"),
                name=args.get("name"),
                workspace_manager=workspace_manager,
                ledger_file=ledger_file
            )
        elif tool_name == "stop_server":
            return self._stop_server(
                args.get("server_id"),
                ledger_file=ledger_file
            )
        elif tool_name == "check_server":
            return self._check_server(
                args.get("server_id"),
                tail_lines=args.get("tail_lines", 20)
            )
        elif tool_name == "list_servers":
            return self._list_servers()
        else:
            return super().dispatch_tool(tool_name, args, **kwargs)

    def _ledger_append(self, kind: str, detail: str, ledger_file: Path | None) -> None:
        """Append action to ledger file for audit trail."""
        if not ledger_file:
            return
        line = f"{kind}\t{detail.replace(chr(10), ' ')[:400]}\n"
        if ledger_file.exists():
            ledger_file.write_text(
                ledger_file.read_text(encoding="utf-8") + line,
                encoding="utf-8"
            )
        else:
            ledger_file.write_text(line, encoding="utf-8")

    def _start_server(
        self,
        cmd: list[str],
        name: str | None = None,
        workspace_manager=None,
        ledger_file: Path | None = None
    ) -> dict[str, Any]:
        """Request orchestrator to start a server in the background."""
        import os

        # Validate command
        if not cmd:
            return {"error": "Command cannot be empty"}

        # Generate server ID
        server_id = name or f"server_{int(time.time())}"

        # Set up paths
        cwd = str(workspace_manager.workspace_dir) if workspace_manager else os.getcwd()
        log_file = os.path.join(cwd, f"{server_id}.log")

        # Write request
        request_file = Path(".agent_context/server_requests.jsonl")
        request_file.parent.mkdir(parents=True, exist_ok=True)

        request = {
            "action": "start",
            "server_id": server_id,
            "cmd": cmd,
            "cwd": cwd,
            "log_file": log_file,
        }

        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        # Poll for response (wait up to 5 seconds)
        response = self._wait_for_server_response(timeout=5.0)

        if response:
            self._ledger_append("SERVER", f"start {server_id} -> {response.get('success', False)}", ledger_file)
            return response
        else:
            return {"error": "Timeout waiting for orchestrator to start server"}

    def _stop_server(
        self,
        server_id: str,
        ledger_file: Path | None = None
    ) -> dict[str, Any]:
        """Request orchestrator to stop a server."""
        request = {"action": "stop", "server_id": server_id}

        request_file = Path(".agent_context/server_requests.jsonl")
        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        response = self._wait_for_server_response(timeout=5.0)

        if response:
            self._ledger_append("SERVER", f"stop {server_id} -> {response.get('success', False)}", ledger_file)

        return response or {"error": "Timeout waiting for response"}

    def _check_server(
        self,
        server_id: str,
        tail_lines: int = 20
    ) -> dict[str, Any]:
        """Check server status and get recent logs."""
        request = {"action": "check", "server_id": server_id, "tail_lines": tail_lines}

        request_file = Path(".agent_context/server_requests.jsonl")
        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        response = self._wait_for_server_response(timeout=5.0)
        return response or {"error": "Timeout waiting for response"}

    def _list_servers(self) -> dict[str, Any]:
        """List all running servers."""
        request = {"action": "list"}

        request_file = Path(".agent_context/server_requests.jsonl")
        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        response = self._wait_for_server_response(timeout=5.0)
        return response or {"error": "Timeout waiting for response"}

    def _wait_for_server_response(self, timeout: float = 5.0) -> dict[str, Any] | None:
        """
        Wait for orchestrator response to server request.

        Polls response file for new line matching our request.

        Args:
            timeout: Max seconds to wait (default 5.0)

        Returns:
            Response dict or None on timeout
        """
        response_file = Path(".agent_context/server_responses.jsonl")

        # Count existing lines to know where to start reading
        existing_lines = 0
        if response_file.exists():
            with open(response_file, 'r') as f:
                existing_lines = len(f.readlines())

        start_time = time.time()

        while time.time() - start_time < timeout:
            if response_file.exists():
                with open(response_file, 'r') as f:
                    lines = f.readlines()

                # Check for new lines
                if len(lines) > existing_lines:
                    # Return the newest response
                    response_line = lines[-1].strip()
                    if response_line:
                        return json.loads(response_line)

            time.sleep(0.1)

        return None
