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

Now uses @tool decorator for automatic tool registration!
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty
from behaviors.tool_decorator import tool


class ServerToolsBehavior(AgentBehavior):
    """
    Provides server management tools: start_server, stop_server, check_server, list_servers.

    Servers are managed by the orchestrator via request/response files.

    Security: DYNAMIC based on workspace and network access
    - [B] SENSITIVE_ACCESS: Only if workspace has sensitive files (servers can access them)
    - [C] EXTERNAL_ACTION: Only if network enabled (servers can bind to public interfaces)
    """

    # Rule of Two: Empty static fallback (dynamically computed at runtime)
    rule_of_two_properties = set()

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

    def get_rule_of_two_properties(self, agent, security_context):
        """
        Get Rule of Two properties (context-aware).

        Dynamic behavior based on workspace and network access:
        - [B] SENSITIVE_ACCESS: Only if workspace has sensitive files (servers can access them)
        - [C] EXTERNAL_ACTION: Only if network enabled (servers can bind to public interfaces)

        Args:
            agent: Agent instance
            security_context: SecurityContext with workspace characteristics

        Returns:
            Set of properties for current context ([], [B], [C], or [BC])
        """
        props = set()

        # [B] SENSITIVE_ACCESS - only if workspace has sensitive files
        if security_context and security_context.workspace_has_sensitive_files:
            props.add(RuleOfTwoProperty.SENSITIVE_ACCESS)

        # [C] EXTERNAL_ACTION - only if workspace has network access
        if security_context and security_context.workspace_has_network_access:
            props.add(RuleOfTwoProperty.EXTERNAL_ACTION)

        return props

    # Tool implementations (using @tool decorator)

    @tool
    def start_server(
        self,
        cmd: list[str],
        name: str | None = None
    ) -> dict[str, Any]:
        """Start a background server process (e.g., web server). Returns server info.

        Args:
            cmd: Command to run (e.g., ['python', '-m', 'http.server', '8000'])
            name: Optional server name (auto-generated if omitted)

        Returns:
            Server info dict with server_id and status
        """
        # Access agent via self.agent (injected by decorator)
        workspace_manager = getattr(self.agent, 'workspace_manager', self.workspace_manager)
        ledger_file = getattr(self.agent, 'ledger_file', self.ledger_file)

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

    @tool
    def stop_server(
        self,
        server_id: str
    ) -> dict[str, Any]:
        """Stop a running background server.

        Args:
            server_id: Server identifier (from start_server or list_servers)

        Returns:
            Status dict with success flag
        """
        # Access agent via self.agent (injected by decorator)
        ledger_file = getattr(self.agent, 'ledger_file', self.ledger_file)
        request = {"action": "stop", "server_id": server_id}

        request_file = Path(".agent_context/server_requests.jsonl")
        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        response = self._wait_for_server_response(timeout=5.0)

        if response:
            self._ledger_append("SERVER", f"stop {server_id} -> {response.get('success', False)}", ledger_file)

        return response or {"error": "Timeout waiting for response"}

    @tool
    def check_server(
        self,
        server_id: str,
        tail_lines: int = 20
    ) -> dict[str, Any]:
        """Check server status and get recent logs.

        Args:
            server_id: Server identifier
            tail_lines: Number of recent log lines (default 20)

        Returns:
            Dict with server status and log output
        """
        request = {"action": "check", "server_id": server_id, "tail_lines": tail_lines}

        request_file = Path(".agent_context/server_requests.jsonl")
        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        response = self._wait_for_server_response(timeout=5.0)
        return response or {"error": "Timeout waiting for response"}

    @tool
    def list_servers(self) -> dict[str, Any]:
        """List all running background servers.

        Returns:
            Dict with list of server info dicts
        """
        request = {"action": "list"}

        request_file = Path(".agent_context/server_requests.jsonl")
        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        response = self._wait_for_server_response(timeout=5.0)
        return response or {"error": "Timeout waiting for response"}

    # Helper methods

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
