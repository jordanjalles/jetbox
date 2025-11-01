"""
Command tools behavior - provides bash command execution.

Extracts command tool from tools.py into a composable AgentBehavior:
- run_bash: Execute shell commands with output capture

Features:
- Full shell access with pipes, redirection, command chaining
- Workspace-aware execution (runs in workspace directory)
- Output capture with size limits
- Timeout support
- Ledger logging for audit trail
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from behaviors.base import AgentBehavior


class CommandToolsBehavior(AgentBehavior):
    """
    Provides bash command execution tool: run_bash.

    Full shell access for flexible command execution in the workspace.
    """

    def __init__(
        self,
        workspace_manager=None,
        ledger_file: Path | None = None,
        **kwargs
    ):
        """
        Initialize CommandToolsBehavior.

        Args:
            workspace_manager: WorkspaceManager instance for workspace directory
            ledger_file: Path to ledger file for audit logging
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager
        self.ledger_file = ledger_file

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "command_tools"

    def get_tools(self) -> list[dict[str, Any]]:
        """Return bash command tool definition."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_bash",
                    "description": "Run any bash command with full shell features. Use for testing, linting, complex file operations, searching, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Full bash command string (e.g., 'pytest tests/ -v', 'grep -r pattern *.py')"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default 60). Use higher values for slow operations."
                            }
                        },
                        "required": ["command"]
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
        Dispatch bash command tool calls.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: Additional context (workspace_manager, ledger_file)

        Returns:
            Dict with returncode, stdout, stderr
        """
        # Allow runtime override of workspace_manager and ledger_file
        workspace_manager = kwargs.get('workspace_manager', self.workspace_manager)
        ledger_file = kwargs.get('ledger_file', self.ledger_file)

        if tool_name == "run_bash":
            return self._run_bash(
                args.get("command"),
                timeout=args.get("timeout", 60),
                workspace_manager=workspace_manager,
                ledger_file=ledger_file
            )
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

    def _run_bash(
        self,
        command: str,
        timeout: int = 60,
        workspace_manager=None,
        ledger_file: Path | None = None
    ) -> dict[str, Any]:
        """
        Run any bash command in the workspace.

        Full shell access with pipes, redirection, and command chaining.
        Use this for flexible file operations, testing, linting, etc.

        Args:
            command: Full bash command string (e.g., "grep -r 'pattern' *.py | wc -l")
            timeout: Timeout in seconds (default 60)
            workspace_manager: WorkspaceManager instance
            ledger_file: Ledger file path

        Returns:
            Dict with returncode, stdout, stderr

        Examples:
            run_bash("python script.py")
            run_bash("pytest tests/ -v")
            run_bash("grep -A 3 'class' file.py")
            run_bash("find . -name '*.py' | xargs wc -l")
            run_bash("cat file1.txt file2.txt > combined.txt")
        """
        # Determine working directory
        cwd = str(workspace_manager.workspace_dir) if workspace_manager else None

        # Set up environment with PYTHONPATH for workspace
        env = os.environ.copy()
        if workspace_manager and cwd:
            env["PYTHONPATH"] = cwd

        try:
            p = subprocess.run(
                command,
                shell=True,  # Enable full shell features
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env
            )
            out = {
                "returncode": p.returncode,
                "stdout": p.stdout[-50_000:],  # Truncate to last 50KB
                "stderr": p.stderr[-50_000:],
            }
            self._ledger_append("BASH", f"{command[:100]} -> rc={p.returncode}", ledger_file)
            if p.returncode != 0:
                self._ledger_append("ERROR", f"run_bash rc={p.returncode}: {p.stderr[:200]}", ledger_file)
            return out
        except subprocess.TimeoutExpired:
            err = f"Command timed out after {timeout}s"
            self._ledger_append("ERROR", f"run_bash timeout: {command[:100]}", ledger_file)
            return {"error": err, "returncode": -1, "stdout": "", "stderr": err}
        except Exception as e:
            err_msg = str(e)
            self._ledger_append("ERROR", f"run_bash exception: {err_msg}", ledger_file)
            return {"error": err_msg, "returncode": -1, "stdout": "", "stderr": err_msg}
