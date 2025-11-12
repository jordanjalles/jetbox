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
from behaviors.rule_of_two_types import RuleOfTwoProperty


class CommandToolsBehavior(AgentBehavior):
    """
    Provides bash command execution tool: run_bash.

    Full shell access for flexible command execution in the workspace.

    Security: DYNAMIC based on whitelist contents and workspace config
    - [A] UNTRUSTED_INPUT: If whitelist contains file-reading commands AND workspace has untrusted files
    - [B] SENSITIVE_ACCESS: If whitelist has non-safe commands AND workspace has sensitive files
    - [C] EXTERNAL_ACTION: If whitelist contains network commands AND workspace has network access
    """

    # Rule of Two: Empty static fallback (dynamically computed at runtime)
    rule_of_two_properties = set()

    # Command categories for dynamic classification
    FILE_READING_COMMANDS = {
        'cat', 'head', 'tail', 'less', 'more',
        'grep', 'rg', 'ag', 'ack', 'egrep', 'fgrep',
        'find', 'diff', 'cmp', 'wc', 'sort', 'uniq'
    }

    CODE_EXECUTION_COMMANDS = {
        'python', 'python3', 'python2',
        'node', 'nodejs',
        'ruby', 'perl', 'php',
        'bash', 'sh', 'zsh', 'fish',
        'java', 'javac',
        'gcc', 'g++', 'clang',
        'cargo', 'rustc',
        'go', 'run'
    }

    NETWORK_COMMANDS = {
        'curl', 'wget', 'http', 'httpie',
        'ping', 'traceroute', 'nslookup', 'dig',
        'ssh', 'scp', 'rsync', 'sftp',
        'npm', 'yarn', 'pip', 'pipenv',  # Package managers can fetch from network
        'git'  # git push/pull/fetch/clone communicate over network
    }

    SAFE_COMMANDS = {
        'echo', 'date', 'pwd', 'whoami', 'sleep', 'true', 'false'
    }

    def __init__(
        self,
        workspace_manager=None,
        ledger_file: Path | None = None,
        whitelist: list[str] | None = None,
        **kwargs
    ):
        """
        Initialize CommandToolsBehavior.

        Args:
            workspace_manager: WorkspaceManager instance for workspace directory
            ledger_file: Path to ledger file for audit logging
            whitelist: List of allowed commands (overrides jetbox_commands_whitelist file)
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager
        self.ledger_file = ledger_file

        # Load whitelist from jetbox_commands_whitelist file or use parameter
        # Convert to set if provided as list
        if whitelist is not None:
            self.whitelist = set(whitelist) if isinstance(whitelist, list) else whitelist
        else:
            self.whitelist = self._load_whitelist_from_file()

    def _load_whitelist_from_file(self) -> set[str] | None:
        """
        Load command whitelist from jetbox_commands_whitelist file.

        Returns:
            Set of allowed commands, or None if no whitelist enforced
        """
        from agent_config import PROJECT_ROOT
        whitelist_file = PROJECT_ROOT / "jetbox_commands_whitelist"

        if not whitelist_file.exists():
            return None

        try:
            lines = whitelist_file.read_text(encoding="utf-8").splitlines()
            # Filter out comments and empty lines
            commands = {
                line.strip()
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            }
            return commands if commands else None
        except Exception as e:
            print(f"[command_tools] Warning: Could not load whitelist from {whitelist_file}: {e}")
            return None

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "command_tools"

    def get_rule_of_two_properties(self, agent, security_context):
        """
        Get Rule of Two properties (context-aware and whitelist-aware).

        Dynamic behavior based on whitelist contents AND workspace config:
        - [A] UNTRUSTED_INPUT: If whitelist has file-reading commands AND workspace has untrusted files
        - [B] SENSITIVE_ACCESS: If whitelist has non-safe commands AND workspace has sensitive files
        - [C] EXTERNAL_ACTION: If whitelist has network commands AND workspace has network access

        Args:
            agent: Agent instance
            security_context: SecurityContext with workspace characteristics

        Returns:
            Set of properties based on whitelist and workspace config
        """
        props = set()

        # If no whitelist, assume all commands possible (maximum properties)
        if self.whitelist is None:
            # No whitelist = full shell access
            if security_context:
                if security_context.workspace_has_untrusted_files:
                    props.add(RuleOfTwoProperty.UNTRUSTED_INPUT)
                if security_context.workspace_has_sensitive_files:
                    props.add(RuleOfTwoProperty.SENSITIVE_ACCESS)
                if security_context.workspace_has_network_access:
                    props.add(RuleOfTwoProperty.EXTERNAL_ACTION)
            return props

        # Check if whitelist can access untrusted files (read or execute)
        has_file_access = bool(self.whitelist & (self.FILE_READING_COMMANDS | self.CODE_EXECUTION_COMMANDS))
        if has_file_access and security_context and security_context.workspace_has_untrusted_files:
            props.add(RuleOfTwoProperty.UNTRUSTED_INPUT)

        # Check if whitelist contains network commands
        has_network = bool(self.whitelist & self.NETWORK_COMMANDS)
        if has_network and security_context and security_context.workspace_has_network_access:
            props.add(RuleOfTwoProperty.EXTERNAL_ACTION)

        # [B] SENSITIVE_ACCESS: Only if whitelist has non-safe commands AND workspace has sensitive files
        # If whitelist is exclusively safe commands (echo, date, pwd), no sensitive access
        # If workspace has no sensitive files (.env, credentials, keys), no sensitive access
        is_only_safe = self.whitelist.issubset(self.SAFE_COMMANDS)
        if not is_only_safe and security_context and security_context.workspace_has_sensitive_files:
            props.add(RuleOfTwoProperty.SENSITIVE_ACCESS)

        return props

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject tool documentation for command tools.

        Called once during agent initialization to document available tools.

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Context with tool documentation injected
        """
        tools = self.get_tools()
        if not tools:
            return context

        # Build tool documentation
        tool_docs = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            # Build parameter signature
            param_strs = []
            for param_name, param_spec in params.items():
                param_type = param_spec.get("type", "any")
                is_required = param_name in required
                if is_required:
                    param_strs.append(f"{param_name}: {param_type}")
                else:
                    param_strs.append(f"{param_name}?: {param_type}")

            param_sig = ", ".join(param_strs) if param_strs else ""
            tool_docs.append(f"  - {name}({param_sig}): {desc}")

        if tool_docs:
            tool_message = f"\n{self.get_name()} tools:\n" + "\n".join(tool_docs)
            return self.inject_user_message_after_system(context, tool_message)

        return context

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
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Dispatch bash command tool calls.

        Args:
            agent: Agent instance
            tool_name: Tool being called
            args: Tool arguments

        Returns:
            Dict with returncode, stdout, stderr
        """
        # Get workspace_manager and ledger_file from agent
        workspace_manager = getattr(agent, 'workspace_manager', self.workspace_manager)
        ledger_file = getattr(agent, 'ledger_file', self.ledger_file)

        if tool_name == "run_bash":
            return self._run_bash(
                args.get("command"),
                timeout=args.get("timeout", 60),
                workspace_manager=workspace_manager,
                ledger_file=ledger_file
            )
        else:
            return super().dispatch_tool(agent, tool_name, args)

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
        # Check whitelist if configured
        if self.whitelist:
            # Extract first token (command name)
            first_token = command.strip().split()[0] if command.strip() else ""

            if first_token not in self.whitelist:
                error_msg = (
                    f"Command not allowed: '{first_token}'. "
                    f"Only these commands are whitelisted: {sorted(self.whitelist)}"
                )
                self._ledger_append("ERROR", f"Blocked command: {command[:100]}", ledger_file)
                return {
                    "error": error_msg,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": error_msg
                }

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
