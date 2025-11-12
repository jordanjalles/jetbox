"""
Read file tools behavior - provides file reading operations.

Extracts read_file tool from file_tools.py into a composable AgentBehavior:
- read_file: Read file contents with size limits and encoding handling

Features:
- Workspace-aware path resolution (uses WorkspaceManager)
- Size limits with truncation warnings
- Encoding error handling (error replacement)
- Parameter invention tolerance (**kwargs)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty


class ReadFileToolsBehavior(AgentBehavior):
    """
    Provides file reading tool: read_file.

    Workspace-aware file reading with size limits and encoding handling.

    Security: DYNAMIC based on workspace characteristics
    - [A] if workspace_has_untrusted_files (external data, uploads, scraping)
    - [B] if workspace_has_sensitive_files (.env, credentials, keys)
    - [] for trusted codebases with no secrets (Jetbox repo, internal tools)
    - Does not write or execute (no [C])
    """

    # Rule of Two: Empty static fallback (dynamically computed from workspace config)
    rule_of_two_properties = set()

    def __init__(
        self,
        workspace_manager=None,
        **kwargs
    ):
        """
        Initialize ReadFileToolsBehavior.

        Args:
            workspace_manager: WorkspaceManager instance for path resolution
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "read_file_tools"

    def get_rule_of_two_properties(self, agent, security_context):
        """
        Get Rule of Two properties (context-aware).

        Dynamic behavior based on workspace characteristics:
        - [A] UNTRUSTED_INPUT: Only if workspace has untrusted files (external data, uploads, etc.)
        - [B] SENSITIVE_ACCESS: Only if workspace has sensitive files (.env, credentials, keys)
        - Current Jetbox repo: [] (trusted codebase, no secrets)

        Args:
            agent: Agent instance
            security_context: SecurityContext with workspace characteristics

        Returns:
            Set of properties for current context (could be [], [A], [B], or [AB])
        """
        props = set()

        # [A] UNTRUSTED_INPUT - only if workspace contains untrusted files
        if security_context and security_context.workspace_has_untrusted_files:
            props.add(RuleOfTwoProperty.UNTRUSTED_INPUT)

        # [B] SENSITIVE_ACCESS - only if workspace contains sensitive files
        if security_context and security_context.workspace_has_sensitive_files:
            props.add(RuleOfTwoProperty.SENSITIVE_ACCESS)

        return props

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject tool documentation for read file tools.

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
        """Return file reading tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a text file (up to 1MB by default). For large files, adjust max_size or use run_bash with head/tail.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path (relative to workspace)"
                            },
                            "encoding": {
                                "type": "string",
                                "description": "Text encoding (default: utf-8)"
                            },
                            "max_size": {
                                "type": "integer",
                                "description": "Maximum bytes to read (default: 1000000)"
                            }
                        },
                        "required": ["path"]
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> str:
        """
        Dispatch file reading tool calls.

        Args:
            agent: Agent instance
            tool_name: Tool being called
            args: Tool arguments

        Returns:
            Tool result (file contents string)
        """
        # Get workspace_manager from agent
        workspace_manager = getattr(agent, 'workspace_manager', self.workspace_manager)

        if tool_name == "read_file":
            return self._read_file(
                args.get("path"),
                encoding=args.get("encoding", "utf-8"),
                max_size=args.get("max_size", 1_000_000),
                workspace_manager=workspace_manager,
                extra_kwargs=args
            )
        else:
            return super().dispatch_tool(agent, tool_name, args)

    def _read_file(
        self,
        path: str,
        encoding: str = "utf-8",
        max_size: int = 1_000_000,
        workspace_manager=None,
        extra_kwargs: dict | None = None
    ) -> str:
        """
        Read a text file (workspace-aware).

        Args:
            path: File path (relative to workspace if set)
            encoding: Text encoding
            max_size: Maximum bytes to read
            workspace_manager: WorkspaceManager instance
            extra_kwargs: Full args dict for parameter invention detection

        Returns:
            File contents (up to max_size, truncated if larger)
        """
        # Warn about unsupported parameters
        if extra_kwargs:
            supported = {"path", "encoding", "max_size"}
            unsupported = set(extra_kwargs.keys()) - supported
            if unsupported:
                ignored = ", ".join(unsupported)
                print(f"[read_file_tools] read_file ignoring unsupported parameters: {ignored}")

        # Resolve path through workspace if available
        if workspace_manager:
            resolved_path = workspace_manager.resolve_path(path)
        else:
            resolved_path = Path(path)

        with open(resolved_path, encoding=encoding, errors="replace") as f:
            content = f.read(max_size)

            file_size = resolved_path.stat().st_size
            if file_size > max_size:
                return content + f"\n\n[TRUNCATED: File is {file_size} bytes, showing first {max_size}. Use run_bash('head -n 100 {path}') or similar for specific sections]"
            return content
