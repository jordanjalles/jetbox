"""
Read file tools behavior - provides file reading operations.

Extracts read_file tool from file_tools.py into a composable AgentBehavior:
- read_file: Read file contents with size limits and encoding handling

Features:
- Workspace-aware path resolution (uses WorkspaceManager)
- Size limits with truncation warnings
- Encoding error handling (error replacement)
- Parameter invention tolerance (**kwargs)

Now uses @tool decorator for automatic tool registration!
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty
from behaviors.tool_decorator import tool


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

    # Tool implementation (using @tool decorator)

    @tool
    def read_file(
        self,
        path: str,
        encoding: str = "utf-8",
        max_size: int = 1_000_000
    ) -> str:
        """Read a text file (up to 1MB by default). For large files, adjust max_size or use run_bash with head/tail.

        Args:
            path: File path (relative to workspace)
            encoding: Text encoding (default: utf-8)
            max_size: Maximum bytes to read (default: 1000000)

        Returns:
            File contents (up to max_size, truncated if larger)
        """
        # Access agent via self.agent (injected by decorator)
        workspace_manager = getattr(self.agent, 'workspace_manager', self.workspace_manager)

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
