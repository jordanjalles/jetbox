"""
Directory tools behavior - provides directory navigation operations.

Extracts list_dir tool from file_tools.py into a composable AgentBehavior:
- list_dir: List directory contents (non-recursive)

Features:
- Workspace-aware path resolution (uses WorkspaceManager)
- Parameter invention tolerance (**kwargs)
- Error handling for missing directories

Now uses @tool decorator for automatic tool registration!
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty
from behaviors.tool_decorator import tool


class DirectoryToolsBehavior(AgentBehavior):
    """
    Provides directory navigation tool: list_dir.

    Workspace-aware directory listing with error handling.

    Security: DYNAMIC based on workspace characteristics
    - [B] SENSITIVE_ACCESS: Only if workspace has sensitive files
    - [] None: If workspace has no sensitive files (e.g., Jetbox repo)
    - Lists directory contents (reveals file structure)
    - Can reveal sensitive file names (.env, credentials.json) IF they exist
    - Does not read file contents or execute commands
    """

    # Rule of Two: Empty static fallback (dynamically computed at runtime)
    rule_of_two_properties = set()

    def __init__(
        self,
        workspace_manager=None,
        **kwargs
    ):
        """
        Initialize DirectoryToolsBehavior.

        Args:
            workspace_manager: WorkspaceManager instance for path resolution
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "directory_tools"

    def get_rule_of_two_properties(self, agent, security_context):
        """
        Get Rule of Two properties (context-aware).

        Dynamic behavior based on workspace characteristics:
        - [B] SENSITIVE_ACCESS: Only if workspace has sensitive files
        - [] None: If workspace has no sensitive files (Jetbox repo, public code)

        Args:
            agent: Agent instance
            security_context: SecurityContext with workspace characteristics

        Returns:
            Set of properties for current context ([] or [B])
        """
        props = set()

        # [B] SENSITIVE_ACCESS - only if workspace contains sensitive files
        if security_context and security_context.workspace_has_sensitive_files:
            props.add(RuleOfTwoProperty.SENSITIVE_ACCESS)

        return props

    # Tool implementation (using @tool decorator)

    @tool
    def list_dir(
        self,
        path: str = ".",
        depth: int = 0
    ) -> list[str]:
        """List files in a directory. Can list recursively with depth parameter.

        Args:
            path: Directory path (relative to workspace), default '.'
            depth: Recursion depth (0=current dir only, 1=one level deep, etc.). Default 0 (non-recursive).

        Returns:
            Sorted list of filenames/paths, or error message list
        """
        # Access agent via self.agent (injected by decorator)
        workspace_manager = getattr(self.agent, 'workspace_manager', self.workspace_manager)

        # Resolve path through workspace if available
        if workspace_manager:
            resolved_path = workspace_manager.resolve_path(path or ".")
        else:
            resolved_path = Path(path or ".")

        try:
            if depth == 0:
                # Non-recursive - just list current directory
                results = sorted(os.listdir(resolved_path))
            else:
                # Recursive - walk directory tree up to specified depth
                results = []
                self._walk_directory(resolved_path, resolved_path, depth, 0, results)
                results = sorted(results)

            # Return clear message if directory is empty to avoid LLM confusion
            if not results:
                return ["(directory empty)"]

            return results
        except FileNotFoundError as e:
            return [f"__error__: {e}"]

    # Helper methods

    def _walk_directory(
        self,
        base_path: Path,
        current_path: Path,
        max_depth: int,
        current_depth: int,
        results: list[str]
    ) -> None:
        """
        Recursively walk directory tree up to max_depth.

        Args:
            base_path: Base directory (for relative path calculation)
            current_path: Current directory being processed
            max_depth: Maximum depth to recurse
            current_depth: Current depth (0 = base level)
            results: List to append results to (modified in place)
        """
        if current_depth > max_depth:
            return

        try:
            entries = sorted(os.listdir(current_path))
        except (PermissionError, OSError):
            # Skip directories we can't read
            return

        for entry in entries:
            full_path = current_path / entry
            # Calculate relative path from base
            try:
                rel_path = full_path.relative_to(base_path)
            except ValueError:
                # Path is not relative to base (shouldn't happen)
                rel_path = full_path

            # Add entry with relative path
            if full_path.is_dir():
                results.append(f"{rel_path}/")  # Mark directories with trailing slash
                # Recurse into subdirectory if not at max depth
                if current_depth < max_depth:
                    self._walk_directory(base_path, full_path, max_depth, current_depth + 1, results)
            else:
                results.append(str(rel_path))
