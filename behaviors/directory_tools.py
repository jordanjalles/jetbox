"""
Directory tools behavior - provides directory navigation operations.

Extracts list_dir tool from file_tools.py into a composable AgentBehavior:
- list_dir: List directory contents (non-recursive)

Features:
- Workspace-aware path resolution (uses WorkspaceManager)
- Parameter invention tolerance (**kwargs)
- Error handling for missing directories
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from behaviors.base import AgentBehavior


class DirectoryToolsBehavior(AgentBehavior):
    """
    Provides directory navigation tool: list_dir.

    Workspace-aware directory listing with error handling.
    """

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

    def get_tools(self) -> list[dict[str, Any]]:
        """Return directory operation tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List files in a directory. Can list recursively with depth parameter.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path (relative to workspace), default '.'"
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Recursion depth (0=current dir only, 1=one level deep, etc.). Default 0 (non-recursive)."
                            }
                        }
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> list[str]:
        """
        Dispatch directory operation tool calls.

        Args:
            agent: Agent instance
            tool_name: Tool being called
            args: Tool arguments

        Returns:
            Tool result (list of filenames or error message list)
        """
        # Get workspace_manager from agent
        workspace_manager = getattr(agent, 'workspace_manager', self.workspace_manager)

        if tool_name == "list_dir":
            return self._list_dir(
                path=args.get("path", "."),
                depth=args.get("depth", 0),
                workspace_manager=workspace_manager,
                extra_kwargs=args
            )
        else:
            return super().dispatch_tool(agent, tool_name, args)

    def _list_dir(
        self,
        path: str | None = ".",
        depth: int = 0,
        workspace_manager=None,
        extra_kwargs: dict | None = None
    ) -> list[str]:
        """
        List files in directory (optionally recursive, workspace-aware).

        Args:
            path: Directory path (relative to workspace if set)
            depth: Recursion depth (0=current dir only, 1=one level deep, etc.)
            workspace_manager: WorkspaceManager instance
            extra_kwargs: Full args dict for parameter invention detection

        Returns:
            Sorted list of filenames/paths, or error message list
        """
        # Warn about unsupported parameters
        if extra_kwargs:
            supported = {"path", "depth"}
            unsupported = set(extra_kwargs.keys()) - supported
            if unsupported:
                ignored = ", ".join(unsupported)
                print(f"[directory_tools] list_dir ignoring unsupported parameters: {ignored}")

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
