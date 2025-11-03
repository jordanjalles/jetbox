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
                    "description": "List files in a directory (non-recursive).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path (relative to workspace), default '.'"
                            }
                        }
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> list[str]:
        """
        Dispatch directory operation tool calls.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: Additional context (workspace_manager)

        Returns:
            Tool result (list of filenames or error message list)
        """
        # Allow runtime override of workspace_manager
        workspace_manager = kwargs.get('workspace_manager', self.workspace_manager)

        if tool_name == "list_dir":
            return self._list_dir(
                args.get("path", "."),
                workspace_manager=workspace_manager,
                extra_kwargs=args
            )
        else:
            return super().dispatch_tool(tool_name, args, **kwargs)

    def _list_dir(
        self,
        path: str | None = ".",
        workspace_manager=None,
        extra_kwargs: dict | None = None
    ) -> list[str]:
        """
        List files in directory (non-recursive, workspace-aware).

        Args:
            path: Directory path (relative to workspace if set)
            workspace_manager: WorkspaceManager instance
            extra_kwargs: Full args dict for parameter invention detection

        Returns:
            Sorted list of filenames, or error message list
        """
        # Warn about unsupported parameters
        if extra_kwargs:
            supported = {"path"}
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
            return sorted(os.listdir(resolved_path))
        except FileNotFoundError as e:
            return [f"__error__: {e}"]
