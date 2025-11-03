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


class ReadFileToolsBehavior(AgentBehavior):
    """
    Provides file reading tool: read_file.

    Workspace-aware file reading with size limits and encoding handling.
    """

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
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> str:
        """
        Dispatch file reading tool calls.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: Additional context (workspace_manager)

        Returns:
            Tool result (file contents string)
        """
        # Allow runtime override of workspace_manager
        workspace_manager = kwargs.get('workspace_manager', self.workspace_manager)

        if tool_name == "read_file":
            return self._read_file(
                args.get("path"),
                encoding=args.get("encoding", "utf-8"),
                max_size=args.get("max_size", 1_000_000),
                workspace_manager=workspace_manager,
                extra_kwargs=args
            )
        else:
            return super().dispatch_tool(tool_name, args, **kwargs)

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
