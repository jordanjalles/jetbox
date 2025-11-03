"""
File tools behavior - provides file operations with workspace awareness.

Extracts file tools from tools.py into a composable AgentBehavior:
- write_file: Write/overwrite files with safety checks
- read_file: Read files with size limits and encoding handling
- list_dir: List directory contents

Features:
- Workspace-aware path resolution (uses WorkspaceManager)
- Safety checks for edit mode (forbidden files)
- Ledger logging for audit trail
- Parameter invention tolerance (**kwargs)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from behaviors.base import AgentBehavior


class FileToolsBehavior(AgentBehavior):
    """
    Provides file operation tools: write_file, read_file, list_dir.

    Workspace-aware file operations with safety checks and audit logging.
    """

    def __init__(
        self,
        workspace_manager=None,
        ledger_file: Path | None = None,
        **kwargs
    ):
        """
        Initialize FileToolsBehavior.

        Args:
            workspace_manager: WorkspaceManager instance for path resolution
            ledger_file: Path to ledger file for audit logging
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager
        self.ledger_file = ledger_file

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "file_tools"

    def get_tools(self) -> list[dict[str, Any]]:
        """Return file operation tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write/overwrite a text file. Supports append mode, custom encoding, line endings, and overwrite control.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path (relative to workspace)"
                            },
                            "content": {
                                "type": "string",
                                "description": "Complete file contents to write"
                            },
                            "append": {
                                "type": "boolean",
                                "description": "If true, append to file instead of overwriting (default: false)"
                            },
                            "encoding": {
                                "type": "string",
                                "description": "Text encoding (default: utf-8)"
                            },
                            "line_end": {
                                "type": "string",
                                "description": "Line ending style: '\\n' (Unix), '\\r\\n' (Windows), or null for system default"
                            },
                            "overwrite": {
                                "type": "boolean",
                                "description": "If false and file exists, return error (default: true)"
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
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
            },
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
    ) -> dict[str, Any] | str | list[str]:
        """
        Dispatch file operation tool calls.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: Additional context (workspace, ledger_file, workspace_manager)

        Returns:
            Tool result (string or dict or list)
        """
        # Allow runtime override of workspace_manager and ledger_file
        workspace_manager = kwargs.get('workspace_manager', self.workspace_manager)
        ledger_file = kwargs.get('ledger_file', self.ledger_file)

        if tool_name == "write_file":
            return self._write_file(
                args.get("path"),
                args.get("content"),
                append=args.get("append", False),
                encoding=args.get("encoding", "utf-8"),
                create_dirs=args.get("create_dirs", True),
                overwrite=args.get("overwrite", True),
                line_end=args.get("line_end"),
                workspace_manager=workspace_manager,
                ledger_file=ledger_file,
                extra_kwargs=args
            )
        elif tool_name == "read_file":
            return self._read_file(
                args.get("path"),
                encoding=args.get("encoding", "utf-8"),
                max_size=args.get("max_size", 1_000_000),
                workspace_manager=workspace_manager,
                extra_kwargs=args
            )
        elif tool_name == "list_dir":
            return self._list_dir(
                args.get("path", "."),
                workspace_manager=workspace_manager,
                extra_kwargs=args
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

    def _write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
        encoding: str = "utf-8",
        create_dirs: bool = True,
        overwrite: bool = True,
        line_end: str | None = None,
        workspace_manager=None,
        ledger_file: Path | None = None,
        extra_kwargs: dict | None = None
    ) -> str:
        """
        Write/overwrite a text file (workspace-aware).

        Args:
            path: File path (relative to workspace if set)
            content: File contents to write
            append: If True, append to file instead of overwriting
            encoding: Text encoding
            create_dirs: Create parent directories if needed
            overwrite: If False and file exists, return error
            line_end: Line ending to use ('\\n', '\\r\\n', or None for system default)
            workspace_manager: WorkspaceManager instance
            ledger_file: Ledger file path
            extra_kwargs: Full args dict for parameter invention detection

        Returns:
            Success message with path and size
        """
        # Warn about unsupported parameters (parameter invention tolerance)
        if extra_kwargs:
            supported = {
                "path", "content", "append", "encoding", "create_dirs",
                "overwrite", "line_end"
            }
            unsupported = set(extra_kwargs.keys()) - supported
            if unsupported:
                ignored = ", ".join(unsupported)
                print(f"[file_tools] write_file ignoring unsupported parameters: {ignored}")

        # Normalize line endings if requested
        if line_end is not None:
            # First normalize to \n, then replace with desired ending
            normalized = content.replace('\r\n', '\n').replace('\r', '\n')
            if line_end != '\n':
                content = normalized.replace('\n', line_end)
            else:
                content = normalized

        # Resolve path through workspace if available
        if workspace_manager:
            resolved_path = workspace_manager.resolve_path(path)

            # Safety check in edit mode: prevent modifying agent code
            if workspace_manager.is_edit_mode:
                forbidden_files = {
                    'agent.py', 'context_manager.py', 'workspace_manager.py',
                    'status_display.py', 'completion_detector.py', 'agent_config.py',
                    'tools.py', 'llm_utils.py'
                }
                if resolved_path.name in forbidden_files:
                    error_msg = f"[SAFETY] Cannot modify agent code in edit mode: {resolved_path.name}"
                    self._ledger_append("ERROR", error_msg, ledger_file)
                    return error_msg

            workspace_manager.track_file(path)  # Track file creation
            display_path = workspace_manager.relative_path(resolved_path)
        else:
            resolved_path = Path(path)
            display_path = path

        # Check overwrite flag
        if not overwrite and resolved_path.exists():
            error_msg = f"[ERROR] File exists and overwrite=False: {display_path}"
            self._ledger_append("ERROR", error_msg, ledger_file)
            return error_msg

        if create_dirs:
            os.makedirs(os.path.dirname(resolved_path) or ".", exist_ok=True)

        # Choose write mode based on append flag
        # Use newline='' to prevent Python from translating line endings
        mode = "a" if append else "w"
        newline = '' if line_end is not None else None
        with open(resolved_path, mode, encoding=encoding, newline=newline) as f:
            f.write(content)

        action = "Appended" if append else "Wrote"
        self._ledger_append("WRITE" if not append else "APPEND", str(resolved_path), ledger_file)
        return f"{action} {len(content)} chars to {display_path}"

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
                print(f"[file_tools] read_file ignoring unsupported parameters: {ignored}")

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
                print(f"[file_tools] list_dir ignoring unsupported parameters: {ignored}")

        # Resolve path through workspace if available
        if workspace_manager:
            resolved_path = workspace_manager.resolve_path(path or ".")
        else:
            resolved_path = Path(path or ".")

        try:
            return sorted(os.listdir(resolved_path))
        except FileNotFoundError as e:
            return [f"__error__: {e}"]
