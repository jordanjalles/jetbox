"""
Write file tools behavior - provides file writing operations.

Extracts write_file tool from file_tools.py into a composable AgentBehavior:
- write_file: Write/overwrite files with safety checks

Features:
- Workspace-aware path resolution (uses WorkspaceManager)
- Safety checks for edit mode (forbidden files)
- Ledger logging for audit trail
- Append mode support
- Custom encoding and line endings
- Overwrite control
- Parameter invention tolerance (**kwargs)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from behaviors.base import AgentBehavior


class WriteFileToolsBehavior(AgentBehavior):
    """
    Provides file writing tool: write_file.

    Workspace-aware file operations with safety checks and audit logging.
    """

    def __init__(
        self,
        workspace_manager=None,
        ledger_file: Path | None = None,
        **kwargs
    ):
        """
        Initialize WriteFileToolsBehavior.

        Args:
            workspace_manager: WorkspaceManager instance for path resolution
            ledger_file: Path to ledger file for audit logging
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager
        self.ledger_file = ledger_file

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "write_file_tools"

    def get_tools(self) -> list[dict[str, Any]]:
        """Return file writing tool definitions."""
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
            }
        ]

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> str:
        """
        Dispatch file writing tool calls.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: Additional context (workspace, ledger_file, workspace_manager)

        Returns:
            Tool result (success message string)
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
                print(f"[write_file_tools] write_file ignoring unsupported parameters: {ignored}")

        # Normalize line endings if requested
        if line_end is not None:
            # First normalize to \n, then replace with desired ending
            normalized = content.replace('\r\n', '\n').replace('\r', '\n')
            if line_end != '\n':
                content = normalized.replace('\n', line_end)
            else:
                content = normalized

        # Resolve path through workspace if available
        logger.debug("Starting write_file for path: %s", path)
        logger.debug("workspace_manager: %s", workspace_manager)

        if workspace_manager:
            logger.debug("workspace_manager.workspace_dir: %s", workspace_manager.workspace_dir)
            resolved_path = workspace_manager.resolve_path(path)
            logger.debug("Resolved path: %s", resolved_path)
            logger.debug("Resolved path absolute: %s", resolved_path.resolve())

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
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            workspace_manager.track_file(path)  # Track file creation
            display_path = workspace_manager.relative_path(resolved_path)
        else:
            logger.debug("No workspace_manager, using raw path")
            resolved_path = Path(path)
            display_path = path
            logger.debug("Resolved path (no wm): %s", resolved_path.resolve())

        # Check overwrite flag
        if not overwrite and resolved_path.exists():
            error_msg = f"[ERROR] File exists and overwrite=False: {display_path}"
            self._ledger_append("ERROR", error_msg, ledger_file)
            logger.error(error_msg)
            raise FileExistsError(error_msg)

        if create_dirs:
            parent_dir = os.path.dirname(resolved_path) or "."
            logger.debug("Creating parent directories: %s", parent_dir)
            os.makedirs(parent_dir, exist_ok=True)

        # Choose write mode based on append flag
        # Use newline='' to prevent Python from translating line endings
        mode = "a" if append else "w"
        newline = '' if line_end is not None else None

        logger.debug("Writing %d chars to %s", len(content), resolved_path)
        try:
            with open(resolved_path, mode, encoding=encoding, newline=newline) as f:
                f.write(content)
            logger.debug("Write successful!")

            # Verify file was created
            if not resolved_path.exists():
                error_msg = f"[ERROR] File write reported success but file doesn't exist: {resolved_path}"
                logger.error(error_msg)
                raise IOError(error_msg)

            file_size = resolved_path.stat().st_size
            logger.debug("File exists after write, size: %d bytes", file_size)

        except Exception as e:
            error_msg = f"[ERROR] Failed to write file {resolved_path}: {e}"
            logger.error(error_msg)
            raise

        action = "Appended" if append else "Wrote"
        self._ledger_append("WRITE" if not append else "APPEND", str(resolved_path), ledger_file)
        return f"{action} {len(content)} chars to {display_path}"
