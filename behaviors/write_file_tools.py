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

Now uses @tool decorator for automatic tool registration!
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from behaviors.base import AgentBehavior
from behaviors.rule_of_two_types import RuleOfTwoProperty
from behaviors.tool_decorator import tool


class WriteFileToolsBehavior(AgentBehavior):
    """
    Provides file writing tool: write_file.

    Workspace-aware file operations with safety checks and audit logging.

    Security: [] None
    - Writes/modifies files locally (internal state change, not external communication)
    - Does not read untrusted input (writes are agent-generated)
    - Does not access sensitive data (not inherently, though could write to sensitive locations)
    - Does not communicate externally via network
    """

    # Rule of Two: [] - local file writes are internal state changes, not network communication
    rule_of_two_properties = set()

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

    # Tool implementation (using @tool decorator)

    @tool(description="Write/overwrite a text file. Supports append mode, custom encoding, line endings, and overwrite control.")
    def write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
        encoding: str = "utf-8",
        line_end: str | None = None,
        overwrite: bool = True
    ) -> str:
        """
        Write/overwrite a text file (workspace-aware).

        Args:
            path: File path (relative to workspace)
            content: Complete file contents to write
            append: If true, append to file instead of overwriting (default: false)
            encoding: Text encoding (default: utf-8)
            line_end: Line ending style: '\\n' (Unix), '\\r\\n' (Windows), or null for system default
            overwrite: If false and file exists, return error (default: true)

        Returns:
            Success message with path and size
        """
        # Access agent via self.agent (injected by decorator)
        workspace_manager = getattr(self.agent, 'workspace_manager', self.workspace_manager)
        ledger_file = getattr(self.agent, 'ledger_file', self.ledger_file)

        # Decode JSON-style escape sequences if present
        content = self._decode_escape_sequences(content)

        # Normalize line endings if requested
        if line_end is not None:
            normalized = content.replace('\r\n', '\n').replace('\r', '\n')
            if line_end != '\n':
                content = normalized.replace('\n', line_end)
            else:
                content = normalized

        # Resolve path through workspace if available
        logger.debug("Starting write_file for path: %s", path)

        if workspace_manager:
            resolved_path = workspace_manager.resolve_path(path)

            # Safety check in edit mode: prevent modifying agent code
            if workspace_manager.is_edit_mode:
                forbidden_files = {
                    'agent.py', 'workspace_manager.py',
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
            resolved_path = Path(path)
            display_path = path

        # Check overwrite flag
        if not overwrite and resolved_path.exists():
            error_msg = f"[ERROR] File exists and overwrite=False: {display_path}"
            self._ledger_append("ERROR", error_msg, ledger_file)
            logger.error(error_msg)
            raise FileExistsError(error_msg)

        # Create parent directories
        parent_dir = os.path.dirname(resolved_path) or "."
        os.makedirs(parent_dir, exist_ok=True)

        # Choose write mode based on append flag
        mode = "a" if append else "w"
        newline = '' if line_end is not None else None

        logger.debug("Writing %d chars to %s", len(content), resolved_path)
        try:
            with open(resolved_path, mode, encoding=encoding, newline=newline) as f:
                f.write(content)

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

    # Helper methods

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

    def _decode_escape_sequences(self, content: str) -> str:
        """
        Decode JSON-style escape sequences in content.

        When LLMs send tool calls, JSON encoding escapes special characters:
        - Newlines become literal \\n
        - Tabs become literal \\t
        - Quotes become literal \\" or \\'

        This method converts these back to actual characters for proper file writing.

        Args:
            content: String potentially containing escape sequences

        Returns:
            String with escape sequences decoded
        """
        # Quick check: if no backslashes, no escaping possible
        if '\\' not in content:
            return content

        # Count literal \n sequences (indicating JSON escaping)
        literal_newlines = content.count('\\n')
        actual_newlines = content.count('\n')

        # Heuristic: If we have many more literal \n than actual newlines,
        # this content is likely JSON-escaped and needs decoding
        if literal_newlines > actual_newlines * 2:
            try:
                # Decode using Python's string-escape codec
                # This handles: \n, \t, \r, \\, \", \', and other escapes
                decoded = content.encode().decode('unicode_escape')

                # Verify decoding made changes (safety check)
                if decoded != content:
                    logger.debug(
                        f"Decoded {literal_newlines} escape sequences "
                        f"(content size: {len(content)} -> {len(decoded)} chars)"
                    )
                    return decoded
            except (UnicodeDecodeError, AttributeError) as e:
                # Decoding failed, return original
                logger.warning(f"Failed to decode escape sequences: {e}, using original content")
                return content

        return content
