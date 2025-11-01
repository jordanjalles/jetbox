"""
Compatibility wrapper for tools.py - delegates to behaviors.

This file contains only the wrapper functions that delegate to behaviors.
The actual tool implementations are in the behaviors/ directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Import behaviors for delegation
from behaviors import FileToolsBehavior, CommandToolsBehavior, ServerToolsBehavior

# Global references (for backward compatibility)
_workspace = None
_ledger_file = None
_file_tools: FileToolsBehavior | None = None
_command_tools: CommandToolsBehavior | None = None
_server_tools: ServerToolsBehavior | None = None


def set_workspace(workspace_manager) -> None:
    """Set workspace manager (backward compatibility)."""
    global _workspace, _file_tools, _command_tools, _server_tools
    _workspace = workspace_manager
    _file_tools = FileToolsBehavior(workspace_manager=workspace_manager, ledger_file=_ledger_file)
    _command_tools = CommandToolsBehavior(workspace_manager=workspace_manager, ledger_file=_ledger_file)
    _server_tools = ServerToolsBehavior(workspace_manager=workspace_manager, ledger_file=_ledger_file)


def set_ledger(ledger_path: Path) -> None:
    """Set ledger file (backward compatibility)."""
    global _ledger_file, _file_tools, _command_tools, _server_tools
    _ledger_file = ledger_path
    if _file_tools:
        _file_tools.ledger_file = ledger_path
    if _command_tools:
        _command_tools.ledger_file = ledger_path
    if _server_tools:
        _server_tools.ledger_file = ledger_path


def list_dir(path: str | None = ".", **kwargs) -> list[str]:
    """List directory contents."""
    global _file_tools
    if not _file_tools:
        _file_tools = FileToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _file_tools.dispatch_tool("list_dir", {"path": path or ".", **kwargs})


def read_file(path: str, encoding: str = "utf-8", max_size: int = 1_000_000, **kwargs) -> str:
    """Read file contents."""
    global _file_tools
    if not _file_tools:
        _file_tools = FileToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _file_tools.dispatch_tool("read_file", {
        "path": path, "encoding": encoding, "max_size": max_size, **kwargs
    })


def write_file(
    path: str,
    content: str,
    append: bool = False,
    encoding: str = "utf-8",
    create_dirs: bool = True,
    overwrite: bool = True,
    line_end: str | None = None,
    **kwargs
) -> str:
    """Write file contents."""
    global _file_tools
    if not _file_tools:
        _file_tools = FileToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _file_tools.dispatch_tool("write_file", {
        "path": path,
        "content": content,
        "append": append,
        "encoding": encoding,
        "create_dirs": create_dirs,
        "overwrite": overwrite,
        "line_end": line_end,
        **kwargs
    })


def run_bash(command: str, timeout: int = 60) -> dict[str, Any]:
    """Run bash command."""
    global _command_tools
    if not _command_tools:
        _command_tools = CommandToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _command_tools.dispatch_tool("run_bash", {"command": command, "timeout": timeout})


def start_server(cmd: list[str], name: str = None) -> dict[str, Any]:
    """Start background server."""
    global _server_tools
    if not _server_tools:
        _server_tools = ServerToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _server_tools.dispatch_tool("start_server", {"cmd": cmd, "name": name})


def stop_server(server_id: str) -> dict[str, Any]:
    """Stop server."""
    global _server_tools
    if not _server_tools:
        _server_tools = ServerToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _server_tools.dispatch_tool("stop_server", {"server_id": server_id})


def check_server(server_id: str, tail_lines: int = 20) -> dict[str, Any]:
    """Check server status."""
    global _server_tools
    if not _server_tools:
        _server_tools = ServerToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _server_tools.dispatch_tool("check_server", {"server_id": server_id, "tail_lines": tail_lines})


def list_servers() -> dict[str, Any]:
    """List all servers."""
    global _server_tools
    if not _server_tools:
        _server_tools = ServerToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _server_tools.dispatch_tool("list_servers", {})


# The rest of tools.py (mark_subtask_complete, decompose_task, etc.) are context management tools
# and should remain in tools.py as they depend on context_manager
