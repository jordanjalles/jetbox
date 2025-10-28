"""
Shared tool implementations for Jetbox agents.

All file-based tools (read_file, write_file, list_dir, grep_file) are workspace-aware:
they use WorkspaceManager to resolve paths and enforce isolation.

All tools return structured results (strings or dicts) suitable for LLM consumption.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# Whitelisted commands for run_cmd (Windows safety)
SAFE_BIN = {"python", "pytest", "ruff", "pip"}

# Global references set by agent at runtime
_workspace = None  # WorkspaceManager instance
_ledger_file = None  # Path to ledger file for audit trail


def set_workspace(workspace_manager) -> None:
    """Set the workspace manager for path resolution."""
    global _workspace
    _workspace = workspace_manager


def set_ledger(ledger_path: Path) -> None:
    """Set the ledger file path for audit logging."""
    global _ledger_file
    _ledger_file = ledger_path


def _ledger_append(kind: str, detail: str) -> None:
    """Append action to ledger file for audit trail."""
    if not _ledger_file:
        return
    line = f"{kind}\t{detail.replace(chr(10), ' ')[:400]}\n"
    if _ledger_file.exists():
        _ledger_file.write_text(
            _ledger_file.read_text(encoding="utf-8") + line,
            encoding="utf-8"
        )
    else:
        _ledger_file.write_text(line, encoding="utf-8")


# ----------------------------
# File Operation Tools
# ----------------------------

def list_dir(path: str | None = ".", **kwargs) -> list[str]:
    """
    List files in directory (non-recursive, workspace-aware).

    Args:
        path: Directory path (relative to workspace if set)

    Returns:
        Sorted list of filenames, or error message list
    """
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path or ".")
    else:
        resolved_path = Path(path or ".")

    try:
        return sorted(os.listdir(resolved_path))
    except FileNotFoundError as e:
        return [f"__error__: {e}"]


def read_file(path: str, max_bytes: int = 200_000, offset: int = 0) -> str:
    """
    Read a text file with optional offset for pagination (workspace-aware).

    Args:
        path: File path (relative to workspace if set)
        max_bytes: Maximum bytes to read (default 200KB)
        offset: Byte offset to start reading from

    Returns:
        File contents with metadata if truncated
    """
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path)
    else:
        resolved_path = Path(path)

    with open(resolved_path, encoding="utf-8", errors="replace") as f:
        if offset > 0:
            f.seek(offset)
        content = f.read(max_bytes)

        # Add helpful metadata if file was truncated or offset was used
        file_size = resolved_path.stat().st_size
        if offset > 0 or len(content) == max_bytes:
            end_pos = offset + len(content)
            metadata = f"\n[FILE READ: bytes {offset}-{end_pos} of {file_size} total]"
            if end_pos < file_size:
                metadata += f"\n[NOTE: {file_size - end_pos} bytes remaining. Use offset={end_pos} to continue reading]"
            return content + metadata
        return content


def grep_file(path: str, pattern: str, context_lines: int = 3, max_matches: int = 50) -> str:
    """
    Search for pattern in file and return matching lines with context.

    Args:
        path: File path (relative to workspace if set)
        pattern: Python regex pattern to search for
        context_lines: Lines of context before/after match (default 3)
        max_matches: Maximum matches to return (default 50)

    Returns:
        Formatted string with matches and context, or error message
    """
    import re
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path)
    else:
        resolved_path = Path(path)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[ERROR] Invalid regex pattern: {e}"

    try:
        with open(resolved_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Find all matching lines
        matching_line_nums = []
        for i, line in enumerate(lines):
            if regex.search(line):
                matching_line_nums.append(i)
                if len(matching_line_nums) >= max_matches:
                    break

        if not matching_line_nums:
            return f"[NO MATCHES] Pattern '{pattern}' not found in {path}"

        # Build output with context
        result_lines = []
        result_lines.append(f"[GREP] Found {len(matching_line_nums)} match(es) for '{pattern}' in {path}")
        result_lines.append("")

        # For each match, include context lines
        included_lines = set()
        for line_num in matching_line_nums:
            start = max(0, line_num - context_lines)
            end = min(len(lines), line_num + context_lines + 1)

            # Add separator if there's a gap from previous match
            if included_lines and start > max(included_lines) + 1:
                result_lines.append("...")
                result_lines.append("")

            for i in range(start, end):
                if i not in included_lines:
                    prefix = ">>> " if i == line_num else "    "
                    result_lines.append(f"{prefix}{i+1:4d} | {lines[i].rstrip()}")
                    included_lines.add(i)

            result_lines.append("")

        if len(matching_line_nums) >= max_matches:
            result_lines.append(f"[NOTE] Showing first {max_matches} matches. File may contain more.")

        return "\n".join(result_lines)

    except Exception as e:
        return f"[ERROR] Failed to search file: {e}"


def write_file(path: str, content: str, create_dirs: bool = True) -> str:
    """
    Write/overwrite a text file (workspace-aware).

    Args:
        path: File path (relative to workspace if set)
        content: File contents to write
        create_dirs: Create parent directories if needed (default True)

    Returns:
        Success message with path and size
    """
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path)

        # Safety check in edit mode: prevent modifying agent code
        if _workspace.is_edit_mode:
            forbidden_files = {
                'agent.py', 'context_manager.py', 'workspace_manager.py',
                'status_display.py', 'completion_detector.py', 'agent_config.py',
                'tools.py', 'llm_utils.py'  # Added new modules
            }
            if resolved_path.name in forbidden_files:
                error_msg = f"[SAFETY] Cannot modify agent code in edit mode: {resolved_path.name}"
                _ledger_append("ERROR", error_msg)
                return error_msg

        _workspace.track_file(path)  # Track file creation
        display_path = _workspace.relative_path(resolved_path)
    else:
        resolved_path = Path(path)
        display_path = path

    if create_dirs:
        os.makedirs(os.path.dirname(resolved_path) or ".", exist_ok=True)

    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(content)

    _ledger_append("WRITE", str(resolved_path))
    return f"Wrote {len(content)} chars to {display_path}"


def run_cmd(cmd: list[str], timeout_sec: int = 60) -> dict[str, Any]:
    """
    Run a whitelisted command (workspace-aware).

    First token must be in SAFE_BIN whitelist for Windows safety.

    Args:
        cmd: Command as list (e.g., ['python', 'script.py'])
        timeout_sec: Timeout in seconds (default 60)

    Returns:
        Dict with returncode, stdout, stderr, or error key
    """
    global _workspace

    if not cmd or cmd[0] not in SAFE_BIN:
        err = f"Command not allowed: {cmd!r}. Use only {sorted(SAFE_BIN)}."
        _ledger_append("ERROR", err)
        return {"error": err}

    # Determine working directory
    cwd = str(_workspace.workspace_dir) if _workspace else None

    # Set up environment with PYTHONPATH for workspace
    env = os.environ.copy()
    if _workspace and cwd:
        # Add workspace directory to PYTHONPATH for pytest imports
        if "pytest" in cmd or "python" in cmd:
            env["PYTHONPATH"] = cwd

    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=cwd,
            env=env
        )
        out = {
            "returncode": p.returncode,
            "stdout": p.stdout[-50_000:],  # Truncate to last 50KB
            "stderr": p.stderr[-50_000:],
        }
        _ledger_append("CMD", f"{cmd} -> rc={p.returncode}")
        if p.returncode != 0:
            _ledger_append("ERROR", f"run_cmd rc={p.returncode}: {p.stderr[:200]}")
        return out
    except subprocess.TimeoutExpired:
        err = f"timeout after {timeout_sec}s"
        _ledger_append("ERROR", f"run_cmd timeout: {cmd}")
        return {"error": err}
    except Exception as e:
        _ledger_append("ERROR", f"run_cmd exception: {e}")
        return {"error": str(e)}


# ----------------------------
# Server Management Tools
# ----------------------------

def start_server(cmd: list[str], name: str = None) -> dict[str, Any]:
    """
    Request orchestrator to start a server in the background.

    Args:
        cmd: Command to run (e.g., ['python', '-m', 'http.server', '8000'])
        name: Optional server name (auto-generated if not provided)

    Returns:
        Server info dict or error dict
    """
    global _workspace

    # Validate command
    if not cmd or cmd[0] not in SAFE_BIN:
        return {"error": f"Command not allowed: {cmd!r}. Use only {sorted(SAFE_BIN)}."}

    # Generate server ID
    server_id = name or f"server_{int(time.time())}"

    # Set up paths
    cwd = str(_workspace.workspace_dir) if _workspace else os.getcwd()
    log_file = os.path.join(cwd, f"{server_id}.log")

    # Write request
    request_file = Path(".agent_context/server_requests.jsonl")
    request_file.parent.mkdir(parents=True, exist_ok=True)

    request = {
        "action": "start",
        "server_id": server_id,
        "cmd": cmd,
        "cwd": cwd,
        "log_file": log_file,
    }

    with open(request_file, 'a') as f:
        f.write(json.dumps(request) + '\n')

    # Poll for response (wait up to 5 seconds)
    response = _wait_for_server_response(timeout=5.0)

    if response:
        _ledger_append("SERVER", f"start {server_id} -> {response.get('success', False)}")
        return response
    else:
        return {"error": "Timeout waiting for orchestrator to start server"}


def stop_server(server_id: str) -> dict[str, Any]:
    """
    Request orchestrator to stop a server.

    Args:
        server_id: Server identifier

    Returns:
        Success/error dict
    """
    request = {"action": "stop", "server_id": server_id}

    request_file = Path(".agent_context/server_requests.jsonl")
    with open(request_file, 'a') as f:
        f.write(json.dumps(request) + '\n')

    response = _wait_for_server_response(timeout=5.0)

    if response:
        _ledger_append("SERVER", f"stop {server_id} -> {response.get('success', False)}")

    return response or {"error": "Timeout waiting for response"}


def check_server(server_id: str, tail_lines: int = 20) -> dict[str, Any]:
    """
    Check server status and get recent logs.

    Args:
        server_id: Server identifier
        tail_lines: Number of recent log lines to return (default 20)

    Returns:
        Server status dict with logs
    """
    request = {"action": "check", "server_id": server_id, "tail_lines": tail_lines}

    request_file = Path(".agent_context/server_requests.jsonl")
    with open(request_file, 'a') as f:
        f.write(json.dumps(request) + '\n')

    response = _wait_for_server_response(timeout=5.0)
    return response or {"error": "Timeout waiting for response"}


def list_servers() -> dict[str, Any]:
    """
    List all running servers.

    Returns:
        Dict with list of servers or error
    """
    request = {"action": "list"}

    request_file = Path(".agent_context/server_requests.jsonl")
    with open(request_file, 'a') as f:
        f.write(json.dumps(request) + '\n')

    response = _wait_for_server_response(timeout=5.0)
    return response or {"error": "Timeout waiting for response"}


def _wait_for_server_response(timeout: float = 5.0) -> dict[str, Any] | None:
    """
    Wait for orchestrator response to server request.

    Polls response file for new line matching our request.

    Args:
        timeout: Max seconds to wait (default 5.0)

    Returns:
        Response dict or None on timeout
    """
    response_file = Path(".agent_context/server_responses.jsonl")

    # Count existing lines to know where to start reading
    existing_lines = 0
    if response_file.exists():
        with open(response_file, 'r') as f:
            existing_lines = len(f.readlines())

    start_time = time.time()

    while time.time() - start_time < timeout:
        if response_file.exists():
            with open(response_file, 'r') as f:
                lines = f.readlines()

            # Check for new lines
            if len(lines) > existing_lines:
                # Return the newest response
                response_line = lines[-1].strip()
                if response_line:
                    return json.loads(response_line)

        time.sleep(0.1)

    return None


# ----------------------------
# Context Management Tool
# ----------------------------

def mark_subtask_complete(
    success: bool = True,
    reason: str = "",
    context_manager=None
) -> dict[str, Any]:
    """
    Mark current subtask as complete or failed.

    This is called by the agent to signal completion of the current subtask
    and advance to the next one.

    Args:
        success: True if subtask completed successfully, False if failed
        reason: Optional explanation (required if success=False)
        context_manager: ContextManager instance (required)

    Returns:
        Status dict with next action info
    """
    if not context_manager:
        return {"error": "No context manager provided"}

    context_manager.mark_subtask_complete(success, reason)
    _ledger_append("SUBTASK", f"marked {'complete' if success else 'failed'}: {reason}")

    # Log visible progress
    if success:
        current_task = context_manager._get_current_task()
        if current_task:
            completed = sum(1 for st in current_task.subtasks if st.status == "completed")
            total = len(current_task.subtasks)
            print(f"\n{'='*70}")
            print(f"✓ SUBTASK COMPLETE: {reason if reason else 'success'}")
            print(f"Progress: {completed}/{total} subtasks ({completed/total*100:.0f}%)")
            print(f"{'='*70}\n")

    if success:
        # Try to advance to next subtask
        has_next = context_manager.advance_to_next_subtask()
        if not has_next:
            # Check if there are more tasks
            context_manager.state.current_task_idx += 1
            if (context_manager.state.goal and
                context_manager.state.current_task_idx >= len(context_manager.state.goal.tasks)):
                # All tasks complete
                return {"status": "goal_complete", "message": "All tasks finished!"}
            else:
                # Move to next task
                task = context_manager._get_current_task()
                if task and task.subtasks:
                    task.subtasks[0].status = "in_progress"
                    context_manager._save_state()
                    return {
                        "status": "task_advanced",
                        "next_task": task.description,
                        "next_subtask": task.subtasks[0].description
                    }
        else:
            # Advanced to next subtask
            task = context_manager._get_current_task()
            subtask = task.active_subtask() if task else None
            return {
                "status": "subtask_advanced",
                "next_subtask": subtask.description if subtask else None
            }
    else:
        # Failed - stay on current subtask
        return {"status": "failed", "reason": reason}


# ----------------------------
# Tool Definitions for LLM
# ----------------------------

def get_tool_definitions() -> list[dict]:
    """
    Return tool definitions in Ollama function calling format.

    Returns:
        List of tool definition dicts
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write/overwrite a text file. Creates parent directories automatically.",
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
                "description": "Read a text file. Returns content with pagination support for large files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (relative to workspace)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Byte offset to start reading (for large files)"
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
        },
        {
            "type": "function",
            "function": {
                "name": "grep_file",
                "description": "Search for regex pattern in a file. Returns matching lines with context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to search"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Python regex pattern to search for"
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Lines of context before/after match (default 3)"
                        }
                    },
                    "required": ["path", "pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_cmd",
                "description": "Run whitelisted command (python, pytest, ruff, pip). Returns stdout/stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Command as array (e.g., ['python', 'test.py'])"
                        }
                    },
                    "required": ["cmd"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mark_subtask_complete",
                "description": "Mark current subtask as complete and advance to next subtask. Call this when you finish a subtask.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "description": "True if subtask succeeded, False if failed"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation (required if success=False)"
                        }
                    },
                    "required": ["success"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "start_server",
                "description": "Start a background server process (e.g., web server). Returns server info.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Command to run (e.g., ['python', '-m', 'http.server', '8000'])"
                        },
                        "name": {
                            "type": "string",
                            "description": "Optional server name (auto-generated if omitted)"
                        }
                    },
                    "required": ["cmd"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stop_server",
                "description": "Stop a running background server.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "server_id": {
                            "type": "string",
                            "description": "Server identifier (from start_server or list_servers)"
                        }
                    },
                    "required": ["server_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_server",
                "description": "Check server status and get recent logs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "server_id": {
                            "type": "string",
                            "description": "Server identifier"
                        },
                        "tail_lines": {
                            "type": "integer",
                            "description": "Number of recent log lines (default 20)"
                        }
                    },
                    "required": ["server_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_servers",
                "description": "List all running background servers.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]
