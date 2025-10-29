"""
Shared tool implementations for Jetbox agents.

All file-based tools (write_file, read_file, list_dir) are workspace-aware:
they use WorkspaceManager to resolve paths and enforce isolation.

write_file and read_file now accept **kwargs to gracefully handle parameter invention:
- Supported parameters: append, encoding, overwrite, max_size
- Unsupported parameters: ignored with warning (no crashes)
- This prevents agent failures when LLM invents reasonable-sounding parameters

All tools return structured results (strings or dicts) suitable for LLM consumption.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# REMOVED: SAFE_BIN whitelist - replaced with run_bash for full flexibility

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


def read_file(path: str, encoding: str = "utf-8", max_size: int = 1_000_000, **kwargs) -> str:
    """
    Read a text file (workspace-aware).

    For large files, use run_bash with head/tail/sed instead.

    Args:
        path: File path (relative to workspace if set)
        encoding: Text encoding (default: utf-8)
        max_size: Maximum bytes to read (default: 1MB)
        **kwargs: Additional parameters (ignored with warning)

    Returns:
        File contents (up to max_size, truncated if larger)
    """
    global _workspace

    # Warn about unsupported parameters
    if kwargs:
        ignored = ", ".join(kwargs.keys())
        print(f"[tools] read_file ignoring unsupported parameters: {ignored}")

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path)
    else:
        resolved_path = Path(path)

    with open(resolved_path, encoding=encoding, errors="replace") as f:
        content = f.read(max_size)

        file_size = resolved_path.stat().st_size
        if file_size > max_size:
            return content + f"\n\n[TRUNCATED: File is {file_size} bytes, showing first {max_size}. Use run_bash('head -n 100 {path}') or similar for specific sections]"
        return content


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
    """
    Write/overwrite a text file (workspace-aware).

    Args:
        path: File path (relative to workspace if set)
        content: File contents to write
        append: If True, append to file instead of overwriting (default False)
        encoding: Text encoding (default: utf-8)
        create_dirs: Create parent directories if needed (default True)
        overwrite: If False and file exists, return error (default True)
        line_end: Line ending to use ('\\n', '\\r\\n', or None for system default)
        **kwargs: Additional parameters (ignored with warning)

    Returns:
        Success message with path and size
    """
    global _workspace

    # Warn about unsupported parameters (e.g., timeout)
    if kwargs:
        ignored = ", ".join(kwargs.keys())
        print(f"[tools] write_file ignoring unsupported parameters: {ignored}")

    # Normalize line endings if requested
    if line_end is not None:
        # First normalize to \n, then replace with desired ending
        normalized = content.replace('\r\n', '\n').replace('\r', '\n')
        if line_end != '\n':
            content = normalized.replace('\n', line_end)
        else:
            content = normalized

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

    # Check overwrite flag
    if not overwrite and resolved_path.exists():
        error_msg = f"[ERROR] File exists and overwrite=False: {display_path}"
        _ledger_append("ERROR", error_msg)
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
    _ledger_append("WRITE" if not append else "APPEND", str(resolved_path))
    return f"{action} {len(content)} chars to {display_path}"


def run_bash(command: str, timeout: int = 60) -> dict[str, Any]:
    """
    Run any bash command in the workspace.

    Full shell access with pipes, redirection, and command chaining.
    Use this for flexible file operations, testing, linting, etc.

    Args:
        command: Full bash command string (e.g., "grep -r 'pattern' *.py | wc -l")
        timeout: Timeout in seconds (default 60)

    Returns:
        Dict with returncode, stdout, stderr

    Examples:
        run_bash("python script.py")
        run_bash("pytest tests/ -v")
        run_bash("grep -A 3 'class' file.py")
        run_bash("find . -name '*.py' | xargs wc -l")
        run_bash("cat file1.txt file2.txt > combined.txt")
    """
    global _workspace

    # Determine working directory
    cwd = str(_workspace.workspace_dir) if _workspace else None

    # Set up environment with PYTHONPATH for workspace
    env = os.environ.copy()
    if _workspace and cwd:
        env["PYTHONPATH"] = cwd

    try:
        p = subprocess.run(
            command,
            shell=True,  # Enable full shell features
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env
        )
        out = {
            "returncode": p.returncode,
            "stdout": p.stdout[-50_000:],  # Truncate to last 50KB
            "stderr": p.stderr[-50_000:],
        }
        _ledger_append("BASH", f"{command[:100]} -> rc={p.returncode}")
        if p.returncode != 0:
            _ledger_append("ERROR", f"run_bash rc={p.returncode}: {p.stderr[:200]}")
        return out
    except subprocess.TimeoutExpired:
        err = f"Command timed out after {timeout}s"
        _ledger_append("ERROR", f"run_bash timeout: {command[:100]}")
        return {"error": err, "returncode": -1, "stdout": "", "stderr": err}
    except Exception as e:
        err_msg = str(e)
        _ledger_append("ERROR", f"run_bash exception: {err_msg}")
        return {"error": err_msg, "returncode": -1, "stdout": "", "stderr": err_msg}


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
    if not cmd:
        return {"error": "Command cannot be empty"}

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
            # Just completed last subtask of current task
            # Get task info before advancing
            completed_task = context_manager._get_current_task()
            completed_task_desc = completed_task.description if completed_task else "unknown"

            # Check if there are more tasks
            context_manager.state.current_task_idx += 1
            if (context_manager.state.goal and
                context_manager.state.current_task_idx >= len(context_manager.state.goal.tasks)):
                # All tasks complete - goal complete (don't summarize task here, will do goal summary)
                return {"status": "goal_complete", "message": "All tasks finished!"}
            else:
                # More tasks ahead - generate task summary before moving to next
                import jetbox_notes
                task_summary = jetbox_notes.prompt_for_task_summary(completed_task_desc)
                jetbox_notes.append_to_jetbox_notes(task_summary, section="task")

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

def decompose_task(subtasks: list[str], context_manager=None) -> dict[str, Any]:
    """
    Break down the goal/task into subtasks.

    This creates the initial task structure from the goal when no tasks exist yet,
    or adds subtasks to the current task.

    Args:
        subtasks: List of subtask descriptions (strings)
        context_manager: ContextManager instance (required)

    Returns:
        Success dict or error
    """
    from context_manager import Task, Subtask

    if not context_manager:
        return {"error": "No context manager provided"}

    if not context_manager.state.goal:
        return {"error": "No active goal"}

    # Normalize subtasks - handle both list[str] and list[dict]
    normalized_subtasks = []
    for item in subtasks:
        if isinstance(item, dict) and 'description' in item:
            normalized_subtasks.append(item['description'])
        elif isinstance(item, str):
            normalized_subtasks.append(item)
        else:
            return {"error": f"Invalid subtask format: {item}"}

    subtasks = normalized_subtasks  # Use normalized version

    # If no tasks exist yet, create initial task structure
    if not context_manager.state.goal.tasks:
        # Create a single task for the goal
        task = Task(
            description=context_manager.state.goal.description,
            parent_goal=context_manager.state.goal.description
        )

        # Add subtasks to this task
        for subtask_desc in subtasks:
            subtask = Subtask(
                description=subtask_desc,
                depth=0
            )
            task.subtasks.append(subtask)

        # Add task to goal
        context_manager.state.goal.tasks.append(task)
        context_manager.state.current_task_idx = 0

        # Mark first subtask as in_progress
        if task.subtasks:
            task.subtasks[0].status = "in_progress"

        # Save state
        context_manager._save_state()

        _ledger_append("DECOMPOSE", f"Created task with {len(subtasks)} subtasks")
        print(f"\n{'='*70}")
        print(f"🔀 TASK DECOMPOSED")
        print(f"Created 1 task with {len(subtasks)} subtasks:")
        for i, desc in enumerate(subtasks, 1):
            print(f"  {i}. {desc}")
        print(f"{'='*70}\n")

        return {"status": "success", "task_count": 1, "subtask_count": len(subtasks)}
    else:
        # Task already exists - add subtasks to current task
        task = context_manager._get_current_task()
        if not task:
            return {"error": "No active task"}

        # Track how many subtasks existed before
        old_count = len(task.subtasks)

        for subtask_desc in subtasks:
            subtask = Subtask(
                description=subtask_desc,
                depth=0
            )
            task.subtasks.append(subtask)

        # If this was the first subtask added, mark it as in_progress
        if old_count == 0 and task.subtasks:
            task.subtasks[0].status = "in_progress"

        context_manager._save_state()
        _ledger_append("DECOMPOSE", f"Added {len(subtasks)} subtasks to current task")
        print(f"\n{'='*70}")
        print(f"🔀 SUBTASKS ADDED")
        print(f"Added {len(subtasks)} subtasks to current task")
        print(f"{'='*70}\n")

        return {"status": "success", "subtasks_added": len(subtasks)}



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
        },
        {
            "type": "function",
            "function": {
                "name": "run_bash",
                "description": "Run any bash command with full shell features. Use for testing, linting, complex file operations, searching, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Full bash command string (e.g., 'pytest tests/ -v', 'grep -r pattern *.py')"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default 60). Use higher values for slow operations."
                        }
                    },
                    "required": ["command"]
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
                "name": "decompose_task",
                "description": "Break down the goal into tasks and subtasks. Use this when NO TASKS YET to create initial structure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subtasks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of subtask descriptions (e.g., ['Create file structure', 'Write code', 'Run tests'])"
                        }
                    },
                    "required": ["subtasks"]
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
