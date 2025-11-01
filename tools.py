"""
Shared tool implementations for Jetbox agents.

DEPRECATED: This module is deprecated in favor of the composable behaviors system.
All tools have been extracted to behaviors/ directory:
- FileToolsBehavior: write_file, read_file, list_dir
- CommandToolsBehavior: run_bash
- ServerToolsBehavior: start_server, stop_server, check_server, list_servers

This file now acts as a compatibility wrapper that delegates to the behaviors.
Existing code will continue to work, but new code should use behaviors directly.

Migration example:
    OLD:
        import tools
        tools.set_workspace(workspace_manager)
        tools.write_file("test.txt", "content")

    NEW:
        from behaviors import FileToolsBehavior
        file_tools = FileToolsBehavior(workspace_manager=workspace_manager)
        file_tools.dispatch_tool("write_file", {"path": "test.txt", "content": "content"})
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

# Import behaviors for delegation
from behaviors import FileToolsBehavior, CommandToolsBehavior, ServerToolsBehavior

# Global references set by agent at runtime (for backward compatibility)
_workspace = None  # WorkspaceManager instance
_ledger_file = None  # Path to ledger file for audit trail

# Global behavior instances (created when set_workspace/set_ledger called)
_file_tools: FileToolsBehavior | None = None
_command_tools: CommandToolsBehavior | None = None
_server_tools: ServerToolsBehavior | None = None


def set_workspace(workspace_manager) -> None:
    """
    Set the workspace manager for path resolution.

    DEPRECATED: This function is maintained for backward compatibility.
    New code should instantiate behaviors directly.
    """
    global _workspace, _file_tools, _command_tools, _server_tools
    _workspace = workspace_manager

    # Create behavior instances with workspace
    _file_tools = FileToolsBehavior(
        workspace_manager=workspace_manager,
        ledger_file=_ledger_file
    )
    _command_tools = CommandToolsBehavior(
        workspace_manager=workspace_manager,
        ledger_file=_ledger_file
    )
    _server_tools = ServerToolsBehavior(
        workspace_manager=workspace_manager,
        ledger_file=_ledger_file
    )


def set_ledger(ledger_path: Path) -> None:
    """
    Set the ledger file path for audit logging.

    DEPRECATED: This function is maintained for backward compatibility.
    New code should instantiate behaviors directly.
    """
    global _ledger_file, _file_tools, _command_tools, _server_tools
    _ledger_file = ledger_path

    # Update behavior instances if they exist
    if _file_tools:
        _file_tools.ledger_file = ledger_path
    if _command_tools:
        _command_tools.ledger_file = ledger_path
    if _server_tools:
        _server_tools.ledger_file = ledger_path


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
# File Operation Tools (Compatibility Wrappers)
# ----------------------------

def list_dir(path: str | None = ".", **kwargs) -> list[str]:
    """
    List files in directory (non-recursive, workspace-aware).

    DEPRECATED: Delegates to FileToolsBehavior. Use behaviors directly in new code.

    Args:
        path: Directory path (relative to workspace if set)

    Returns:
        Sorted list of filenames, or error message list
    """
    global _file_tools

    # Create behavior if not initialized
    if not _file_tools:
        _file_tools = FileToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)

    return _file_tools.dispatch_tool("list_dir", {"path": path or ".", **kwargs})


def read_file(path: str, encoding: str = "utf-8", max_size: int = 1_000_000, **kwargs) -> str:
    """
    Read a text file (workspace-aware).

    DEPRECATED: Delegates to FileToolsBehavior. Use behaviors directly in new code.

    Args:
        path: File path (relative to workspace if set)
        encoding: Text encoding (default: utf-8)
        max_size: Maximum bytes to read (default: 1MB)
        **kwargs: Additional parameters (ignored with warning)

    Returns:
        File contents (up to max_size, truncated if larger)
    """
    global _file_tools

    # Create behavior if not initialized
    if not _file_tools:
        _file_tools = FileToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)

    return _file_tools.dispatch_tool("read_file", {
        "path": path,
        "encoding": encoding,
        "max_size": max_size,
        **kwargs
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
    """
    Write/overwrite a text file (workspace-aware).

    DEPRECATED: Delegates to FileToolsBehavior. Use behaviors directly in new code.

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
    global _file_tools

    # Create behavior if not initialized
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
    """
    Run any bash command in the workspace.

    DEPRECATED: Delegates to CommandToolsBehavior.

    Full shell access with pipes, redirection, and command chaining.
    """
    global _command_tools
    if not _command_tools:
        _command_tools = CommandToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _command_tools.dispatch_tool("run_bash", {"command": command, "timeout": timeout})

# ----------------------------
# Server Management Tools
# ----------------------------

def start_server(cmd: list[str], name: str = None) -> dict[str, Any]:
    """
    Request orchestrator to start a server in the background.
    
    DEPRECATED: Delegates to ServerToolsBehavior.

    Args:
        cmd: Command to run (e.g., ['python', '-m', 'http.server', '8000'])
        name: Optional server name (auto-generated if not provided)

    Returns:
        Server info dict or error dict
    """
    global _server_tools
    if not _server_tools:
        _server_tools = ServerToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _server_tools.dispatch_tool("start_server", {"cmd": cmd, "name": name})


def stop_server(server_id: str) -> dict[str, Any]:
    """Request orchestrator to stop a server. DEPRECATED: Delegates to ServerToolsBehavior."""
    global _server_tools
    if not _server_tools:
        _server_tools = ServerToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _server_tools.dispatch_tool("stop_server", {"server_id": server_id})


def check_server(server_id: str, tail_lines: int = 20) -> dict[str, Any]:
    """Check server status. DEPRECATED: Delegates to ServerToolsBehavior."""
    global _server_tools
    if not _server_tools:
        _server_tools = ServerToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _server_tools.dispatch_tool("check_server", {"server_id": server_id, "tail_lines": tail_lines})


def list_servers() -> dict[str, Any]:
    """List servers. DEPRECATED: Delegates to ServerToolsBehavior."""
    global _server_tools
    if not _server_tools:
        _server_tools = ServerToolsBehavior(workspace_manager=_workspace, ledger_file=_ledger_file)
    return _server_tools.dispatch_tool("list_servers", {})




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

def mark_goal_complete(summary: str = "", context_manager=None) -> dict[str, Any]:
    """
    Mark the entire goal as complete (for non-hierarchical strategies).

    This is used by simple strategies like append-until-full that don't
    decompose into tasks/subtasks. The agent just does the work and
    signals completion when done.

    Args:
        summary: Brief summary of what was accomplished
        context_manager: ContextManager instance (optional, not used but kept for consistency)

    Returns:
        Status dict indicating goal completion
    """
    _ledger_append("GOAL", f"completed: {summary}")

    print(f"\n{'='*70}")
    print(f"✓ GOAL COMPLETE: {summary}")
    print(f"{'='*70}\n")

    return {"status": "goal_complete", "message": "Goal completed!", "summary": summary}

def mark_complete(summary: str = "", context_manager=None) -> dict[str, Any]:
    """
    Mark delegated task as complete (for SubAgentStrategy).

    This is used by sub-agents working on delegated tasks to signal
    successful completion back to the controlling agent.

    Args:
        summary: Brief summary of what was accomplished (2-4 sentences)
        context_manager: ContextManager instance (optional, for consistency)

    Returns:
        Status dict indicating task completion with summary for controlling agent
    """
    _ledger_append("DELEGATED_TASK", f"completed: {summary}")

    print(f"\n{'='*70}")
    print(f"✓ DELEGATED TASK COMPLETE")
    print(f"{'='*70}")
    print(f"Summary: {summary}")
    print(f"{'='*70}\n")

    # Return goal_complete status to trigger agent exit
    # Include summary for controlling agent
    return {
        "status": "goal_complete",
        "message": "Delegated task completed successfully",
        "summary": summary,
        "success": True
    }

def mark_failed(reason: str = "", context_manager=None) -> dict[str, Any]:
    """
    Mark delegated task as failed (for SubAgentStrategy).

    This is used by sub-agents working on delegated tasks to signal
    failure back to the controlling agent.

    Args:
        reason: Explanation of why the task could not be completed
        context_manager: ContextManager instance (optional, for consistency)

    Returns:
        Status dict indicating task failure with reason for controlling agent
    """
    _ledger_append("DELEGATED_TASK", f"failed: {reason}")

    print(f"\n{'='*70}")
    print(f"✗ DELEGATED TASK FAILED")
    print(f"{'='*70}")
    print(f"Reason: {reason}")
    print(f"{'='*70}\n")

    # Return goal_complete status to trigger agent exit
    # Include failure flag and reason for controlling agent
    return {
        "status": "goal_complete",
        "message": "Delegated task failed",
        "reason": reason,
        "success": False
    }

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
