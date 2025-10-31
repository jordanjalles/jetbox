"""
Task Management Tools

Provides CRUD operations for managing structured tasks in architecture/task-breakdown.json.
Used by Orchestrator and Architect agents via TaskManagementStrategy.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import json
from datetime import datetime


# Global workspace manager (set by agent using this strategy)
_workspace_manager = None


def set_workspace(workspace_manager) -> None:
    """Set the workspace manager for task management tools."""
    global _workspace_manager
    _workspace_manager = workspace_manager


def _get_task_file() -> Path:
    """Get path to task breakdown file."""
    if not _workspace_manager:
        raise RuntimeError("Task management workspace not configured")

    return _workspace_manager.workspace_dir / "architecture" / "task-breakdown.json"


def read_task_breakdown() -> dict[str, Any]:
    """
    Read the complete task breakdown from workspace.

    Returns:
        {
            "status": "success",
            "tasks": [...],  # Full task list
            "total_tasks": int,
            "pending_count": int,
            "completed_count": int,
            "failed_count": int
        }
    """
    task_file = _get_task_file()

    if not task_file.exists():
        return {
            "status": "success",
            "tasks": [],
            "total_tasks": 0,
            "pending_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "message": "No task breakdown file found"
        }

    try:
        with open(task_file) as f:
            data = json.load(f)

        tasks = data.get("tasks", [])

        # Count task statuses
        pending = sum(1 for t in tasks if t.get("status", "pending") == "pending")
        completed = sum(1 for t in tasks if t.get("status") == "completed")
        failed = sum(1 for t in tasks if t.get("status") == "failed")
        in_progress = sum(1 for t in tasks if t.get("status") == "in_progress")

        return {
            "status": "success",
            "tasks": tasks,
            "total_tasks": len(tasks),
            "pending_count": pending,
            "completed_count": completed,
            "failed_count": failed,
            "in_progress_count": in_progress,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read task breakdown: {e}"
        }


def get_next_task(skip_dependencies: bool = False) -> dict[str, Any]:
    """
    Get the next pending task to work on.

    Respects task dependencies unless skip_dependencies=True.

    Args:
        skip_dependencies: If True, return any pending task. If False, only return
                          tasks whose dependencies are completed.

    Returns:
        {
            "status": "success",
            "task": {...},  # Next task to work on
            "message": "..."
        }
    """
    breakdown = read_task_breakdown()

    if breakdown["status"] != "success":
        return breakdown

    tasks = breakdown["tasks"]

    if not tasks:
        return {
            "status": "success",
            "task": None,
            "message": "No tasks in breakdown"
        }

    # Find completed task IDs
    completed_ids = {t["id"] for t in tasks if t.get("status") == "completed"}

    # Find next pending task
    for task in tasks:
        status = task.get("status", "pending")

        if status != "pending":
            continue

        # Check dependencies
        if not skip_dependencies:
            dependencies = task.get("dependencies", [])
            if dependencies and not all(dep in completed_ids for dep in dependencies):
                continue  # Dependencies not met

        return {
            "status": "success",
            "task": task,
            "message": f"Next task: {task['id']} - {task['description']}"
        }

    # No pending tasks found
    return {
        "status": "success",
        "task": None,
        "message": "No pending tasks (all completed, failed, or blocked by dependencies)"
    }


def mark_task_status(
    task_id: str,
    status: str,
    notes: str = ""
) -> dict[str, Any]:
    """
    Update task status (pending, in_progress, completed, failed).

    Args:
        task_id: Task identifier (e.g., "T1", "T2")
        status: New status - one of: "pending", "in_progress", "completed", "failed"
        notes: Optional notes about the status change

    Returns:
        {
            "status": "success",
            "message": "Task T1 marked as completed"
        }
    """
    if status not in ["pending", "in_progress", "completed", "failed"]:
        return {
            "status": "error",
            "message": f"Invalid status: {status}. Must be: pending, in_progress, completed, failed"
        }

    task_file = _get_task_file()

    if not task_file.exists():
        return {
            "status": "error",
            "message": "No task breakdown file found. Create tasks first."
        }

    try:
        # Read current breakdown
        with open(task_file) as f:
            data = json.load(f)

        # Find and update task
        task_found = False
        for task in data.get("tasks", []):
            if task["id"] == task_id:
                old_status = task.get("status", "pending")
                task["status"] = status
                task["status_updated_at"] = datetime.now().isoformat()

                if notes:
                    if "notes" not in task:
                        task["notes"] = []
                    task["notes"].append({
                        "timestamp": datetime.now().isoformat(),
                        "note": notes
                    })

                task_found = True
                break

        if not task_found:
            return {
                "status": "error",
                "message": f"Task {task_id} not found in breakdown"
            }

        # Write updated breakdown
        with open(task_file, "w") as f:
            json.dump(data, f, indent=2)

        return {
            "status": "success",
            "message": f"Task {task_id} marked as {status}",
            "task_id": task_id,
            "new_status": status
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to update task status: {e}"
        }


def update_task(
    task_id: str,
    updates: dict[str, Any]
) -> dict[str, Any]:
    """
    Update task properties (description, priority, complexity, etc.).

    Args:
        task_id: Task identifier
        updates: Dict of fields to update (e.g., {"priority": 1, "estimated_complexity": "high"})

    Returns:
        {
            "status": "success",
            "message": "Task T1 updated"
        }
    """
    task_file = _get_task_file()

    if not task_file.exists():
        return {
            "status": "error",
            "message": "No task breakdown file found"
        }

    try:
        with open(task_file) as f:
            data = json.load(f)

        task_found = False
        for task in data.get("tasks", []):
            if task["id"] == task_id:
                # Update allowed fields
                for key, value in updates.items():
                    if key not in ["id", "status"]:  # Protect critical fields
                        task[key] = value

                task["updated_at"] = datetime.now().isoformat()
                task_found = True
                break

        if not task_found:
            return {
                "status": "error",
                "message": f"Task {task_id} not found"
            }

        with open(task_file, "w") as f:
            json.dump(data, f, indent=2)

        return {
            "status": "success",
            "message": f"Task {task_id} updated",
            "updates": updates
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to update task: {e}"
        }


def get_task_management_tool_definitions() -> list[dict[str, Any]]:
    """
    Get tool definitions for task management.

    Returns tools that can be injected into agent tool list.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_task_breakdown",
                "description": "Read the complete task breakdown with status counts. Returns all tasks with their current status (pending/in_progress/completed/failed).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_next_task",
                "description": "Get the next pending task to work on. Respects task dependencies by default (only returns tasks whose dependencies are completed). Returns None if no tasks are ready.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skip_dependencies": {
                            "type": "boolean",
                            "description": "If true, return any pending task regardless of dependencies. Default: false"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mark_task_status",
                "description": "Mark a task's status (pending, in_progress, completed, failed). Use this to track progress through the task breakdown.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID (e.g., 'T1', 'T2')"
                        },
                        "status": {
                            "type": "string",
                            "description": "New status: 'pending', 'in_progress', 'completed', or 'failed'",
                            "enum": ["pending", "in_progress", "completed", "failed"]
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional notes about the status change"
                        }
                    },
                    "required": ["task_id", "status"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_task",
                "description": "Update task properties like description, priority, complexity, or dependencies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to update"
                        },
                        "updates": {
                            "type": "object",
                            "description": "Fields to update (e.g., {'priority': 1, 'estimated_complexity': 'high'})"
                        }
                    },
                    "required": ["task_id", "updates"]
                }
            }
        }
    ]
