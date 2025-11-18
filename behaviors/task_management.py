"""
TaskManagementBehavior - Provides CRUD operations for managing structured task breakdowns.

Used by Orchestrator agents to manage architecture/task-breakdown.json files created
by the Architect. Enables tracking task status, dependencies, and progress.

Now uses @tool decorator for automatic tool registration!"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import json
from datetime import datetime
from behaviors.base import AgentBehavior
from behaviors.tool_decorator import tool
from behaviors.rule_of_two_types import RuleOfTwoProperty


class TaskManagementBehavior(AgentBehavior):
    """
    Provides tools for managing structured task breakdowns in workspace.

    Security: [] None (manages agent-generated task breakdowns, not user data)
    """

    # Rule of Two: Empty (utility behavior for task tracking)
    rule_of_two_properties = set()

    def __init__(self, workspace_manager=None):
        """
        Initialize task management behavior.

        Args:
            workspace_manager: WorkspaceManager instance for file operations
        """
        self.workspace_manager = workspace_manager

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "task_management"

    def _get_task_file(self) -> Path:
        """Get path to task breakdown file."""
        if not self.workspace_manager:
            raise RuntimeError("Task management workspace not configured")

        return self.workspace_manager.workspace_dir / "architecture" / "task-breakdown.json"

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return task management tool definitions.

        Returns:
            List of tool definitions for LLM
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
                    "description": "Mark a task's status (pending, in_progress, completed, failed). Automatically manages timestamps: sets started_at on in_progress, completed_at on completion. Use this to track progress through the task breakdown.",
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
                            },
                            "result": {
                                "type": "string",
                                "description": "Optional result summary when marking completed or failed (e.g., 'Created auth module with JWT support', 'Failed: missing database connection')"
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

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> Any:
        """
        Dispatch task management tool calls.

        Args:
            agent: Agent instance
            tool_name: Tool name (read_task_breakdown, get_next_task, etc.)
            args: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name == "read_task_breakdown":
            return self._read_task_breakdown()
        elif tool_name == "get_next_task":
            skip_dependencies = args.get("skip_dependencies", False)
            return self._get_next_task(skip_dependencies)
        elif tool_name == "mark_task_status":
            return self._mark_task_status(
                task_id=args["task_id"],
                status=args["status"],
                notes=args.get("notes", ""),
                result=args.get("result", "")
            )
        elif tool_name == "update_task":
            return self._update_task(
                task_id=args["task_id"],
                updates=args["updates"]
            )
        else:
            return super().dispatch_tool(agent, tool_name, args)

    def _read_task_breakdown(self) -> dict[str, Any]:
        """
        Read the complete task breakdown from workspace.

        Returns:
            {
                "status": "success",
                "tasks": [...],
                "total_tasks": int,
                "pending_count": int,
                "completed_count": int,
                "failed_count": int
            }
        """
        task_file = self._get_task_file()

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

    def _get_next_task(self, skip_dependencies: bool = False) -> dict[str, Any]:
        """
        Get the next pending task to work on.

        Respects task dependencies unless skip_dependencies=True.

        Args:
            skip_dependencies: If True, return any pending task

        Returns:
            {"status": "success", "task": {...} or None}
        """
        breakdown = self._read_task_breakdown()

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

    def _mark_task_status(
        self,
        task_id: str,
        status: str,
        notes: str = "",
        result: str = ""
    ) -> dict[str, Any]:
        """
        Update task status (pending, in_progress, completed, failed).

        Automatically manages timestamps and attempts.

        Args:
            task_id: Task identifier (e.g., "T1", "T2")
            status: New status
            notes: Optional notes about the status change
            result: Optional result summary

        Returns:
            {"status": "success", "message": "...", "task_id": "...", "new_status": "..."}
        """
        if status not in ["pending", "in_progress", "completed", "failed"]:
            return {
                "status": "error",
                "message": f"Invalid status: {status}. Must be: pending, in_progress, completed, failed"
            }

        task_file = self._get_task_file()

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
                    task["status"] = status
                    task["status_updated_at"] = datetime.now().isoformat()

                    # Handle status-specific updates
                    if status == "in_progress":
                        # Set started_at on first transition to in_progress
                        if not task.get("started_at"):
                            task["started_at"] = datetime.now().isoformat()
                        # Increment attempts
                        task["attempts"] = task.get("attempts", 0) + 1

                    elif status in ["completed", "failed"]:
                        # Set completed_at
                        task["completed_at"] = datetime.now().isoformat()
                        # Store result if provided
                        if result:
                            task["result"] = result

                    # Add notes if provided
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

    def _update_task(
        self,
        task_id: str,
        updates: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Update task properties.

        Args:
            task_id: Task identifier
            updates: Dict of fields to update

        Returns:
            {"status": "success", "message": "...", "updates": {...}}
        """
        task_file = self._get_task_file()

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
