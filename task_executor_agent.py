"""
TaskExecutor agent - focused on executing specific coding tasks.

Uses hierarchical context management (Goal → Task → Subtask → Action).
Keeps last N message exchanges to stay focused on current work.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path

from base_agent import BaseAgent
from context_manager import ContextManager
from agent_config import config
from context_strategies import build_simple_hierarchical_context


class TaskExecutorAgent(BaseAgent):
    """
    Agent specialized for executing coding tasks.

    Context strategy: Hierarchical (Goal/Task/Subtask tree)
    Tools: File operations, command execution, task completion markers
    """

    def __init__(
        self,
        workspace: Path,
        goal: str | None = None,
    ):
        """
        Initialize TaskExecutor agent.

        Args:
            workspace: Working directory for task execution
            goal: Optional initial goal to set
        """
        super().__init__(
            name="task_executor",
            role="Code task executor",
            workspace=workspace,
            config=config,
        )

        # Initialize context manager for hierarchical task tracking
        # Note: ContextManager uses hardcoded .agent_context directory
        self.context_manager = ContextManager()

        # Set initial goal if provided
        if goal:
            self.context_manager.load_or_init(goal)

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tools available to TaskExecutor.

        Tools:
        - write_file: Write content to a file
        - read_file: Read file contents
        - list_dir: List directory contents
        - run_cmd: Execute shell commands (whitelisted)
        - mark_subtask_complete: Mark current subtask as done
        - decompose_task: Break task into subtasks
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates parent dirs if needed)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to workspace"},
                            "content": {"type": "string", "description": "File content to write"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to workspace"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List directory contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path (default: workspace root)"},
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_cmd",
                    "description": "Run shell command (only python, pytest, ruff, pip allowed)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to execute"},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_subtask_complete",
                    "description": "Mark current subtask as complete (success or failure)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "description": "True if subtask completed successfully"},
                            "reason": {"type": "string", "description": "Reason for failure (if success=False)"},
                        },
                        "required": ["success"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "decompose_task",
                    "description": "Break current task into smaller subtasks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtasks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of subtask descriptions",
                            },
                        },
                        "required": ["subtasks"],
                    },
                },
            },
        ]

    def get_system_prompt(self) -> str:
        """Return system prompt from config."""
        return config.llm.system_prompt

    def get_context_strategy(self) -> str:
        """TaskExecutor uses hierarchical context management."""
        return "hierarchical"

    def build_context(self) -> list[dict[str, Any]]:
        """
        Build context using hierarchical strategy.

        Returns:
            [system_prompt, task_info, last_N_messages]
        """
        # Use shared hierarchical context strategy
        return build_simple_hierarchical_context(
            context_manager=self.context_manager,
            messages=self.state.messages,
            system_prompt=self.get_system_prompt(),
            config=self.config,
        )

    def execute_round(self, model: str, temperature: float) -> dict[str, Any]:
        """
        Execute one round of task execution.

        Returns:
            LLM response with tool calls
        """
        self.increment_round()
        response = self.call_llm(model, temperature)

        # Add assistant message to history
        if "message" in response:
            self.add_message(response["message"])

        return response

    def set_goal(self, goal: str) -> None:
        """Set a new goal for the task executor."""
        self.context_manager.load_or_init(goal)

    def get_current_status(self) -> dict[str, Any]:
        """
        Get current task execution status.

        Returns:
            Dict with goal, task, subtask info
        """
        goal = self.context_manager.state.goal
        if not goal:
            return {"status": "idle", "goal": None}

        task = self.context_manager._get_current_task()
        subtask = task.active_subtask() if task else None

        return {
            "status": "active",
            "goal": goal.description,
            "task": task.description if task else None,
            "subtask": subtask.description if subtask else None,
            "rounds": self.state.total_rounds,
        }
