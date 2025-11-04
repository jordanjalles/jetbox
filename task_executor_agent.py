"""
TaskExecutor agent - focused on executing specific coding tasks.

This agent is purely config-driven. All logic is in base_agent.py and behaviors.
"""
from __future__ import annotations
from pathlib import Path

from base_agent import BaseAgent


class TaskExecutorAgent(BaseAgent):
    """
    Agent specialized for executing coding tasks.

    This is a minimal wrapper around BaseAgent that passes config to BaseAgent.__init__().
    Everything (role, system_prompt, behaviors) is loaded from task_executor_config.yaml.

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(
        self,
        workspace: Path | str | None = None,
        goal: str | None = None,
    ):
        """
        Initialize TaskExecutor agent.

        Args:
            workspace: Workspace directory (None=.agent_workspaces, Path=specific dir)
            goal: Optional initial goal to set
        """
        # Resolve workspace path - if None, use .agent_workspaces (absolute path)
        if workspace:
            workspace_path = Path(workspace).resolve()
        else:
            # IMPORTANT: resolve() makes this an absolute path, not relative
            # This prevents issues if cwd changes during execution
            workspace_path = Path(".agent_workspaces").resolve()

        super().__init__(
            name="task_executor",
            workspace=workspace_path,
            config_file="task_executor_config.yaml",
        )

        # Set goal if provided (triggers on_goal_set event in SubAgentModeBehavior)
        if goal:
            self.trigger_behavior_event(
                "on_goal_set",
                goal=goal,
                workspace=Path(workspace) if workspace else None,
            )
