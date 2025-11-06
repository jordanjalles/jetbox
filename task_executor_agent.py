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
        timeout_seconds: int = 600,
        exclude_behaviors: list[str] | None = None,
    ):
        """
        Initialize TaskExecutor agent.

        Args:
            workspace: Workspace directory (None=.agent_workspaces, Path=specific dir)
            goal: Optional initial goal to set
            timeout_seconds: Subprocess timeout in seconds (default: 600 = 10 minutes)
            exclude_behaviors: List of behavior names to exclude (e.g., ["ChatbotBehavior"])
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
            timeout_seconds=timeout_seconds,
            exclude_behaviors=exclude_behaviors,
        )

        # Set goal if provided (triggers onGoalSet event in SubAgentModeBehavior)
        if goal:
            self.trigger_behavior_event(
                "onGoalSet",
                goal=goal,
                workspace=Path(workspace) if workspace else None,
            )


if __name__ == "__main__":
    TaskExecutorAgent.main()
