"""
TaskExecutor agent - focused on executing specific coding tasks.

This agent is purely config-driven. All logic is in base_agent.py and behaviors.
"""
from __future__ import annotations
from pathlib import Path
import os

from base_agent import BaseAgent
from agent_config import config


class TaskExecutorAgent(BaseAgent):
    """
    Agent specialized for executing coding tasks.

    This is a thin wrapper around BaseAgent that:
    1. Loads behaviors from task_executor_config.yaml
    2. Sets the goal via on_goal_set event
    3. Uses generic run() from base_agent

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(
        self,
        workspace: Path | str | None = None,
        goal: str | None = None,
        max_rounds: int = 128,
        model: str | None = None,
        temperature: float = 0.2,
        timeout: int | None = None,
        use_behaviors: bool = True,
        config_file: str = "task_executor_config.yaml",
    ):
        """
        Initialize TaskExecutor agent.

        Args:
            workspace: Workspace directory (None=create new, Path=reuse existing)
            goal: Optional initial goal to set
            max_rounds: Maximum rounds before giving up
            model: Ollama model to use
            temperature: LLM temperature
            timeout: Optional timeout override
            use_behaviors: If True, use behavior system (REQUIRED for new mode)
            config_file: Path to behavior config YAML
        """
        # Determine base workspace for BaseAgent
        base_workspace = Path(workspace) if workspace else Path(".")

        super().__init__(
            name="task_executor",
            role="Code task executor",
            workspace=base_workspace,
            config=config,
        )

        # Store configuration
        self.use_behaviors = use_behaviors
        self.model = model or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        self.temperature = temperature
        self.max_rounds = max_rounds
        self.timeout_override = timeout
        self.goal_start_time = None

        # Store workspace parameter for on_goal_set event
        self._workspace_param = Path(workspace) if workspace else None

        # Initialize context manager
        self.init_context_manager()

        # Initialize perf stats
        self.init_perf_stats()

        # Load behaviors from config
        if use_behaviors:
            print(f"[task_executor] Loading behaviors from {config_file}")
            self.load_behaviors_from_config(config_file)
        else:
            raise ValueError("TaskExecutorAgent requires use_behaviors=True in new architecture")

        # Set goal if provided (triggers on_goal_set event)
        if goal:
            self.trigger_behavior_event(
                "on_goal_set",
                goal=goal,
                workspace=self._workspace_param,
            )

    def get_context_strategy(self) -> str:
        """TaskExecutor uses hierarchical context management (from behavior)."""
        return "hierarchical"
