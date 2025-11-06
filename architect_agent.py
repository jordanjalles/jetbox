"""
Software Architect Agent - Architecture design and planning consultant.

This agent is purely config-driven. All logic is in base_agent.py and behaviors.
"""
from __future__ import annotations
from pathlib import Path

from base_agent import BaseAgent


class ArchitectAgent(BaseAgent):
    """
    Agent specialized for architecture design and planning.

    This is a minimal wrapper around BaseAgent that passes config to BaseAgent.__init__().
    Everything (role, system_prompt, behaviors) is loaded from architect_config.yaml.

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(
        self,
        workspace: Path | None = None,
        goal: str | None = None,
        timeout_seconds: int = 600,
        exclude_behaviors: list[str] | None = None,
    ):
        """
        Initialize Architect agent.

        Args:
            workspace: Working directory (defaults to .agent_workspaces)
            goal: Initial goal/project description (optional)
            timeout_seconds: Subprocess timeout in seconds (default: 600 = 10 minutes)
            exclude_behaviors: List of behavior names to exclude (e.g., ["ChatbotBehavior"])
        """
        super().__init__(
            name="architect",
            workspace=workspace or Path(".agent_workspaces"),
            config_file="architect_config.yaml",
            timeout_seconds=timeout_seconds,
            exclude_behaviors=exclude_behaviors,
        )

        # Set goal if provided (triggers onGoalSet event in BaseAgent)
        if goal:
            self.trigger_behavior_event(
                "onGoalSet",
                goal=goal,
                workspace=workspace,
            )


if __name__ == "__main__":
    ArchitectAgent.main()
