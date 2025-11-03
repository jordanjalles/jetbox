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
    ):
        """
        Initialize Architect agent.

        Args:
            workspace: Working directory (defaults to .agent_workspaces)
            goal: Initial goal/project description (optional)
        """
        super().__init__(
            name="architect",
            workspace=workspace or Path(".agent_workspaces"),
            config_file="architect_config.yaml",
        )

        # Set goal if provided (triggers on_goal_set event in SubAgentModeBehavior)
        if goal:
            self.trigger_behavior_event(
                "on_goal_set",
                goal=goal,
                workspace=workspace,
            )
