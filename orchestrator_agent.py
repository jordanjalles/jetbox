"""
Orchestrator agent - manages user conversation and delegates to other agents.

This agent is purely config-driven. All logic is in base_agent.py and behaviors.
"""
from __future__ import annotations
from pathlib import Path

from base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    """
    Agent specialized for user interaction and task delegation.

    This is a minimal wrapper around BaseAgent that passes config to BaseAgent.__init__().
    Everything (role, system_prompt, behaviors) is loaded from orchestrator_config.yaml.

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(self, workspace: Path | None = None):
        """
        Initialize Orchestrator agent.

        Args:
            workspace: Working directory (defaults to current directory)
        """
        super().__init__(
            name="orchestrator",
            workspace=workspace or Path("."),
            config_file="orchestrator_config.yaml",
        )
