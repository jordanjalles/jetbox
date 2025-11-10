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
    Everything (role, system_prompt, behaviors) is loaded from config/agents/orchestrator.yaml.

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(self, workspace: Path | None = None, exclude_behaviors: list[str] | None = None, timeout_seconds: int = 600, config_file: str = "config/agents/orchestrator.yaml"):
        """
        Initialize Orchestrator agent.

        Args:
            workspace: Working directory (defaults to .agent_workspaces)
            exclude_behaviors: List of behavior names to exclude (e.g., ["ChatbotBehavior"])
            timeout_seconds: Subprocess timeout in seconds (default: 600 = 10 minutes)
            config_file: Path to agent config file (default: orchestrator.yaml)
        """
        super().__init__(
            name="orchestrator",
            workspace=workspace or Path(".agent_workspaces"),
            config_file=config_file,
            exclude_behaviors=exclude_behaviors,
            timeout_seconds=timeout_seconds,
        )


if __name__ == "__main__":
    OrchestratorAgent.main()
