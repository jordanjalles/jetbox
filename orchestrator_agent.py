"""
Orchestrator agent - manages user conversation and delegates to other agents.

This agent is purely config-driven. All logic is in base_agent.py and behaviors.
"""
from __future__ import annotations
from pathlib import Path
import os

from base_agent import BaseAgent
from agent_config import config


class OrchestratorAgent(BaseAgent):
    """
    Agent specialized for user interaction and task delegation.

    This is a thin wrapper around BaseAgent that:
    1. Loads behaviors from orchestrator_config.yaml (including DelegationBehavior)
    2. Uses generic run() from base_agent
    3. Delegation tracking handled by DelegationBehavior

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(
        self,
        workspace: Path | None = None,
        use_behaviors: bool = True,
        config_file: str = "orchestrator_config.yaml",
    ):
        """
        Initialize Orchestrator agent.

        Args:
            workspace: Working directory (defaults to current directory)
            use_behaviors: If True, use behavior system (REQUIRED for new mode)
            config_file: Path to behavior config YAML
        """
        # Default workspace to current directory if not provided
        if workspace is None:
            workspace = Path(".")

        super().__init__(
            name="orchestrator",
            role="User interface and task coordinator",
            workspace=workspace,
            config=config,
        )

        # Store configuration
        self.use_behaviors = use_behaviors
        self.model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

        # Initialize context manager
        from context_manager import ContextManager
        self.context_manager = ContextManager()
        self.context_manager.load_or_init("User conversation")

        # Load behaviors from config
        if use_behaviors:
            print(f"[orchestrator] Loading behaviors from {config_file}")
            self.load_behaviors_from_config(config_file)
        else:
            raise ValueError("OrchestratorAgent requires use_behaviors=True in new architecture")

    def get_context_strategy(self) -> str:
        """Orchestrator uses append-until-full strategy (from behavior)."""
        return "append_until_full"

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to conversation.

        Args:
            content: User message content
        """
        self.add_message({"role": "user", "content": content})
