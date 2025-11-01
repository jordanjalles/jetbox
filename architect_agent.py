"""
Software Architect Agent - Architecture design and planning consultant.

This agent is purely config-driven. All logic is in base_agent.py and behaviors.
"""
from __future__ import annotations
from pathlib import Path

from base_agent import BaseAgent
from agent_config import config


class ArchitectAgent(BaseAgent):
    """
    Agent specialized for architecture design and planning.

    This is a thin wrapper around BaseAgent that:
    1. Loads behaviors from architect_config.yaml (including ArchitectToolsBehavior)
    2. Uses generic run() from base_agent
    3. Architecture workflow handled by ArchitectToolsBehavior

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(
        self,
        workspace: Path | None = None,
        project_description: str = "",
        use_behaviors: bool = True,
        config_file: str = "architect_config.yaml",
    ):
        """
        Initialize Architect agent.

        Args:
            workspace: Working directory (defaults to current directory)
            project_description: Initial project description (optional)
            use_behaviors: If True, use behavior system (REQUIRED for new mode)
            config_file: Path to behavior config YAML
        """
        # Default workspace to current directory if not provided
        if workspace is None:
            workspace = Path(".")

        super().__init__(
            name="architect",
            role="Software architecture consultant",
            workspace=workspace,
            config=config,
        )

        # Store configuration
        self.use_behaviors = use_behaviors

        # Initialize context manager
        from context_manager import ContextManager
        self.context_manager = ContextManager()

        # Load behaviors from config
        if use_behaviors:
            print(f"[architect] Loading behaviors from {config_file}")
            self.load_behaviors_from_config(config_file)
        else:
            raise ValueError("ArchitectAgent requires use_behaviors=True in new architecture")

        # Set project description if provided
        if project_description:
            self.context_manager.load_or_init(project_description)

    def get_context_strategy(self) -> str:
        """Architect uses append-until-full strategy (from behavior)."""
        return "append_until_full"
