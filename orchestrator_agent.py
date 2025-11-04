"""
Orchestrator agent - manages user conversation and delegates to other agents.

This agent is purely config-driven. All logic is in base_agent.py and behaviors.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

from base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    """
    Agent specialized for user interaction and task delegation.

    This is a minimal wrapper around BaseAgent that passes config to BaseAgent.__init__().
    Everything (role, system_prompt, behaviors) is loaded from orchestrator_config.yaml.

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(self, workspace: Path | None = None, exclude_behaviors: list[str] | None = None, timeout_seconds: int = 600):
        """
        Initialize Orchestrator agent.

        Args:
            workspace: Working directory (defaults to .agent_workspaces)
            exclude_behaviors: List of behavior names to exclude (e.g., ["ChatbotBehavior"])
            timeout_seconds: Subprocess timeout in seconds (default: 600 = 10 minutes)
        """
        super().__init__(
            name="orchestrator",
            workspace=workspace or Path(".agent_workspaces"),
            config_file="orchestrator_config.yaml",
            exclude_behaviors=exclude_behaviors,
            timeout_seconds=timeout_seconds,
        )

    def pre_task_hook(self) -> None:
        """
        Hook called before each task in multi-task chat mode.

        Used by base_agent._run_multi_task_chat_mode().
        """
        # Clean up old server requests before starting new task
        if self.server_manager:
            self.server_manager.cleanup_old_requests()

    def cleanup_hook(self) -> None:
        """
        Hook called at end of agent execution for cleanup.

        Used by base_agent._run_multi_task_chat_mode().
        """
        # Stop all servers and monitoring
        if self.server_manager:
            print("\n[Orchestrator] Stopping all servers...")
            self.server_manager.stop_all_servers()
            self.server_manager.stop_monitoring()
            print("Goodbye!")

    # ===========================
    # CLI customization
    # ===========================

    @classmethod
    def create_agent_instance(cls, workspace: Path, args: dict[str, Any]):
        """
        Create orchestrator agent with conditional ChatbotBehavior exclusion.

        Args:
            workspace: Workspace directory path
            args: Parsed CLI arguments

        Returns:
            OrchestratorAgent instance
        """
        initial_message = args["initial_message"]
        force_chat_mode = args["force_chat_mode"]
        timeout_seconds = args.get("timeout_seconds", 600)

        # Determine if ChatbotBehavior should be excluded
        # Exclude it when goal string is provided UNLESS --chat flag is set
        # Include it when no goal string (interactive mode) OR --chat flag
        exclude_behaviors = []
        if initial_message and not force_chat_mode:
            # Autonomous mode: exclude chatbot behavior to prevent conversational mode
            exclude_behaviors = ["ChatbotBehavior"]
            print("[OrchestratorAgent] Autonomous mode: ChatbotBehavior excluded")
        else:
            # Interactive mode or chat mode: include chatbot behavior for user interaction
            if force_chat_mode:
                print("[OrchestratorAgent] Chat mode (--chat): ChatbotBehavior enabled")
            else:
                print("[OrchestratorAgent] Interactive mode: ChatbotBehavior enabled")

        return cls(workspace=workspace, exclude_behaviors=exclude_behaviors, timeout_seconds=timeout_seconds)


if __name__ == "__main__":
    OrchestratorAgent.main()
