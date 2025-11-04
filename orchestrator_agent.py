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

        # Initialize orchestrator-specific subsystems
        self.init_server_manager()
        self.init_registry()

    def dispatch_tool(self, tool_call: dict[str, Any], **extra_context) -> dict[str, Any]:
        """
        Dispatch tool call with orchestrator context (registry, server_manager).

        Overrides BaseAgent.dispatch_tool() to automatically provide registry
        and server_manager to behaviors that need them (e.g., DelegationBehavior).

        Args:
            tool_call: Tool call dict with function name and arguments
            **extra_context: Additional context (will be merged with orchestrator context)

        Returns:
            Tool result dict
        """
        # Merge orchestrator-specific context with any provided context
        orchestrator_context = {
            "registry": self.registry,
            "server_manager": self.server_manager,
            **extra_context  # Allow override if needed
        }

        # Dispatch with merged context
        return super().dispatch_tool(tool_call, **orchestrator_context)

    def execute_task(self, user_message: str, chatbot_behavior: Any | None = None) -> None:
        """
        Execute a single orchestrator task.

        Args:
            user_message: User message to execute
            chatbot_behavior: Optional ChatbotBehavior instance for completion detection
        """
        # Clean up old server requests
        self.server_manager.cleanup_old_requests()

        # Reset ChatbotBehavior task completion flags
        if chatbot_behavior:
            chatbot_behavior.task_complete_flag = False
            chatbot_behavior.consecutive_empty_rounds = 0

        # Execute task using base_agent's run_task_round_loop
        self.run_task_round_loop(
            user_message=user_message,
            max_rounds=100,
            check_completion_callback=lambda: (
                chatbot_behavior.task_complete_flag if chatbot_behavior else False
            )
        )

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

    @classmethod
    def run_agent(cls, agent: BaseAgent, args: dict[str, Any]) -> None:
        """
        Execute orchestrator with ServerManager and ChatbotBehavior support.

        Args:
            agent: OrchestratorAgent instance
            args: Parsed CLI arguments
        """
        initial_message = args["initial_message"]
        exit_after_initial = args["exit_after_initial"]

        # Get ChatbotBehavior instance for multi-task mode
        chatbot_behavior = None
        for behavior in agent.behaviors:
            if behavior.get_name() == "chatbot":
                chatbot_behavior = behavior
                break

        try:
            # Use ChatbotBehavior's multi-task chat loop if available
            if chatbot_behavior and not exit_after_initial:
                # Multi-task chat mode
                chatbot_behavior.run_multi_task_chat_loop(
                    agent=agent,
                    execute_task_callback=lambda msg: agent.execute_task(msg, chatbot_behavior),
                    initial_message=initial_message
                )
            elif initial_message and exit_after_initial:
                # Single task mode (--once flag)
                print(f"User: {initial_message}\n")
                agent.execute_task(initial_message, chatbot_behavior)
                print("\nTask completed. Exiting...")
            else:
                # Fallback to manual loop if ChatbotBehavior not available
                print("Warning: ChatbotBehavior not found, using fallback loop")
                if initial_message:
                    print(f"User: {initial_message}\n")
                    agent.execute_task(initial_message, chatbot_behavior)
                    if exit_after_initial:
                        print("\nTask completed. Exiting...")
                        return
                    print("\n✅ Task completed. Ready for next request.\n")

                while True:
                    try:
                        user_input = input("You: ").strip()
                        if not user_input:
                            continue
                        if user_input.lower() in ["quit", "exit", "q"]:
                            print("\nShutting down...")
                            break
                        agent.execute_task(user_input, chatbot_behavior)
                        print("\n✅ Task completed. Ready for next request.\n")
                    except (EOFError, KeyboardInterrupt):
                        print("\nShutting down...")
                        break
                    except Exception as e:
                        print(f"\nError: {e}")
                        import traceback
                        traceback.print_exc()

        finally:
            # Clean shutdown
            print("\n[Orchestrator] Stopping all servers...")
            agent.server_manager.stop_all_servers()
            agent.server_manager.stop_monitoring()
            print("Goodbye!")


if __name__ == "__main__":
    OrchestratorAgent.main()
