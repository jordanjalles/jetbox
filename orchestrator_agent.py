"""
Orchestrator agent - manages user conversation and delegates to other agents.

This agent is purely config-driven. All logic is in base_agent.py and behaviors.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json

from base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    """
    Agent specialized for user interaction and task delegation.

    This is a minimal wrapper around BaseAgent that passes config to BaseAgent.__init__().
    Everything (role, system_prompt, behaviors) is loaded from orchestrator_config.yaml.

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(self, workspace: Path | None = None, exclude_behaviors: list[str] | None = None):
        """
        Initialize Orchestrator agent.

        Args:
            workspace: Working directory (defaults to .agent_workspaces)
            exclude_behaviors: List of behavior names to exclude (e.g., ["ChatbotBehavior"])
        """
        super().__init__(
            name="orchestrator",
            workspace=workspace or Path(".agent_workspaces"),
            config_file="orchestrator_config.yaml",
            exclude_behaviors=exclude_behaviors,
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

        return cls(workspace=workspace, exclude_behaviors=exclude_behaviors)

    @classmethod
    def run_agent(cls, agent: BaseAgent, args: dict[str, Any]) -> None:
        """
        Execute orchestrator with ServerManager and ChatbotBehavior support.

        Args:
            agent: OrchestratorAgent instance
            args: Parsed CLI arguments
        """
        from server_manager import ServerManager
        from agent_registry import AgentRegistry

        initial_message = args["initial_message"]
        exit_after_initial = args["exit_after_initial"]

        # Initialize ServerManager
        server_manager = ServerManager(agent.workspace)
        server_manager.start_monitoring()

        # Initialize agent registry
        registry = AgentRegistry(config_path="agents.yaml", workspace=agent.workspace)

        # Get ChatbotBehavior instance for task completion detection
        chatbot_behavior = None
        for behavior in agent.behaviors:
            if behavior.get_name() == "chatbot":
                chatbot_behavior = behavior
                break

        # Define task execution callback for ChatbotBehavior
        def execute_task(user_message: str) -> None:
            """
            Execute a single orchestrator task.

            This function is called by ChatbotBehavior for each user message.
            It runs the orchestrator's LLM loop until the task completes.
            """
            # Clean up old server requests
            server_manager.cleanup_old_requests()

            # Add user message to history
            agent.add_message({"role": "user", "content": user_message})

            # Reset task_complete_flag for new task
            if chatbot_behavior:
                chatbot_behavior.task_complete_flag = False
                chatbot_behavior.consecutive_empty_rounds = 0

            # Execute rounds until task complete
            round_num = 0
            max_rounds = 100

            while True:
                round_num += 1

                # Check if ChatbotBehavior detected task completion (2 consecutive empty rounds)
                if chatbot_behavior and chatbot_behavior.task_complete_flag:
                    print("[orchestrator] Task complete (detected by ChatbotBehavior), returning to prompt")
                    break

                response = agent._execute_round(
                    round_no=round_num,
                    max_rounds=max_rounds,
                    model=agent.config.llm.model,
                    temperature=agent.config.llm.temperature,
                )

                # Check if goal completed/failed
                if response is None:
                    continue

                # Display response
                if "message" in response:
                    msg = response["message"]

                    if isinstance(msg, dict):
                        if msg.get("content"):
                            print(f"Orchestrator: {msg['content']}")
                            print()
                    elif isinstance(msg, str):
                        if msg:
                            print(f"Orchestrator: {msg}")
                            print()

                    # Execute tool calls
                    if isinstance(msg, dict) and "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            tool_name = tc["function"]["name"]
                            tool_args = tc["function"]["arguments"]

                            # Show delegation events
                            if tool_name == "clarify_with_user":
                                print(f"Orchestrator: {tool_args.get('question', '')}\n")
                            elif tool_name == "consult_architect":
                                print(f"→ Consulting Architect: {tool_args.get('project_description', '')[:60]}...\n")
                            elif tool_name == "delegate_to_executor":
                                print(f"→ Delegating to TaskExecutor: {tool_args.get('task_description', '')[:60]}...\n")

                            # Dispatch tool via behavior system with extra context
                            result = agent.dispatch_tool(
                                tc,
                                registry=registry,
                                server_manager=server_manager
                            )

                            # Add tool result
                            agent.add_message({
                                "role": "tool",
                                "content": json.dumps(result),
                            })

                # Check max rounds
                if round_num >= max_rounds:
                    print("[orchestrator] Max rounds reached")
                    break

        try:
            # Use ChatbotBehavior's multi-task chat loop if available
            if chatbot_behavior and not exit_after_initial:
                # Multi-task chat mode
                chatbot_behavior.run_multi_task_chat_loop(
                    agent=agent,
                    execute_task_callback=execute_task,
                    initial_message=initial_message
                )
            elif initial_message and exit_after_initial:
                # Single task mode (--once flag)
                print(f"User: {initial_message}\n")
                execute_task(initial_message)
                print("\nTask completed. Exiting...")
            else:
                # Fallback to manual loop if ChatbotBehavior not available
                print("Warning: ChatbotBehavior not found, using fallback loop")
                if initial_message:
                    print(f"User: {initial_message}\n")
                    execute_task(initial_message)
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
                        execute_task(user_input)
                        print("\n✅ Task completed. Ready for next request.\n")
                    except KeyboardInterrupt:
                        print("\n\nInterrupted. Shutting down...")
                        break
                    except Exception as e:
                        print(f"\nError: {e}")
                        import traceback
                        traceback.print_exc()

        finally:
            # Clean shutdown
            print("\n[Orchestrator] Stopping all servers...")
            server_manager.stop_all_servers()
            server_manager.stop_monitoring()
            print("Goodbye!")


if __name__ == "__main__":
    OrchestratorAgent.main()
