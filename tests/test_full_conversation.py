"""
Simulate a full conversation with the orchestrator.
"""
from pathlib import Path
from agent_registry import AgentRegistry
from agent_config import config
from orchestrator_main import execute_orchestrator_tool


def simulate_conversation():
    """Simulate multi-turn conversation."""

    workspace = Path.cwd()
    registry = AgentRegistry(config_path="agents.yaml", workspace=workspace)
    orchestrator = registry.get_agent("orchestrator")

    print("=" * 60)
    print("SIMULATED CONVERSATION")
    print("=" * 60)
    print()

    # Turn 1: Initial request
    print("User: make a simple web calculator")
    orchestrator.add_user_message("make a simple web calculator")

    response = orchestrator.execute_round(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )

    display_response(response, registry)

    # Turn 2: Answer clarification
    print("\nUser: just HTML with basic math operations")
    orchestrator.add_user_message("just HTML with basic math operations")

    response = orchestrator.execute_round(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )

    display_response(response, registry)


def display_response(response, registry):
    """Display orchestrator response."""
    if "message" not in response:
        print(f"Error: {response}")
        return

    msg = response["message"]

    # Show content
    if msg.get("content"):
        print(f"\nOrchestrator: {msg['content']}\n")

    # Handle tool calls
    if "tool_calls" in msg:
        for tc in msg["tool_calls"]:
            tool_name = tc["function"]["name"]
            args = tc["function"]["arguments"]

            # Show clarifications
            if tool_name == "clarify_with_user":
                question = args.get("question", "")
                print(f"\nOrchestrator: {question}\n")

            # Show delegations
            elif tool_name == "delegate_to_executor":
                task_desc = args.get("task_description", "")
                print(f"\n→ Delegating to TaskExecutor: {task_desc}\n")

            # Show task plans
            elif tool_name == "create_task_plan":
                tasks = args.get("tasks", [])
                print(f"\n→ Created plan with {len(tasks)} tasks:")
                for i, task in enumerate(tasks, 1):
                    desc = task.get("description", "")
                    print(f"  {i}. {desc}")
                print()

            # Execute tool
            result = execute_orchestrator_tool(tc, registry)


if __name__ == "__main__":
    simulate_conversation()
