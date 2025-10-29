"""
Test orchestrator with simulated user responses.
"""
from pathlib import Path
from agent_registry import AgentRegistry
from agent_config import config
from orchestrator_main import execute_orchestrator_tool
import time


def test_live_orchestrator():
    """Test orchestrator with realistic conversation."""

    workspace = Path.cwd()
    registry = AgentRegistry(config_path="agents.yaml", workspace=workspace)
    orchestrator = registry.get_agent("orchestrator")

    print("=" * 60)
    print("TESTING ORCHESTRATOR LIVE")
    print("=" * 60)
    print()

    # Conversation 1: Request something simple
    user_input = "make a simple HTML calculator"
    print(f"User: {user_input}\n")
    orchestrator.add_user_message(user_input)

    # Get response
    response = orchestrator.execute_round(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )

    handle_response(response, registry, orchestrator)

    # Conversation 2: Answer clarifications
    user_input = "just basic operations, single HTML file, simple styling"
    print(f"\nUser: {user_input}\n")
    orchestrator.add_user_message(user_input)

    # Get response
    response = orchestrator.execute_round(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )

    handle_response(response, registry, orchestrator)

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


def handle_response(response, registry, orchestrator):
    """Handle and display orchestrator response."""
    if "message" not in response:
        print(f"Error: {response}")
        return

    msg = response["message"]

    # Show content
    if msg.get("content"):
        print(f"Orchestrator: {msg['content']}\n")

    # Handle tool calls
    if "tool_calls" in msg:
        tool_results = []
        for tc in msg["tool_calls"]:
            tool_name = tc["function"]["name"]
            args = tc["function"]["arguments"]

            # Show clarifications
            if tool_name == "clarify_with_user":
                question = args.get("question", "")
                print(f"Orchestrator: {question}\n")

            # Show delegations
            elif tool_name == "delegate_to_executor":
                task_desc = args.get("task_description", "")
                print(f"→ Delegating to TaskExecutor: {task_desc[:100]}...\n")

            # Show task plans
            elif tool_name == "create_task_plan":
                tasks = args.get("tasks", [])
                print(f"→ Created plan with {len(tasks)} tasks\n")

            # Execute tool
            print(f"Executing tool: {tool_name}...")
            result = execute_orchestrator_tool(tc, registry)
            print(f"Result: {result.get('message', 'done')}\n")
            tool_results.append(result)

        # Add tool results to conversation
        orchestrator.add_message({
            "role": "tool",
            "content": str(tool_results),
        })


if __name__ == "__main__":
    test_live_orchestrator()
