"""
Test orchestrator to see if it delegates after planning.
"""
from pathlib import Path
from agent_registry import AgentRegistry
from agent_config import config
from orchestrator_main import execute_orchestrator_tool


def test_delegation():
    """Test if orchestrator delegates after creating plan."""

    workspace = Path.cwd()
    registry = AgentRegistry(config_path="agents.yaml", workspace=workspace)
    orchestrator = registry.get_agent("orchestrator")

    print("=" * 60)
    print("TESTING DELEGATION")
    print("=" * 60)
    print()

    # Turn 1: Initial request
    print("User: make a hello.txt file with 'Hello World' inside\n")
    orchestrator.add_user_message("make a hello.txt file with 'Hello World' inside")

    response = orchestrator.execute_round(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )

    display_and_execute(response, registry, orchestrator, turn=1)

    # Turn 2: If it asked clarification, confirm to proceed
    if any("clarify" in str(tc.get("function", {}).get("name", ""))
           for tc in response.get("message", {}).get("tool_calls", [])):
        print("\nUser: yes, just do that\n")
        orchestrator.add_user_message("yes, just do that")

        response = orchestrator.execute_round(
            model=config.llm.model,
            temperature=config.llm.temperature,
        )

        display_and_execute(response, registry, orchestrator, turn=2)

    # Turn 3: Check if it needs another nudge
    print("\nUser: please proceed with creating it\n")
    orchestrator.add_user_message("please proceed with creating it")

    response = orchestrator.execute_round(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )

    display_and_execute(response, registry, orchestrator, turn=3)


def display_and_execute(response, registry, orchestrator, turn):
    """Display and execute response."""
    print(f"\n--- Turn {turn} Response ---")

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

            print(f"\nTool: {tool_name}")

            # Show specific tool info
            if tool_name == "clarify_with_user":
                print(f"  Question: {args.get('question', '')[:80]}...")
            elif tool_name == "delegate_to_executor":
                print(f"  Task: {args.get('task_description', '')[:80]}...")
            elif tool_name == "create_task_plan":
                tasks = args.get("tasks", [])
                print(f"  Tasks: {len(tasks)}")

            # Execute (but timeout delegation for testing)
            if tool_name == "delegate_to_executor":
                print("  (Skipping actual execution for this test)")
                result = {"success": True, "message": "Skipped for test"}
            else:
                result = execute_orchestrator_tool(tc, registry)

            print(f"  Result: {result.get('message', 'done')}")
            tool_results.append(result)

        # Add tool results to conversation
        orchestrator.add_message({
            "role": "tool",
            "content": str(tool_results),
        })
    else:
        print("No tool calls")


if __name__ == "__main__":
    test_delegation()
