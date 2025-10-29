"""
Test orchestrator interaction without interactive input.
"""
from pathlib import Path
from agent_registry import AgentRegistry
from agent_config import config


def test_orchestrator_simple():
    """Test orchestrator with a simple request."""

    workspace = Path.cwd()
    registry = AgentRegistry(config_path="agents.yaml", workspace=workspace)
    orchestrator = registry.get_agent("orchestrator")

    print("=" * 60)
    print("TESTING ORCHESTRATOR")
    print("=" * 60)
    print()

    # User request
    user_message = "make a simple calculator"
    print(f"User: {user_message}")
    print()

    # Add to orchestrator
    orchestrator.add_user_message(user_message)

    # Execute round
    print("Calling LLM...")
    response = orchestrator.execute_round(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )

    print("\nResponse received:")
    print(f"Response keys: {response.keys()}")

    if "message" in response:
        msg = response["message"]
        print(f"Message keys: {msg.keys()}")
        print(f"Role: {msg.get('role')}")
        print(f"Content: {msg.get('content', '')[:200]}")

        if "tool_calls" in msg:
            print(f"\nTool calls: {len(msg['tool_calls'])}")
            for tc in msg["tool_calls"]:
                print(f"  - {tc['function']['name']}")
                print(f"    Args: {tc['function']['arguments']}")
        else:
            print("\nNo tool calls")
    else:
        print(f"Full response: {response}")


if __name__ == "__main__":
    test_orchestrator_simple()
