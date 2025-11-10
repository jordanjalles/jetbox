"""
Debug script to see actual LLM prompts and responses.
"""
import tempfile
from pathlib import Path
from agents.task_executor_agent import TaskExecutorAgent

# Create temp workspace
with tempfile.TemporaryDirectory() as tmp:
    ws = Path(tmp) / "debug_test"
    ws.mkdir()

    # Create agent
    print("=" * 80)
    print("CREATING AGENT")
    print("=" * 80)
    agent = TaskExecutorAgent(
        workspace=ws,
        goal='Create a Python file called hello.py that prints Hello World'
    )

    # Check agent state
    print(f"\nAgent has {len(agent.get_tools())} tools")
    print(f"Context manager state: {agent.context_manager.state.goal}")

    # Build context for first round
    print("\n" + "=" * 80)
    print("FIRST ROUND CONTEXT")
    print("=" * 80)
    context = agent.build_context()

    for i, msg in enumerate(context):
        print(f"\n--- Message {i} ({msg['role']}) ---")
        content = msg.get('content', '')
        if len(content) > 500:
            print(content[:250] + "\n...\n" + content[-250:])
        else:
            print(content)

    # List behaviors
    print("\n" + "=" * 80)
    print("LOADED BEHAVIORS")
    print("=" * 80)
    for behavior in agent.behaviors:
        print(f"  - {behavior.get_name()}: {type(behavior).__name__}")

    # List tools
    print("\n" + "=" * 80)
    print("REGISTERED TOOLS")
    print("=" * 80)
    for tool_name, behavior in agent.tool_registry.items():
        print(f"  - {tool_name} (from {behavior.get_name()})")

    # Call LLM once
    print("\n" + "=" * 80)
    print("CALLING LLM")
    print("=" * 80)
    response = agent.call_llm(
        model=agent.config.llm.model,
        temperature=agent.config.llm.temperature
    )

    print(f"\nResponse keys: {response.keys()}")

    if 'message' in response:
        msg = response['message']
        print(f"\nMessage role: {msg.get('role')}")
        print(f"Message content: {msg.get('content', '')[:500]}")

        if 'tool_calls' in msg:
            print(f"\nTool calls: {len(msg.get('tool_calls', []))}")
            for tc in msg.get('tool_calls', []):
                print(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
        else:
            print("\nNo tool calls in response")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
