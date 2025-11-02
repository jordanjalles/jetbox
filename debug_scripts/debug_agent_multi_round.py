"""
Debug script to run multiple agent rounds and see the full conversation.
"""
import tempfile
from pathlib import Path
from task_executor_agent import TaskExecutorAgent

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

    # Run 3 rounds manually
    for round_num in range(1, 4):
        print(f"\n{'=' * 80}")
        print(f"ROUND {round_num}")
        print(f"{'=' * 80}")

        # Build context
        context = agent.build_context()
        print(f"\nContext messages: {len(context)}")

        # Call LLM
        response = agent.call_llm(
            model=agent.config.llm.model,
            temperature=agent.config.llm.temperature
        )

        if 'message' not in response:
            print("No message in response!")
            break

        msg = response['message']
        print(f"\nMessage content: {msg.get('content', '')[:200]}")

        # Handle tool calls
        if 'tool_calls' in msg:
            print(f"\nTool calls: {len(msg.get('tool_calls', []))}")
            for tc in msg.get('tool_calls', []):
                tool_name = tc['function']['name']
                args = tc['function']['arguments']
                print(f"  - {tool_name}: {args}")

                # Dispatch tool
                result = agent.dispatch_tool(tc)
                print(f"    Result: {str(result)[:100]}")

            # Add message to history
            agent.add_message(msg)

            # Add tool results to history (simplified)
            for tc in msg.get('tool_calls', []):
                result = agent.dispatch_tool(tc)
                agent.add_message({
                    "role": "tool",
                    "content": str(result)
                })
        else:
            print("\nNo tool calls - agent just responded with text")
            agent.add_message(msg)

        # Check goal status
        if agent.context_manager.state.goal:
            print(f"\nGoal status: {agent.context_manager.state.goal.status}")

    # Final state
    print(f"\n{'=' * 80}")
    print("FINAL STATE")
    print(f"{'=' * 80}")
    print(f"\nWorkspace files:")
    for f in ws.rglob("*"):
        if f.is_file():
            print(f"  - {f.relative_to(ws)}")
            if f.suffix == ".py":
                print(f"    Content: {f.read_text()[:100]}")

    print(f"\nGoal: {agent.context_manager.state.goal.description}")
    print(f"Status: {agent.context_manager.state.goal.status}")
