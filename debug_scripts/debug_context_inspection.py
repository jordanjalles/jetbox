"""
Debug script to inspect the exact context being sent to LLM.
"""
import tempfile
from pathlib import Path
from agents.task_executor_agent import TaskExecutorAgent

# Create temp workspace
with tempfile.TemporaryDirectory() as tmp:
    ws = Path(tmp) / "debug_test"
    ws.mkdir()

    # Create agent
    agent = TaskExecutorAgent(
        workspace=ws,
        goal='Create a Python file called hello.py that prints Hello World'
    )

    # Build context for first round
    context = agent.build_context()

    print("=" * 80)
    print("FULL CONTEXT SENT TO LLM")
    print("=" * 80)

    for i, msg in enumerate(context):
        print(f"\n--- Message {i}: {msg['role']} ---")
        print(msg['content'])
        print("---" * 20)

    print("\n" * 2)
    print("=" * 80)
    print(f"TOTAL CONTEXT LENGTH: {sum(len(m['content']) for m in context)} chars")
    print("=" * 80)
