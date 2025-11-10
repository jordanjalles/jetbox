#!/usr/bin/env python3
"""
Test that malformed tool call errors are passed back to the LLM with actionable feedback.

This test validates that when the LLM generates malformed tool calls (e.g., text before JSON),
the error is immediately fed back with clear instructions on how to fix it.
"""
from pathlib import Path
from agents.task_executor_agent import TaskExecutorAgent

if __name__ == "__main__":
    print("=" * 80)
    print("TEST: Tool Call Error Feedback")
    print("=" * 80)
    print()
    print("This test validates that malformed tool call errors are:")
    print("  1. Detected immediately")
    print("  2. Sent back to LLM as user feedback")
    print("  3. Include actionable instructions")
    print()
    print("Watch for:")
    print("  - 'ERROR: Your last response had a malformed tool call'")
    print("  - 'CORRECT FORMAT' examples in feedback")
    print("  - LLM correcting itself on next round")
    print()
    print("=" * 80)
    print()

    workspace = Path(".agent_workspaces/tests/test-tool-call-feedback")

    executor = TaskExecutorAgent(workspace=workspace)
    executor.add_message({
        "role": "user",
        "content": "Create a simple hello.py file that prints 'Hello World'"
    })

    result = executor.run(max_rounds=10)

    print("\n" + "=" * 80)
    print("RESULT:")
    print(f"  Status: {result.get('status')}")
    print(f"  Summary: {result.get('summary', result.get('message', result.get('reason', 'N/A')))}")
    print(f"  Workspace: {result.get('workspace')}")
    print("=" * 80)

    # Check if hello.py was created
    hello_file = workspace / "hello.py"
    if hello_file.exists():
        print("\n✓ SUCCESS: hello.py was created")
        print(f"  Content preview: {hello_file.read_text()[:100]}...")
    else:
        print("\n✗ FAILURE: hello.py was not created")

    print("\nLOOKING FOR ERROR FEEDBACK IN OUTPUT ABOVE:")
    print("  - Did we see 'ERROR: Your last response had a malformed tool call'?")
    print("  - Did the feedback include CORRECT FORMAT examples?")
    print("  - Did the LLM correct itself in subsequent rounds?")
    print()
