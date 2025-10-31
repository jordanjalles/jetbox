#!/usr/bin/env python3
"""Test goal wall-clock timeout handling."""

import time
from pathlib import Path
from task_executor_agent import TaskExecutorAgent


def test_timeout_handling():
    """Test that goal timeout creates summary and context dump."""

    # Create agent with very short timeout (5 seconds) using timeout parameter
    agent = TaskExecutorAgent(
        workspace=Path.cwd(),
        goal="Create a complex system with many files and tests that will take more than 5 seconds",
        timeout=5  # Override default 10-minute timeout to 5 seconds for testing
    )

    print("Starting agent with 5-second timeout override...")
    print(f"Goal start time: {agent.goal_start_time}")
    print(f"Timeout override: {agent.timeout_override}s")
    print(f"Config default: {agent.config.timeouts.max_goal_time}s")

    # Run agent - should timeout
    result = agent.run(max_rounds=100)

    print(f"\nResult: {result}")
    print(f"Status: {result.get('status')}")
    print(f"Reason: {result.get('reason')}")
    print(f"Elapsed: {result.get('elapsed_seconds', 0):.1f}s")

    # Check that timeout artifacts were created
    timeout_dumps = list(Path(".agent_context/timeout_dumps").glob("goal_timeout_*.json"))
    print(f"\nTimeout dumps found: {len(timeout_dumps)}")
    if timeout_dumps:
        latest_dump = max(timeout_dumps, key=lambda p: p.stat().st_mtime)
        print(f"Latest dump: {latest_dump}")
        print(f"Size: {latest_dump.stat().st_size:,} bytes")

    # Check jetbox notes
    jetbox_file = agent.workspace_manager.workspace_dir / "jetboxnotes.md"
    if jetbox_file.exists():
        content = jetbox_file.read_text()
        print(f"\nJetbox notes size: {len(content)} chars")
        if "TIMEOUT" in content:
            print("✓ Timeout summary found in jetbox notes")
        else:
            print("✗ No timeout summary in jetbox notes")

    # Verify result
    if result.get("status") == "timeout":
        print("\n✓ Timeout handling working correctly!")
    else:
        print(f"\n✗ Expected timeout status, got: {result.get('status')}")


if __name__ == "__main__":
    test_timeout_handling()
