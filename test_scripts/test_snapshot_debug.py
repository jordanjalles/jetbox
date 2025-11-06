#!/usr/bin/env python
"""Test workspace_task_notes snapshot behavior with debug logging."""

import sys
from pathlib import Path
import shutil

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from task_executor_agent import TaskExecutorAgent

def test_snapshot():
    """Test that snapshot file is created on first run."""

    # Clean up any existing workspace
    workspace_path = Path(".agent_workspaces/test-snapshot-debug")
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

    print("=" * 80)
    print("TEST: Snapshot file creation on first run")
    print("=" * 80)

    # Run task executor with a simple goal
    print("\n>>> Creating TaskExecutorAgent...")
    agent = TaskExecutorAgent(
        goal="Create a file called hello.txt with 'Hello, World!' in it",
        workspace=str(workspace_path),
        timeout_seconds=60
    )

    print("\n>>> Running agent...")
    try:
        agent.run()
    except Exception as e:
        print(f"Agent run error: {e}")

    # Check if snapshot file exists
    snapshot_file = workspace_path / ".agent_context" / "wtn_file_snapshot.json"
    print(f"\n>>> Checking for snapshot file: {snapshot_file}")
    print(f">>> Snapshot file exists: {snapshot_file.exists()}")

    if snapshot_file.exists():
        print(f">>> Snapshot file contents:")
        print(snapshot_file.read_text())
    else:
        print(">>> Snapshot file NOT created!")

    # Check WTN file
    wtn_file = workspace_path / ".agent_context" / "workspace_task_notes.md"
    print(f"\n>>> Checking for WTN file: {wtn_file}")
    print(f">>> WTN file exists: {wtn_file.exists()}")

    if wtn_file.exists():
        print(f">>> WTN file contents:")
        print(wtn_file.read_text())

if __name__ == "__main__":
    test_snapshot()
