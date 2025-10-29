#!/usr/bin/env python3
"""Quick test to see what the agent is actually doing."""

from pathlib import Path
import tempfile
from task_executor_agent import TaskExecutorAgent

# Create temp workspace
with tempfile.TemporaryDirectory() as tmpdir:
    workspace = Path(tmpdir)

    print("=" * 70)
    print("TESTING: Simple function task")
    print("=" * 70)

    # Create agent with very short timeout
    agent = TaskExecutorAgent(
        workspace=workspace,
        goal="Create a Python file called test.py with a function hello() that returns 'Hi!'",
        max_rounds=5,  # Give it 5 rounds
        model="gpt-oss:20b"  # Use gpt-oss, NOT qwen
    )

    print(f"\nWorkspace: {agent.workspace_manager.workspace_dir}")
    print(f"Goal: {agent.context_manager.state.goal.description if agent.context_manager.state.goal else 'None'}")
    print()

    # Run agent
    result = agent.run()

    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(f"Status: {result.get('status')}")
    print(f"Message: {result.get('message', 'N/A')}")

    # Check files
    workspace_dir = agent.workspace_manager.workspace_dir
    files = list(workspace_dir.glob("**/*.py"))
    print(f"\nFiles created: {[f.name for f in files]}")

    if files:
        print("\n" + "=" * 70)
        print("FILE CONTENTS:")
        print("=" * 70)
        for f in files:
            print(f"\n--- {f.name} ---")
            print(f.read_text())

    # Check context manager state
    print("\n" + "=" * 70)
    print("CONTEXT MANAGER STATE:")
    print("=" * 70)
    if agent.context_manager.state.goal:
        print(f"Goal tasks: {len(agent.context_manager.state.goal.tasks)}")
        if agent.context_manager.state.goal.tasks:
            task = agent.context_manager.state.goal.tasks[0]
            print(f"Task 0 subtasks: {len(task.subtasks)}")
            print(f"Task 0 status: {task.status}")
    else:
        print("No goal in context manager!")
