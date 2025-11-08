#!/usr/bin/env python3
"""
Direct runner for TaskExecutorAgent (bypasses orchestrator).
Used by benchmarks to test task_executor performance directly.
"""

import sys
from pathlib import Path
from task_executor_agent import TaskExecutorAgent

def main():
    """Run task executor directly."""
    if len(sys.argv) < 2:
        print("Usage: python run_task_executor.py <goal> [--workspace PATH] [--config FILE]")
        sys.exit(1)

    goal = sys.argv[1]
    workspace = None

    # Parse optional args
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--workspace" and i + 1 < len(sys.argv):
            workspace = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--config":
            # Ignore --config flag (always uses task_executor_config.yaml)
            i += 2
        else:
            i += 1

    # Create and run agent
    agent = TaskExecutorAgent(
        workspace=workspace,
        goal=goal
    )

    result = agent.run()

    # Exit with appropriate code
    if result.get("status") == "success":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
