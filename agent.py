#!/usr/bin/env python3
"""
agent.py - CLI wrapper for TaskExecutorAgent

This file is now a thin wrapper around TaskExecutorAgent.
All agent logic has been moved to task_executor_agent.py.

The original 2068-line agent.py has been backed up to agent_legacy.py.
"""
import argparse
import sys
from pathlib import Path

from task_executor_agent import TaskExecutorAgent


def dispatch(call):
    """
    Dispatch tool calls - imported by base_agent.py.

    This delegates to the tools module which has all tool implementations.
    """
    import tools

    # Get tool name and args
    tool_name = call["function"]["name"]
    args = call["function"]["arguments"]

    # Map tool names to functions
    tool_map = {
        "list_dir": tools.list_dir,
        "read_file": tools.read_file,
        "grep_file": tools.grep_file,
        "write_file": tools.write_file,
        "run_cmd": tools.run_cmd,
        "start_server": tools.start_server,
        "stop_server": tools.stop_server,
        "check_server": tools.check_server,
        "list_servers": tools.list_servers,
        "mark_subtask_complete": tools.mark_subtask_complete,
    }

    # Execute the tool
    if tool_name in tool_map:
        result = tool_map[tool_name](**args)
        return {"result": result}
    else:
        return {"result": {"status": "error", "message": f"Unknown tool: {tool_name}"}}


def main() -> None:
    """Main entry point - parse args and run TaskExecutorAgent."""
    parser = argparse.ArgumentParser(description="Jetbox coding agent")
    parser.add_argument("goal", nargs="?", default="", help="Goal description")
    parser.add_argument("--workspace", type=str, help="Existing workspace path")
    parser.add_argument("--context", type=str, help="Additional context")
    parser.add_argument("--model", type=str, help="Ollama model to use")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-rounds", type=int, default=128, help="Max rounds")

    args = parser.parse_args()

    # Require goal
    if not args.goal:
        print("Error: Goal required")
        print("Usage: python agent.py \"Your goal description\"")
        sys.exit(1)

    # Determine workspace
    workspace = Path(args.workspace) if args.workspace else Path(".")

    # Create TaskExecutorAgent
    print(f"[agent] Initializing TaskExecutorAgent...")
    executor = TaskExecutorAgent(
        workspace=workspace,
        goal=args.goal,
        max_rounds=args.max_rounds,
        model=args.model,
        temperature=args.temperature,
    )

    # Run it
    print(f"[agent] Running goal: {args.goal}")
    result = executor.run()

    # Exit with appropriate code
    if result["status"] == "success":
        print(f"\n[agent] ✅ Goal completed successfully")
        sys.exit(0)
    else:
        print(f"\n[agent] ❌ Goal failed: {result.get('reason', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
