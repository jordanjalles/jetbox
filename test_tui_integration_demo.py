#!/usr/bin/env python3
"""
Demo/test script for TUIDisplayBehavior integration.

This demonstrates how TUIDisplayBehavior integrates with the TUI system
and can be used in agent configurations.

Run with:
    python test_tui_integration_demo.py          # Auto mode (TUI if TTY)
    python test_tui_integration_demo.py --plain  # Force plain text
    python test_tui_integration_demo.py --tui    # Force TUI
"""

import sys
import time
from pathlib import Path
from behaviors.tui_display_behavior import TUIDisplayBehavior


class MockAgent:
    """Mock agent for testing TUIDisplayBehavior."""

    def __init__(self, goal: str):
        self.goal = goal
        self.name = "test_agent"
        self.model = "qwen3:8b"
        self.max_rounds = 10
        self.workspace = Path("/tmp/test_workspace")
        self.workspace_manager = MockWorkspaceManager()

    def get_context_stats(self):
        """Return mock context stats."""
        return {
            'tokens_used': 5000,
            'tokens_max': 128000,
        }


class MockWorkspaceManager:
    """Mock workspace manager."""

    def __init__(self):
        self.created_files = [
            "calculator.py",
            "test_calculator.py",
            "README.md",
        ]


def run_demo(display_mode: str = "auto"):
    """
    Run a demo of TUIDisplayBehavior.

    Args:
        display_mode: "auto", "plain", or "textual"
    """
    print(f"Starting TUIDisplayBehavior demo (mode: {display_mode})")
    print("="*70)

    # Create behavior
    behavior = TUIDisplayBehavior(display_mode=display_mode)

    # Create mock agent
    goal = "Create a calculator with tests"
    agent = MockAgent(goal)

    # Initialize context
    context = [
        {"role": "system", "content": "You are a helpful coding assistant."}
    ]

    # Lifecycle: on_goal_start
    behavior.on_goal_start(agent, goal)

    # Simulate 5 rounds with tool calls
    for round_num in range(1, 6):
        # on_round_start
        context = behavior.on_round_start(agent, round_num, context)

        time.sleep(0.3)  # Simulate LLM thinking

        # Simulate tool calls
        if round_num == 1:
            # Write calculator.py
            behavior.on_tool_call(
                agent,
                "write_file",
                {"path": "calculator.py", "content": "def add(a, b): return a + b"},
                {"success": True, "result": "File written"}
            )
        elif round_num == 2:
            # Write test file
            behavior.on_tool_call(
                agent,
                "write_file",
                {"path": "test_calculator.py", "content": "import calculator..."},
                {"success": True, "result": "Test file created"}
            )
        elif round_num == 3:
            # Run tests (success)
            behavior.on_tool_call(
                agent,
                "run_bash",
                {"command": "pytest test_calculator.py"},
                {"success": True, "result": "All tests passed"}
            )
        elif round_num == 4:
            # Read file (simulate error)
            behavior.on_tool_call(
                agent,
                "read_file",
                {"path": "missing.py"},
                {"error": "File not found: missing.py"}
            )
        elif round_num == 5:
            # Write README
            behavior.on_tool_call(
                agent,
                "write_file",
                {"path": "README.md", "content": "# Calculator\n\nSimple calculator."},
                {"success": True, "result": "Documentation created"}
            )

        time.sleep(0.2)  # Simulate tool execution

        # on_round_end
        behavior.on_round_end(agent, round_num)

    # Lifecycle: on_goal_complete
    summary = "Created calculator with tests and documentation"
    behavior.on_goal_complete(agent, success=True, summary=summary)

    print("\n" + "="*70)
    print("Demo completed successfully!")


if __name__ == "__main__":
    # Parse command line arguments
    display_mode = "auto"
    if "--plain" in sys.argv or "--no-tui" in sys.argv:
        display_mode = "plain"
    elif "--tui" in sys.argv or "--textual" in sys.argv:
        display_mode = "textual"

    run_demo(display_mode)
