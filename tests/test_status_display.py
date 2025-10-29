"""Test script to demonstrate the enhanced status display.

This creates a mock agent run to show the status display in action.
"""

from context_manager import ContextManager, Goal, Task, Subtask, Action
from status_display import StatusDisplay


def demo_status_display():
    """Run a demo of the status display."""

    # Create context manager with a sample goal
    ctx = ContextManager()

    # Create a sample goal with hierarchical tasks
    goal = Goal(
        description="Create a Python calculator package with tests",
        status="in_progress"
    )

    # Task 1: Create package structure
    task1 = Task(
        description="Create calculator package structure",
        status="in_progress",
        parent_goal=goal.description,
        subtasks=[
            Subtask(
                description="Write calculator/__init__.py with add() and subtract() functions",
                status="completed",
                actions=[
                    Action(
                        name="write_file",
                        args={"path": "calculator/__init__.py", "content": "..."},
                        result="success"
                    )
                ]
            ),
            Subtask(
                description="Write calculator/advanced.py with multiply() and divide() functions",
                status="in_progress",
                actions=[
                    Action(
                        name="write_file",
                        args={"path": "calculator/advanced.py", "content": "..."},
                        result="success"
                    ),
                    Action(
                        name="run_cmd",
                        args={"cmd": ["python", "-m", "py_compile", "calculator/advanced.py"]},
                        result="success"
                    )
                ]
            ),
            Subtask(
                description="Write pyproject.toml for package configuration",
                status="pending"
            )
        ]
    )

    # Task 2: Write tests
    task2 = Task(
        description="Write comprehensive tests",
        status="pending",
        parent_goal=goal.description,
        subtasks=[
            Subtask(
                description="Write tests/test_basic.py for add() and subtract()",
                status="pending"
            ),
            Subtask(
                description="Write tests/test_advanced.py for multiply() and divide()",
                status="pending"
            )
        ]
    )

    # Task 3: Quality checks
    task3 = Task(
        description="Run quality checks",
        status="pending",
        parent_goal=goal.description,
        subtasks=[
            Subtask(description="Run ruff check", status="pending"),
            Subtask(description="Run pytest", status="pending")
        ]
    )

    goal.tasks = [task1, task2, task3]
    ctx.state.goal = goal
    ctx.state.current_task_idx = 0

    # Set up mock probe state
    ctx.state.last_probe_state = {
        "files_written": ["calculator/__init__.py", "calculator/advanced.py"],
        "files_exist": ["calculator/__init__.py", "calculator/advanced.py"],
        "files_missing": [],
        "commands_run": ["python -m py_compile calculator/advanced.py"],
        "recent_errors": []
    }

    # Create status display
    status = StatusDisplay(ctx)

    # Simulate some performance stats
    status.stats.llm_call_times = [2.3, 1.8, 2.1, 1.5, 2.0]
    status.stats.messages_sent = 15
    status.stats.total_tokens_estimated = 1500
    status.stats.actions_total = 8
    status.stats.actions_successful = 7
    status.stats.actions_failed = 1
    status.stats.subtasks_completed = 1
    status.stats.total_runtime = 45.0

    # Display full status
    print("\n" + "="*70)
    print("DEMO: Enhanced Agent Status Display")
    print("="*70 + "\n")

    print(status.render(round_no=5))

    print("\n" + "="*70)
    print("COMPACT STATUS (one-line)")
    print("="*70)
    print(status.render_compact())
    print()

    # Simulate progress
    print("\n" + "="*70)
    print("SIMULATING PROGRESS...")
    print("="*70 + "\n")

    # Complete current subtask
    task1.subtasks[1].status = "completed"
    task1.subtasks[2].status = "in_progress"
    status.stats.subtasks_completed = 2
    status.stats.total_runtime = 60.0
    status.stats.actions_total = 12
    status.stats.actions_successful = 11

    print(status.render(round_no=8))

    # Add an error scenario
    print("\n" + "="*70)
    print("SIMULATING ERROR SCENARIO...")
    print("="*70 + "\n")

    ctx.state.last_probe_state["recent_errors"] = [
        "run_cmd rc=1: ruff: command not found",
        "File 'pyproject.toml' does not exist"
    ]
    status.stats.loops_detected = 1

    print(status.render(round_no=10))


if __name__ == "__main__":
    demo_status_display()
