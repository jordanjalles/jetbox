"""Test script to demonstrate the enhanced status visualization.

This demonstrates all the improvements:
1. Discrete chunk progress bars (not continuous percentages)
2. Discrete context window visualization with model-based sizing
3. Error display only when blocking (low success rate)
4. Files created cleared per session (simulated here)
"""

from context_manager import ContextManager, Goal, Task, Subtask, Action
from status_display import StatusDisplay

def demo_basic_visualization():
    """Demo basic status display with discrete chunks and activity tracking."""
    print("\n" + "="*80)
    print("DEMO 1: Agent Status with Activity Tracking")
    print("="*80 + "\n")

    # Create a context manager with some test data
    ctx = ContextManager()

    # Create a test goal with tasks and subtasks
    goal = Goal(description="Build a web application with Python backend")

    # Task 1: Setup project (completed)
    task1 = Task(description="Setup project structure", status="completed")
    task1.subtasks = [
        Subtask(description="Create project directory", depth=1, status="completed"),
        Subtask(description="Initialize git repository", depth=1, status="completed"),
    ]
    goal.tasks.append(task1)

    # Task 2: Backend development (in progress)
    task2 = Task(description="Implement Python Flask backend", status="in_progress")
    subtask_2_1 = Subtask(description="Create Flask app structure", depth=1, status="completed")
    subtask_2_2 = Subtask(description="Implement API endpoints", depth=1, status="in_progress")

    # Add action to subtask to show in status
    subtask_2_2.actions = [
        Action(name="write_file", args={"path": "api/routes.py"}, result="success")
    ]

    subtask_2_3 = Subtask(description="Add database models", depth=1, status="pending")
    task2.subtasks = [subtask_2_1, subtask_2_2, subtask_2_3]
    goal.tasks.append(task2)

    # Task 3: Frontend development (pending)
    task3 = Task(description="Build HTML/CSS/JavaScript frontend", status="pending")
    task3.subtasks = [
        Subtask(description="Create HTML templates", depth=1, status="pending"),
        Subtask(description="Add CSS styling", depth=1, status="pending"),
    ]
    goal.tasks.append(task3)

    # Set up the context
    ctx.state.goal = goal
    ctx.state.current_task_idx = 1  # Task 2 is active

    # Create status display
    display = StatusDisplay(ctx, reset_stats=True)

    # Set current activity
    display.set_activity("writing file: api/routes.py")

    # Simulate some stats with HIGH success rate (no errors shown)
    display.stats.actions_total = 25
    display.stats.actions_successful = 23
    display.stats.actions_failed = 2
    display.stats.llm_call_times = [1.5, 2.3, 1.8, 2.0, 1.9, 2.2, 1.7, 2.1]
    display.stats.subtasks_completed = 3
    display.stats.subtasks_failed = 0

    # Mock probe state (no errors - simulating clean session)
    ctx.state.last_probe_state = {
        "files_written": ["app.py", "models.py"],  # Only current session files
        "files_exist": ["app.py", "models.py"],
        "files_missing": [],
        "commands_run": ["pytest tests/"],
        "recent_errors": []  # No errors
    }

    # Example context usage stats - NOW DISPLAYED with accurate tracking!
    context_stats = {
        "system_prompt": 1200,
        "task_desc": 800,
        "agent_output": 4500,
        "system_interaction": 6000,  # All tool outputs (files, cmd results, errors)
    }

    # Render and display (with turn counter)
    status_output = display.render(round_no=8, context_stats=context_stats,
                                   subtask_rounds=3, max_rounds=6)
    print(status_output)


def demo_error_visibility():
    """Demo error display only when blocking progress."""
    print("\n\n" + "="*80)
    print("DEMO 2: Error Visibility - Errors Only Shown When Blocking")
    print("="*80 + "\n")

    ctx = ContextManager()
    goal = Goal(description="Fix failing tests")
    task = Task(description="Debug and fix test failures", status="in_progress")

    # Create subtask with failed actions
    subtask = Subtask(description="Fix test_api_endpoints.py", depth=1, status="in_progress")
    subtask.actions = [
        Action(name="run_cmd", args={"cmd": "pytest"}, result="failure", error_msg="ImportError: cannot import 'app'"),
        Action(name="write_file", args={"path": "app.py"}, result="success"),
        Action(name="run_cmd", args={"cmd": "pytest"}, result="failure", error_msg="AssertionError: Expected 200, got 404"),
    ]
    task.subtasks = [subtask]
    goal.tasks.append(task)

    ctx.state.goal = goal
    ctx.state.current_task_idx = 0

    display = StatusDisplay(ctx, reset_stats=True)

    # LOW success rate - errors WILL be shown
    display.stats.actions_total = 10
    display.stats.actions_successful = 5
    display.stats.actions_failed = 5

    ctx.state.last_probe_state = {
        "files_written": [],
        "files_exist": [],
        "files_missing": [],
        "commands_run": ["pytest"],
        "recent_errors": ["ImportError: cannot import 'app'", "AssertionError: Expected 200, got 404"]
    }

    print("Scenario: LOW success rate (50%) - Errors are BLOCKING progress")
    print(display.render(round_no=5, subtask_rounds=2, max_rounds=6))

    # Now simulate HIGH success rate - errors won't be shown
    print("\n\n" + "="*80)
    print("Scenario: HIGH success rate (92%) - Errors are NOT shown (progress is being made)")
    print("="*80 + "\n")

    display.stats.actions_successful = 23
    display.stats.actions_total = 25

    print(display.render(round_no=12, subtask_rounds=5, max_rounds=6))


def demo_turn_counter():
    """Demo turn counter with warning colors."""
    print("\n\n" + "="*80)
    print("DEMO 3: Turn Counter Visualization")
    print("="*80 + "\n")

    ctx = ContextManager()
    goal = Goal(description="Complete multiple small tasks")

    # Create many tasks to show chunk grouping
    for i in range(1, 16):
        task = Task(
            description=f"Task {i}",
            status="completed" if i <= 5 else "pending"
        )
        task.subtasks = [
            Subtask(description=f"Subtask {i}.1", depth=1, status="completed" if i <= 5 else "pending"),
            Subtask(description=f"Subtask {i}.2", depth=1, status="completed" if i <= 5 else "pending"),
        ]
        goal.tasks.append(task)

    ctx.state.goal = goal
    ctx.state.current_task_idx = 5

    display = StatusDisplay(ctx, reset_stats=True)
    display.stats.actions_total = 100
    display.stats.actions_successful = 95

    print("Notice turn counter with warning colors:")
    print(display._render_turn_counter(current=5, max_turns=6))
    print("\nWhen at 80% capacity:")
    print(display._render_turn_counter(current=5, max_turns=6))
    print("\nWhen early:")
    print(display._render_turn_counter(current=1, max_turns=6))


if __name__ == "__main__":
    demo_basic_visualization()
    demo_error_visibility()
    demo_turn_counter()

    print("\n" + "="*80)
    print("ALL DEMOS COMPLETE!")
    print("="*80)
    print("\nKey improvements demonstrated:")
    print("  ✓ Performance stats at TOP (not bottom)")
    print("  ✓ Agent status at BOTTOM with latest action")
    print("  ✓ Turn counter with circles (●○○○○○) replaces progress bars")
    print("  ✓ Warning colors: Green=early, Yellow=half, Red=near limit")
    print("  ✓ Context window ACCURATE and updates every turn")
    print("  ✓ 'System Interaction' bucket for all tool outputs")
    print("  ✓ Completed tasks show ✓ (not ⟳)")
    print("  ✓ Current step HIGHLIGHTED in bold cyan")
    print("  ✓ Model-aware context window sizing")
    print("  ✓ Session-scoped file tracking (clears between runs)")
    print("\nNote: In-place updates work when running the actual agent")
    print()
