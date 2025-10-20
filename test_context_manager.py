"""Test and demonstrate the hierarchical context manager."""

import sys

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from context_manager import ContextManager, Subtask, Task


def test_basic_workflow() -> None:
    """Test basic workflow with goal → task → subtask → action."""
    ctx = ContextManager()

    # Initialize with a goal
    goal_text = "Create mathx package with add function and tests"
    ctx.load_or_init(goal_text)

    assert ctx.state.goal is not None
    assert ctx.state.goal.description == goal_text
    print("✓ Goal initialized")

    # Add tasks
    task1 = Task(description="Create package structure", parent_goal=goal_text)
    task1.subtasks = [
        Subtask(description="Create mathx/__init__.py"),
        Subtask(description="Create mathx/add function"),
    ]
    task1.status = "in_progress"

    ctx.state.goal.tasks.append(task1)
    ctx._save_state()
    print("✓ Task added with subtasks")

    # Mark first subtask active
    task1.subtasks[0].status = "in_progress"

    # Record some actions
    action_allowed = ctx.record_action(
        name="write_file",
        args={"path": "mathx/__init__.py", "content": "# init"},
        result="success",
    )
    assert action_allowed
    print("✓ Action recorded")

    # Get compact context
    context = ctx.get_compact_context()
    print("\n=== Compact Context ===")
    print(context)
    print("=" * 40)


def test_loop_detection() -> None:
    """Test that loop detection prevents infinite retries."""
    ctx = ContextManager()
    ctx.load_or_init("Test loop detection")

    # Add a task
    task = Task(description="Fix failing test")
    subtask = Subtask(description="Run pytest")
    subtask.status = "in_progress"
    task.subtasks.append(subtask)
    task.status = "in_progress"
    ctx.state.goal.tasks.append(task)

    # Try the same action multiple times
    args = {"cmd": ["pytest", "-q"]}
    for i in range(5):
        allowed = ctx.record_action(
            name="run_cmd", args=args, result="error", error_msg="test failed"
        )
        print(f"Attempt {i+1}: {'allowed' if allowed else 'BLOCKED (loop detected)'}")

        if not allowed:
            print("✓ Loop detection working")
            break
    else:
        print("✗ Loop detection failed - action should have been blocked")

    # Show loop summary
    print("\n=== Loop Summary ===")
    print(ctx.get_loop_summary())
    print("=" * 40)


def test_crash_recovery() -> None:
    """Test that context can be recovered after crash."""
    goal = "Test crash recovery"

    # Simulate first session
    print("\n=== Session 1 (before crash) ===")
    ctx1 = ContextManager()
    ctx1.load_or_init(goal)

    task = Task(description="Write code")
    task.status = "in_progress"
    subtask = Subtask(description="Create file")
    subtask.status = "in_progress"
    task.subtasks.append(subtask)
    ctx1.state.goal.tasks.append(task)

    ctx1.record_action(
        name="write_file",
        args={"path": "test.py", "content": "print('hi')"},
        result="success",
    )
    ctx1._save_state()

    print(ctx1.get_compact_context())

    # Simulate crash and recovery
    print("\n=== Session 2 (after crash recovery) ===")
    ctx2 = ContextManager()
    ctx2.load_or_init(goal)

    # State should be restored
    assert ctx2.state.goal is not None
    assert ctx2.state.goal.description == goal
    assert len(ctx2.state.goal.tasks) == 1
    assert ctx2.state.goal.tasks[0].description == "Write code"
    assert len(ctx2.state.goal.tasks[0].subtasks) == 1

    print(ctx2.get_compact_context())
    print("✓ Context successfully recovered after crash")


def test_hierarchical_focus() -> None:
    """Test that context only shows need-to-know info."""
    ctx = ContextManager()
    ctx.load_or_init("Build complete system")

    # Add multiple tasks (only current one should be in context)
    tasks = [
        Task(description="Setup project structure"),
        Task(description="Write core logic"),
        Task(description="Add tests"),
        Task(description="Add documentation"),
    ]

    for task in tasks:
        task.subtasks = [
            Subtask(description=f"{task.description} - step 1"),
            Subtask(description=f"{task.description} - step 2"),
        ]
        ctx.state.goal.tasks.append(task)

    # Mark first task as in progress
    tasks[0].status = "in_progress"
    tasks[0].subtasks[0].status = "in_progress"

    # Add some actions to first subtask
    for i in range(5):
        ctx.record_action(
            name="write_file",
            args={"path": f"file{i}.py", "content": "test"},
            result="success",
        )

    # Get context - should only show current task/subtask
    context = ctx.get_compact_context()

    print("\n=== Hierarchical Context (focused) ===")
    print(context)
    print("=" * 40)

    # Verify it doesn't show future tasks in detail
    assert "Add documentation" not in context
    assert "Setup project structure" in context  # Current task
    print("✓ Context properly filtered to current hierarchy")


def test_probe_state_integration() -> None:
    """Test integration with probe state (filesystem checks)."""
    ctx = ContextManager()
    ctx.load_or_init("Create and test package")

    # Simulate probe state
    probe_state = {
        "pkg_exists": True,
        "tests_exist": False,
        "pytest_ok": False,
        "ruff_ok": True,
    }

    ctx.update_probe_state(probe_state)

    # Add task
    task = Task(description="Fix missing tests")
    task.status = "in_progress"
    subtask = Subtask(description="Create test file")
    subtask.status = "in_progress"
    task.subtasks.append(subtask)
    ctx.state.goal.tasks.append(task)

    context = ctx.get_compact_context()

    print("\n=== Context with Probe State ===")
    print(context)
    print("=" * 40)

    # Verify probe state is shown
    assert "tests_exist: ✗" in context
    assert "ruff_ok: ✓" in context
    print("✓ Probe state integrated into context")


if __name__ == "__main__":
    print("Running context manager tests...\n")

    test_basic_workflow()
    print("\n" + "=" * 60 + "\n")

    test_loop_detection()
    print("\n" + "=" * 60 + "\n")

    test_crash_recovery()
    print("\n" + "=" * 60 + "\n")

    test_hierarchical_focus()
    print("\n" + "=" * 60 + "\n")

    test_probe_state_integration()

    print("\n" + "=" * 60)
    print("All tests completed!")
