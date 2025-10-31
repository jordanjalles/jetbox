"""
Integration test for JSON serialization bug fix.

This test reproduces the exact scenario from the bug report:
1. TaskExecutorAgent.dispatch_tool injects context_manager into args
2. record_action tries to JSON serialize args
3. Should NOT crash on non-serializable objects
"""
import pytest
from context_manager import ContextManager
from context_strategies import HierarchicalStrategy, SubAgentStrategy, AppendUntilFullStrategy


def test_exact_bug_scenario():
    """
    Reproduce the exact bug scenario from the report.

    Bug: TaskExecutorAgent.dispatch_tool passes args with context_manager
    to record_action, which crashes on json.dumps(args).
    """
    # Create context manager (non-serializable object)
    context_manager = ContextManager()
    context_manager.load_or_init("Test goal")

    # Create strategy
    strategy = SubAgentStrategy()

    # Simulate dispatch_tool scenario
    tool_name = "mark_complete"
    args = {
        "summary": "Task completed successfully",
        "context_manager": context_manager  # Non-serializable!
    }
    result = {
        "status": "goal_complete",
        "message": "Delegated task completed successfully",
        "summary": "Task completed successfully",
        "success": True
    }

    # This should NOT crash with TypeError: Object of type ContextManager is not JSON serializable
    warning = strategy.record_action(
        tool_name=tool_name,
        args=args,
        result=result,
        success=True
    )

    # Should succeed without warning (first call)
    assert warning is None

    # Verify action was recorded
    assert len(strategy.action_history) == 1
    assert strategy.action_history[0]["tool_name"] == "mark_complete"


def test_all_context_manager_tools():
    """
    Test all tools that receive context_manager injection.

    From task_executor_agent.py:199-205:
    tools_needing_context = {
        "mark_subtask_complete",
        "mark_goal_complete",
        "mark_complete",
        "mark_failed",
        "decompose_task"
    }
    """
    context_manager = ContextManager()
    context_manager.load_or_init("Test goal")
    strategy = HierarchicalStrategy()

    test_cases = [
        ("mark_subtask_complete", {"success": True, "reason": "Done", "context_manager": context_manager}),
        ("mark_goal_complete", {"summary": "All done", "context_manager": context_manager}),
        ("mark_complete", {"summary": "Task done", "context_manager": context_manager}),
        ("mark_failed", {"reason": "Failed", "context_manager": context_manager}),
        ("decompose_task", {"subtasks": ["step1", "step2"], "context_manager": context_manager}),
    ]

    for tool_name, args in test_cases:
        # Should not crash
        warning = strategy.record_action(
            tool_name=tool_name,
            args=args,
            result={"status": "success"},
            success=True
        )
        assert warning is None, f"Tool {tool_name} failed"


def test_mixed_serializable_and_non_serializable_args():
    """Test args with both serializable and non-serializable values."""
    strategy = AppendUntilFullStrategy()
    context_manager = ContextManager()

    # Complex mixed args
    args = {
        "path": "/tmp/test.py",
        "content": "print('hello')",
        "append": False,
        "context_manager": context_manager,  # Non-serializable
        "options": {
            "encoding": "utf-8",
            "create_dirs": True,
            "workspace": context_manager  # Nested non-serializable
        },
        "callbacks": [lambda x: x, None, "valid_string"]  # List with non-serializable
    }

    # Should handle gracefully
    warning = strategy.record_action(
        tool_name="write_file",
        args=args,
        result="Wrote file",
        success=True
    )

    assert warning is None


def test_performance_no_regression():
    """Verify the fix doesn't significantly impact performance."""
    import time

    strategy = HierarchicalStrategy()

    # Test with serializable args (should be fast)
    args_serializable = {
        "path": "/tmp/test.py",
        "content": "x" * 1000,  # 1KB of content
        "append": False
    }

    start = time.time()
    for _ in range(100):
        strategy.record_action("write_file", args_serializable, "success", True)
    elapsed_serializable = time.time() - start

    # Reset for fair comparison
    strategy = HierarchicalStrategy()

    # Test with non-serializable args (should still be fast)
    context_manager = ContextManager()
    args_non_serializable = {
        "path": "/tmp/test.py",
        "content": "x" * 1000,
        "append": False,
        "context_manager": context_manager
    }

    start = time.time()
    for _ in range(100):
        strategy.record_action("write_file", args_non_serializable, "success", True)
    elapsed_non_serializable = time.time() - start

    # Non-serializable should be within 2x of serializable (generous threshold)
    assert elapsed_non_serializable < elapsed_serializable * 2, \
        f"Performance regression: {elapsed_non_serializable:.3f}s vs {elapsed_serializable:.3f}s"


def test_loop_detection_still_accurate_after_fix():
    """
    Verify loop detection still works correctly with the fix.

    The fix should not affect loop detection accuracy - identical calls
    with non-serializable objects should still be detected as loops.
    """
    strategy = SubAgentStrategy()
    context_manager = ContextManager()

    # Same args, same result - should trigger loop
    args = {
        "reason": "Cannot find module",
        "context_manager": context_manager
    }
    result = {"status": "failed", "reason": "Cannot find module"}

    # Repeat 5 times - should detect loop
    for i in range(5):
        warning = strategy.record_action(
            tool_name="mark_failed",
            args=args,
            result=result,
            success=False
        )

        if i < 4:
            assert warning is None, f"False positive on iteration {i+1}"
        else:
            assert warning is not None, "Loop not detected after 5 repeats"
            assert "repeated" in warning["warning"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
