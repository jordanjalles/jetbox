"""
Test that loop detection handles non-serializable objects gracefully.

This test verifies the fix for the JSON serialization crash when
mark_complete and other tools pass ContextManager objects in args.
"""
import pytest
from context_strategies import HierarchicalStrategy, AppendUntilFullStrategy, SubAgentStrategy
from context_manager import ContextManager


def test_loop_detection_with_context_manager_arg():
    """Test that record_action handles ContextManager in args."""
    strategy = HierarchicalStrategy()
    context_manager = ContextManager()

    # Simulate mark_complete call with context_manager in args
    # This is what TaskExecutorAgent.dispatch_tool does
    args = {
        "summary": "Task completed successfully",
        "context_manager": context_manager  # Non-serializable object
    }

    # This should NOT crash
    warning = strategy.record_action(
        tool_name="mark_complete",
        args=args,
        result={"status": "goal_complete"},
        success=True
    )

    # No loop detected on first call
    assert warning is None


def test_loop_detection_with_various_non_serializable_objects():
    """Test that record_action handles various non-serializable objects."""
    strategy = AppendUntilFullStrategy()

    # Test with different non-serializable types
    test_cases = [
        {"path": "/tmp/file.txt", "obj": ContextManager()},
        {"command": "pytest", "function": lambda x: x},
        {"data": [1, 2, {"nested": object()}]},
    ]

    for args in test_cases:
        # Should not crash
        warning = strategy.record_action(
            tool_name="test_tool",
            args=args,
            result="success",
            success=True
        )
        assert warning is None


def test_loop_detection_still_works_after_fix():
    """Verify that loop detection still works correctly after the fix."""
    strategy = SubAgentStrategy()

    # Create non-serializable args
    args = {
        "reason": "Failed to compile",
        "context_manager": ContextManager()
    }

    # Repeat the same action 5 times
    for i in range(5):
        warning = strategy.record_action(
            tool_name="mark_failed",
            args=args,
            result={"status": "failed", "reason": "Failed to compile"},
            success=False
        )

        if i < 4:
            # No warning yet
            assert warning is None
        else:
            # Loop detected on 5th repeat
            assert warning is not None
            assert "repeated" in warning["warning"].lower()
            assert "suggestion" in warning


def test_serializable_args_unchanged():
    """Test that serializable args are handled correctly (no regression)."""
    strategy = HierarchicalStrategy()

    args = {
        "path": "/tmp/test.py",
        "content": "print('hello')",
        "append": False
    }

    # Should work as before
    warning = strategy.record_action(
        tool_name="write_file",
        args=args,
        result="Wrote 15 chars to /tmp/test.py",
        success=True
    )

    assert warning is None


def test_make_serializable_method():
    """Test the _make_serializable helper method directly."""
    strategy = HierarchicalStrategy()

    # Primitives should pass through
    assert strategy._make_serializable(None) is None
    assert strategy._make_serializable(True) is True
    assert strategy._make_serializable(42) == 42
    assert strategy._make_serializable(3.14) == 3.14
    assert strategy._make_serializable("hello") == "hello"

    # Dicts with primitives
    assert strategy._make_serializable({"a": 1, "b": "test"}) == {"a": 1, "b": "test"}

    # Lists with primitives
    assert strategy._make_serializable([1, 2, "three"]) == [1, 2, "three"]

    # Non-serializable objects become type strings
    result = strategy._make_serializable(ContextManager())
    assert result == "<ContextManager>"

    # Nested structures
    nested = {
        "path": "/tmp/file",
        "context": ContextManager(),
        "data": [1, 2, {"obj": object()}]
    }
    result = strategy._make_serializable(nested)
    assert result["path"] == "/tmp/file"
    assert result["context"] == "<ContextManager>"
    assert result["data"][0] == 1
    assert result["data"][2]["obj"] == "<object>"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
