"""
Tests for LoopDetectionBehavior.

Tests:
- Action tracking via on_tool_call event
- Loop detection (same action + result repeated)
- Warning injection into context
- max_repeats threshold
"""

import pytest
from unittest.mock import Mock
from behaviors.loop_detection import LoopDetectionBehavior


class TestLoopDetectionBehavior:
    """Test suite for LoopDetectionBehavior."""

    def test_get_name(self):
        """Test behavior returns correct name."""
        behavior = LoopDetectionBehavior()
        assert behavior.get_name() == "loop_detection"

    def test_initialization_with_defaults(self):
        """Test behavior initializes with default parameters."""
        behavior = LoopDetectionBehavior()
        assert behavior.max_repeats == 5
        assert behavior.action_history == []
        assert behavior.loop_warnings == []

    def test_initialization_with_custom_max_repeats(self):
        """Test behavior initializes with custom max_repeats."""
        behavior = LoopDetectionBehavior(max_repeats=10)
        assert behavior.max_repeats == 10

    def test_on_tool_call_records_action(self):
        """Test on_tool_call records actions in history."""
        behavior = LoopDetectionBehavior()

        behavior.on_tool_call(
            "write_file",
            {"path": "test.py", "content": "print('hello')"},
            {"success": True}
        )

        assert len(behavior.action_history) == 1
        assert behavior.action_history[0]["tool_name"] == "write_file"
        assert behavior.action_history[0]["success"] is True

    def test_on_tool_call_detects_identical_result_loop(self):
        """Test loop detection for identical action+result pairs."""
        behavior = LoopDetectionBehavior(max_repeats=3)

        # Repeat same action with same result 5 times
        for _ in range(5):
            behavior.on_tool_call(
                "write_file",
                {"path": "test.py", "content": "code"},
                {"error": "Permission denied"}
            )

        # Should have detected loop
        assert len(behavior.loop_warnings) > 0
        assert "write_file" in behavior.loop_warnings[0]
        assert "identical results" in behavior.loop_warnings[0]

    def test_on_tool_call_detects_repeated_action_loop(self):
        """Test loop detection for repeated actions with varying results."""
        behavior = LoopDetectionBehavior(max_repeats=3)

        # Repeat same action with different results
        for i in range(10):
            behavior.on_tool_call(
                "run_bash",
                {"command": "pytest"},
                {"output": f"Different error {i}"}
            )

        # Should have detected loop (attempts, not results)
        assert len(behavior.loop_warnings) > 0
        assert "run_bash" in behavior.loop_warnings[0]
        assert "attempted" in behavior.loop_warnings[0]

    def test_on_tool_call_no_loop_with_different_actions(self):
        """Test no loop detected when actions are different."""
        behavior = LoopDetectionBehavior(max_repeats=3)

        # Different actions
        behavior.on_tool_call("write_file", {"path": "a.py"}, {"success": True})
        behavior.on_tool_call("read_file", {"path": "b.py"}, {"content": "x"})
        behavior.on_tool_call("run_bash", {"command": "pytest"}, {"success": True})

        # No loops should be detected
        assert len(behavior.loop_warnings) == 0

    def test_enhance_context_injects_warnings(self):
        """Test enhance_context injects loop warnings into context."""
        behavior = LoopDetectionBehavior(max_repeats=2)

        # Trigger loop detection
        for _ in range(5):
            behavior.on_tool_call(
                "write_file",
                {"path": "test.py"},
                {"error": "Same error"}
            )

        context = [{"role": "system", "content": "System prompt"}]

        result = behavior.enhance_context(context)

        # Should have warning injected after system prompt
        assert len(result) >= 2
        warning_msg = result[1]
        assert "LOOP DETECTION WARNING" in warning_msg["content"]
        assert "write_file" in warning_msg["content"]

    def test_enhance_context_no_warnings_when_no_loops(self):
        """Test enhance_context returns unchanged context when no loops."""
        behavior = LoopDetectionBehavior()

        # No tool calls, no loops
        context = [{"role": "system", "content": "System"}]

        result = behavior.enhance_context(context)

        # Should return unchanged
        assert result == context

    def test_enhance_context_limits_warnings_to_last_three(self):
        """Test enhance_context only shows last 3 warnings."""
        behavior = LoopDetectionBehavior(max_repeats=2)

        # Create multiple different loop warnings
        for i in range(5):
            tool_name = f"tool_{i}"
            for _ in range(5):
                behavior.on_tool_call(
                    tool_name,
                    {"arg": "value"},
                    {"error": "error"}
                )

        context = [{"role": "system", "content": "System"}]

        result = behavior.enhance_context(context)

        warning_msg = result[1]["content"]

        # Count how many tool names appear in warning
        tool_mentions = sum(1 for i in range(5) if f"tool_{i}" in warning_msg)

        # Should show at most 3 warnings
        assert tool_mentions <= 3

    def test_make_serializable_handles_primitives(self):
        """Test _make_serializable handles primitives correctly."""
        behavior = LoopDetectionBehavior()

        assert behavior._make_serializable(None) is None
        assert behavior._make_serializable(True) is True
        assert behavior._make_serializable(42) == 42
        assert behavior._make_serializable(3.14) == 3.14
        assert behavior._make_serializable("test") == "test"

    def test_make_serializable_handles_dicts(self):
        """Test _make_serializable handles nested dicts."""
        behavior = LoopDetectionBehavior()

        obj = {"a": 1, "b": {"c": 2, "d": 3}}
        result = behavior._make_serializable(obj)

        assert result == {"a": 1, "b": {"c": 2, "d": 3}}

    def test_make_serializable_handles_lists(self):
        """Test _make_serializable handles lists."""
        behavior = LoopDetectionBehavior()

        obj = [1, 2, [3, 4]]
        result = behavior._make_serializable(obj)

        assert result == [1, 2, [3, 4]]

    def test_make_serializable_handles_non_serializable_objects(self):
        """Test _make_serializable converts non-serializable objects to strings."""
        behavior = LoopDetectionBehavior()

        class CustomClass:
            pass

        obj = CustomClass()
        result = behavior._make_serializable(obj)

        assert isinstance(result, str)
        assert "CustomClass" in result

    def test_get_instructions_returns_loop_detection_info(self):
        """Test get_instructions returns loop detection guidance."""
        behavior = LoopDetectionBehavior()
        instructions = behavior.get_instructions()

        assert "LOOP DETECTION" in instructions
        assert "repeating" in instructions
        assert "different strategy" in instructions.lower()

    def test_action_signature_includes_args(self):
        """Test action signature includes serialized arguments."""
        behavior = LoopDetectionBehavior(max_repeats=2)

        # Same tool, different args - should not trigger loop
        behavior.on_tool_call("write_file", {"path": "a.py"}, {"success": True})
        behavior.on_tool_call("write_file", {"path": "b.py"}, {"success": True})
        behavior.on_tool_call("write_file", {"path": "c.py"}, {"success": True})

        # No loop should be detected (different args)
        assert len(behavior.loop_warnings) == 0

    def test_result_signature_includes_result_hash(self):
        """Test result signature includes result content hash."""
        behavior = LoopDetectionBehavior(max_repeats=2)

        # Same action, similar but not identical results
        behavior.on_tool_call("run_bash", {"cmd": "pytest"}, {"output": "1 passed"})
        behavior.on_tool_call("run_bash", {"cmd": "pytest"}, {"output": "2 passed"})
        behavior.on_tool_call("run_bash", {"cmd": "pytest"}, {"output": "3 passed"})

        # Should not trigger identical result loop
        warnings = [w for w in behavior.loop_warnings if "identical results" in w]
        assert len(warnings) == 0

    def test_loop_warnings_deduplication(self):
        """Test loop warnings are not duplicated."""
        behavior = LoopDetectionBehavior(max_repeats=2)

        # Trigger same loop multiple times
        for _ in range(10):
            behavior.on_tool_call("write_file", {"path": "test.py"}, {"error": "Same error"})

        # Should only have one warning (no duplicates)
        assert len(behavior.loop_warnings) == 1

    def test_on_tool_call_tracks_success_status(self):
        """Test on_tool_call correctly determines success status."""
        behavior = LoopDetectionBehavior()

        # Success case
        behavior.on_tool_call("tool1", {}, {"success": True})
        assert behavior.action_history[-1]["success"] is True

        # Error case (has "error" key)
        behavior.on_tool_call("tool2", {}, {"error": "Failed"})
        assert behavior.action_history[-1]["success"] is False

        # Explicit failure
        behavior.on_tool_call("tool3", {}, {"success": False})
        assert behavior.action_history[-1]["success"] is False

    def test_recent_history_window(self):
        """Test loop detection only considers recent 20 actions."""
        behavior = LoopDetectionBehavior(max_repeats=3)

        # Add 30 different actions
        for i in range(30):
            behavior.on_tool_call(f"tool_{i}", {}, {"result": i})

        # Now repeat the same action 5 times
        for _ in range(5):
            behavior.on_tool_call("repeated_tool", {"arg": "val"}, {"error": "Same"})

        # Should detect loop (only looking at recent 20 actions)
        assert len(behavior.loop_warnings) > 0
