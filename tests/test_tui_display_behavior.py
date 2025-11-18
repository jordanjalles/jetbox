"""
Tests for TUIDisplayBehavior.

Verifies that the TUI display behavior correctly integrates with the
behavior framework and properly wraps DisplayFactory/DisplayInterface.
"""

from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from behaviors.tui_display_behavior import TUIDisplayBehavior
from tui import EventType


class TestTUIDisplayBehavior:
    """Test suite for TUIDisplayBehavior."""

    def test_get_name(self):
        """Behavior should return correct name."""
        behavior = TUIDisplayBehavior()
        assert behavior.get_name() == "tui_display"

    def test_rule_of_two_properties(self):
        """Behavior should have empty Rule of Two properties (utility behavior)."""
        behavior = TUIDisplayBehavior()
        assert behavior.rule_of_two_properties == set()

    def test_display_mode_default(self):
        """Default display mode should be 'auto'."""
        behavior = TUIDisplayBehavior()
        assert behavior.display_mode == "auto"

    def test_display_mode_custom(self):
        """Should accept custom display mode."""
        behavior = TUIDisplayBehavior(display_mode="plain")
        assert behavior.display_mode == "plain"

    @patch('behaviors.tui_display_behavior.DisplayFactory')
    def test_on_goal_start(self, mock_factory):
        """on_goal_start should create and start display."""
        # Setup
        mock_display = Mock()
        mock_factory.create.return_value = mock_display
        behavior = TUIDisplayBehavior(display_mode="plain")
        agent = Mock()
        goal = "Test goal"

        # Execute
        behavior.on_goal_start(agent, goal)

        # Verify
        mock_factory.create.assert_called_once_with(force_mode="plain")
        mock_display.start.assert_called_once()
        assert behavior.display is mock_display
        assert behavior.goal == goal
        assert behavior.start_time is not None

    @patch('behaviors.tui_display_behavior.DisplayFactory')
    def test_on_round_start(self, mock_factory):
        """on_round_start should update status display."""
        # Setup
        mock_display = Mock()
        mock_factory.create.return_value = mock_display
        behavior = TUIDisplayBehavior()
        agent = Mock()
        agent.goal = "Test goal"
        agent.name = "test_agent"
        agent.model = "qwen3:8b"
        agent.max_rounds = 50

        # Initialize behavior
        behavior.on_goal_start(agent, "Test goal")
        context = [{"role": "system", "content": "System prompt"}]

        # Execute
        result = behavior.on_round_start(agent, 1, context)

        # Verify
        assert result == context  # Context unchanged
        mock_display.update_status.assert_called_once()
        call_args = mock_display.update_status.call_args[1]
        assert call_args['goal'] == "Test goal"
        assert call_args['agent_name'] == "test_agent"
        assert call_args['model'] == "qwen3:8b"
        assert call_args['current_round'] == 1
        assert call_args['max_rounds'] == 50
        assert call_args['status'] == "Running"

    @patch('behaviors.tui_display_behavior.DisplayFactory')
    def test_on_tool_call_success(self, mock_factory):
        """on_tool_call should log tool call and success."""
        # Setup
        mock_display = Mock()
        mock_factory.create.return_value = mock_display
        behavior = TUIDisplayBehavior()
        agent = Mock()
        behavior.on_goal_start(agent, "Test goal")

        # Execute
        behavior.on_tool_call(
            agent,
            "write_file",
            {"path": "test.txt", "content": "hello"},
            {"success": True, "result": "File written"}
        )

        # Verify - should log 2 events (tool call + result)
        assert mock_display.log_event.call_count == 2

        # First call: tool call
        first_call = mock_display.log_event.call_args_list[0][0][0]
        assert first_call.type == EventType.TOOL_CALL
        assert "write_file" in first_call.message

        # Second call: success
        second_call = mock_display.log_event.call_args_list[1][0][0]
        assert second_call.type == EventType.SUCCESS

    @patch('behaviors.tui_display_behavior.DisplayFactory')
    def test_on_tool_call_error(self, mock_factory):
        """on_tool_call should log tool call and error."""
        # Setup
        mock_display = Mock()
        mock_factory.create.return_value = mock_display
        behavior = TUIDisplayBehavior()
        agent = Mock()
        behavior.on_goal_start(agent, "Test goal")

        # Execute
        behavior.on_tool_call(
            agent,
            "read_file",
            {"path": "missing.txt"},
            {"error": "File not found"}
        )

        # Verify
        assert mock_display.log_event.call_count == 2

        # Second call should be ERROR
        second_call = mock_display.log_event.call_args_list[1][0][0]
        assert second_call.type == EventType.ERROR
        assert "File not found" in second_call.message

    @patch('behaviors.tui_display_behavior.DisplayFactory')
    def test_on_round_end_with_pause(self, mock_factory):
        """on_round_end should check for pause if display supports it."""
        # Setup
        mock_display = Mock()
        mock_display.can_pause.return_value = True
        mock_factory.create.return_value = mock_display
        behavior = TUIDisplayBehavior()
        agent = Mock()
        behavior.on_goal_start(agent, "Test goal")

        # Execute
        behavior.on_round_end(agent, 1)

        # Verify
        mock_display.can_pause.assert_called_once()
        mock_display.wait_if_paused.assert_called_once()

    @patch('behaviors.tui_display_behavior.DisplayFactory')
    def test_on_round_end_no_pause(self, mock_factory):
        """on_round_end should skip pause check if not supported."""
        # Setup
        mock_display = Mock()
        mock_display.can_pause.return_value = False
        mock_factory.create.return_value = mock_display
        behavior = TUIDisplayBehavior()
        agent = Mock()
        behavior.on_goal_start(agent, "Test goal")

        # Execute
        behavior.on_round_end(agent, 1)

        # Verify
        mock_display.can_pause.assert_called_once()
        mock_display.wait_if_paused.assert_not_called()

    @patch('behaviors.tui_display_behavior.DisplayFactory')
    def test_on_timeout(self, mock_factory):
        """on_timeout should log timeout and stop display."""
        # Setup
        mock_display = Mock()
        mock_factory.create.return_value = mock_display
        behavior = TUIDisplayBehavior()
        agent = Mock()
        behavior.on_goal_start(agent, "Test goal")

        # Execute
        behavior.on_timeout(agent, 30.5)

        # Verify
        mock_display.log_event.assert_called_once()
        event = mock_display.log_event.call_args[0][0]
        assert event.type == EventType.ERROR
        assert "timed out" in event.message.lower()
        mock_display.stop.assert_called_once()

    @patch('behaviors.tui_display_behavior.DisplayFactory')
    def test_on_goal_complete_success(self, mock_factory):
        """on_goal_complete should show completion and stop display."""
        # Setup
        mock_display = Mock()
        mock_factory.create.return_value = mock_display
        behavior = TUIDisplayBehavior()
        agent = Mock()
        agent.workspace = Path("/tmp/test_workspace")
        # Mock workspace_manager to avoid iteration error
        agent.workspace_manager = Mock()
        agent.workspace_manager.created_files = ["file1.py", "file2.py"]
        behavior.on_goal_start(agent, "Test goal")

        # Execute
        behavior.on_goal_complete(agent, success=True, summary="All tests passed")

        # Verify
        mock_display.show_completion.assert_called_once()
        call_args = mock_display.show_completion.call_args[1]
        assert call_args['success'] is True
        assert call_args['summary'] == "All tests passed"
        assert call_args['duration'] > 0
        mock_display.stop.assert_called_once()

    @patch('behaviors.tui_display_behavior.DisplayFactory')
    def test_on_goal_complete_failure(self, mock_factory):
        """on_goal_complete should handle failure correctly."""
        # Setup
        mock_display = Mock()
        mock_factory.create.return_value = mock_display
        behavior = TUIDisplayBehavior()
        agent = Mock()
        agent.workspace = Path("/tmp/test_workspace")
        # Mock workspace_manager to avoid iteration error
        agent.workspace_manager = Mock()
        agent.workspace_manager.created_files = []
        behavior.on_goal_start(agent, "Test goal")

        # Execute
        behavior.on_goal_complete(agent, success=False, summary="Task failed")

        # Verify
        mock_display.show_completion.assert_called_once()
        call_args = mock_display.show_completion.call_args[1]
        assert call_args['success'] is False
        assert call_args['summary'] == "Task failed"
        mock_display.stop.assert_called_once()

    def test_get_instructions(self):
        """Should return TUI instructions."""
        behavior = TUIDisplayBehavior()
        instructions = behavior.get_instructions()
        assert "TUI DISPLAY" in instructions
        assert "progress" in instructions.lower()
        assert "automatically" in instructions.lower()

    def test_format_args_simple(self):
        """_format_args should format simple arguments."""
        behavior = TUIDisplayBehavior()
        args = {"path": "test.txt", "mode": "w"}
        result = behavior._format_args(args)
        assert "path=test.txt" in result
        assert "mode=w" in result

    def test_format_args_truncate(self):
        """_format_args should truncate long values."""
        behavior = TUIDisplayBehavior()
        long_content = "x" * 100
        args = {"content": long_content}
        result = behavior._format_args(args, max_length=50)
        assert len(result) < len(f"content={long_content}")
        assert "..." in result

    def test_analyze_result_dict_success(self):
        """_analyze_result should handle dict success."""
        behavior = TUIDisplayBehavior()
        result = {"success": True, "result": "Operation completed"}
        event_type, message, details = behavior._analyze_result("test_tool", result)
        assert event_type == EventType.SUCCESS
        assert "Operation completed" in message

    def test_analyze_result_dict_error(self):
        """_analyze_result should handle dict error."""
        behavior = TUIDisplayBehavior()
        result = {"error": "Something went wrong"}
        event_type, message, details = behavior._analyze_result("test_tool", result)
        assert event_type == EventType.ERROR
        assert "Something went wrong" in message

    def test_analyze_result_string_error(self):
        """_analyze_result should detect error in string."""
        behavior = TUIDisplayBehavior()
        result = "Error: File not found"
        event_type, message, details = behavior._analyze_result("test_tool", result)
        assert event_type == EventType.ERROR
        assert "File not found" in message

    def test_analyze_result_string_success(self):
        """_analyze_result should handle string success."""
        behavior = TUIDisplayBehavior()
        result = "File created successfully"
        event_type, message, details = behavior._analyze_result("test_tool", result)
        assert event_type == EventType.SUCCESS
        assert "created successfully" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
