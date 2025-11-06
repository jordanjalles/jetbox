"""
Tests for SubAgentContextBehavior.

Tests:
- "DELEGATED GOAL" injection
- mark_complete/mark_failed tools
- Timeout nudging
- Higher token limit
"""

import pytest
from unittest.mock import Mock, patch
from behaviors.subagent_context import SubAgentContextBehavior


class TestSubAgentContextBehavior:
    """Test suite for SubAgentContextBehavior."""

    def test_get_name(self):
        """Test behavior returns correct name."""
        behavior = SubAgentContextBehavior()
        assert behavior.get_name() == "subagent_context"

    def test_initialization_with_defaults(self):
        """Test behavior initializes with default parameters."""
        behavior = SubAgentContextBehavior()
        assert behavior.max_tokens == 128000
        assert behavior.compact_threshold == 0.75
        assert behavior.keep_recent == 20

    def test_initialization_with_custom_params(self):
        """Test behavior initializes with custom parameters."""
        behavior = SubAgentContextBehavior(
            max_tokens=64000,
            compact_threshold=0.8,
            keep_recent=30
        )
        assert behavior.max_tokens == 64000
        assert behavior.compact_threshold == 0.8
        assert behavior.keep_recent == 30

    def test_enhance_context_injects_delegated_goal(self):
        """Test 'DELEGATED GOAL' header is injected."""
        behavior = SubAgentContextBehavior()

        # Mock context manager
        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Implement user authentication"
        mock_cm.state.goal = mock_goal

        context = [{"role": "system", "content": "System prompt"}]

        result = behavior.enhance_context(context, context_manager=mock_cm)

        # Should have system + delegated goal
        assert len(result) >= 2
        delegated_msg = result[1]
        assert "DELEGATED GOAL: Implement user authentication" in delegated_msg["content"]
        assert "orchestrator" in delegated_msg["content"]
        assert "mark_complete" in delegated_msg["content"]
        assert "mark_failed" in delegated_msg["content"]

    def test_enhance_context_includes_jetbox_notes(self):
        """Test jetbox notes are included when available."""
        behavior = SubAgentContextBehavior()

        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test goal"
        mock_cm.state.goal = mock_goal

        context = [{"role": "system", "content": "System"}]

        # Mock _get_jetbox_notes
        with patch.object(behavior, '_get_jetbox_notes', return_value="Previous work summary"):
            result = behavior.enhance_context(
                context,
                context_manager=mock_cm,
                workspace="/workspace"
            )

            delegated_msg = result[1]
            assert "PREVIOUS WORK" in delegated_msg["content"]
            assert "Previous work summary" in delegated_msg["content"]

    def test_enhance_context_triggers_compaction(self):
        """Test compaction triggers when context exceeds threshold."""
        behavior = SubAgentContextBehavior(max_tokens=100, keep_recent=2)

        # Create large context
        messages = [
            {"role": "user", "content": "x" * 100} for _ in range(50)
        ]

        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test"
        mock_cm.state.goal = mock_goal

        context = [
            {"role": "system", "content": "System"},
        ] + messages

        with patch.object(behavior, '_summarize_messages', return_value="Summary"):
            result = behavior.enhance_context(context, context_manager=mock_cm)

            # Should have compacted
            assert any("Summary" in msg.get("content", "") for msg in result)

    def test_get_tools_returns_completion_tools(self):
        """Test get_tools returns mark_complete and mark_failed."""
        behavior = SubAgentContextBehavior()
        tools = behavior.get_tools()

        assert len(tools) == 2
        tool_names = [t["function"]["name"] for t in tools]
        assert "mark_complete" in tool_names
        assert "mark_failed" in tool_names

    def test_dispatch_tool_mark_complete(self):
        """Test dispatch_tool handles mark_complete."""
        behavior = SubAgentContextBehavior()

        mock_cm = Mock()
        mock_goal = Mock()
        mock_cm.state.goal = mock_goal

        result = behavior.dispatch_tool(
            "mark_complete",
            {"summary": "Authentication implemented successfully"},
            context_manager=mock_cm
        )

        assert result["success"] is True
        assert "Authentication implemented successfully" in result["summary"]
        mock_goal.mark_complete.assert_called_once_with(success=True)

    def test_dispatch_tool_mark_failed(self):
        """Test dispatch_tool handles mark_failed."""
        behavior = SubAgentContextBehavior()

        mock_cm = Mock()
        mock_goal = Mock()
        mock_cm.state.goal = mock_goal

        result = behavior.dispatch_tool(
            "mark_failed",
            {"reason": "Missing required dependencies"},
            context_manager=mock_cm
        )

        assert result["success"] is False
        assert "Missing required dependencies" in result["reason"]
        mock_goal.mark_complete.assert_called_once_with(success=False)

    def test_dispatch_tool_unknown_tool(self):
        """Test dispatch_tool raises NotImplementedError for unknown tools."""
        behavior = SubAgentContextBehavior()

        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool("unknown_tool", {})

    def test_get_instructions_returns_subagent_workflow(self):
        """Test get_instructions returns sub-agent workflow."""
        behavior = SubAgentContextBehavior()
        instructions = behavior.get_instructions()

        assert "SUB-AGENT WORKFLOW" in instructions
        assert "delegated" in instructions
        assert "mark_complete" in instructions
        assert "mark_failed" in instructions
        assert "MUST SIGNAL COMPLETION" in instructions

    def test_enhance_context_no_context_manager(self):
        """Test enhance_context returns unchanged if no context manager."""
        behavior = SubAgentContextBehavior()

        context = [{"role": "system", "content": "System"}]

        result = behavior.enhance_context(context)

        assert result == context

    def test_get_jetbox_notes_returns_none_on_error(self):
        """Test _get_jetbox_notes returns None on import error."""
        behavior = SubAgentContextBehavior()

        with patch('behaviors.subagent_context.jetbox_notes', side_effect=ImportError):
            result = behavior._get_jetbox_notes()

            assert result is None

    def test_estimate_context_size(self):
        """Test token estimation."""
        behavior = SubAgentContextBehavior()
        context = [
            {"role": "system", "content": "a" * 400},  # 100 tokens
            {"role": "user", "content": "b" * 800},    # 200 tokens
        ]

        estimated = behavior._estimate_context_size(context)

        assert estimated == 300

    def test_summarize_messages_handles_failure(self):
        """Test _summarize_messages handles LLM failures."""
        behavior = SubAgentContextBehavior()
        messages = [{"role": "user", "content": "Test"}]

        with patch('behaviors.subagent_context.chat_with_inactivity_timeout', side_effect=Exception("Error")):
            result = behavior._summarize_messages(messages)

            assert "Summarization failed" in result
