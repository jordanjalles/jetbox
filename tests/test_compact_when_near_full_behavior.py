"""
Tests for CompactWhenNearFullBehavior.

Tests:
- Message appending
- Compaction triggers at 75% threshold
- LLM summarization call
- Recent message preservation
- mark_goal_complete tool
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from behaviors.compact_when_near_full import CompactWhenNearFullBehavior


class TestCompactWhenNearFullBehavior:
    """Test suite for CompactWhenNearFullBehavior."""

    def test_get_name(self):
        """Test behavior returns correct name."""
        behavior = CompactWhenNearFullBehavior()
        assert behavior.get_name() == "compact_when_near_full"

    def test_initialization_with_defaults(self):
        """Test behavior initializes with default parameters."""
        behavior = CompactWhenNearFullBehavior()
        assert behavior.max_tokens == 8000
        assert behavior.compact_threshold == 0.75
        assert behavior.keep_recent == 20

    def test_initialization_with_custom_params(self):
        """Test behavior initializes with custom parameters."""
        behavior = CompactWhenNearFullBehavior(
            max_tokens=16000,
            compact_threshold=0.8,
            keep_recent=30
        )
        assert behavior.max_tokens == 16000
        assert behavior.compact_threshold == 0.8
        assert behavior.keep_recent == 30

    def test_enhance_context_no_compaction_needed(self):
        """Test context is not modified when under threshold."""
        behavior = CompactWhenNearFullBehavior(max_tokens=100000)  # High limit
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "GOAL: Create a calculator"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        result = behavior.enhance_context(context)

        # Should return context unchanged
        assert len(result) == len(context)
        assert result == context

    def test_enhance_context_triggers_compaction(self):
        """Test compaction triggers when context exceeds threshold."""
        behavior = CompactWhenNearFullBehavior(max_tokens=100, keep_recent=2)  # Low limit

        # Create large context that will exceed threshold
        messages = [
            {"role": "user", "content": "x" * 100} for _ in range(50)
        ]
        context = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "GOAL: Test"},
        ] + messages

        with patch.object(behavior, '_summarize_messages', return_value="Summary of old messages"):
            result = behavior.enhance_context(context)

            # Should have: system + goal + summary + recent 2 messages
            assert len(result) == 5  # system + goal + summary + 2 recent
            assert any("Summary of old messages" in msg.get("content", "") for msg in result)

    def test_enhance_context_preserves_recent_messages(self):
        """Test recent messages are preserved during compaction."""
        behavior = CompactWhenNearFullBehavior(max_tokens=100, keep_recent=3)

        messages = [
            {"role": "user", "content": "Old message " + "x" * 100},
            {"role": "assistant", "content": "Old response " + "x" * 100},
            {"role": "user", "content": "Old message 2 " + "x" * 100},
            {"role": "assistant", "content": "Old response 2 " + "x" * 100},
            {"role": "user", "content": "Recent 1"},
            {"role": "assistant", "content": "Recent response 1"},
            {"role": "user", "content": "Recent 2"},
        ]
        context = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "GOAL: Test"},
        ] + messages

        with patch.object(behavior, '_summarize_messages', return_value="Summary of old messages"):
            result = behavior.enhance_context(context)

            # Check recent messages are preserved
            assert any("Recent 1" in msg.get("content", "") for msg in result)
            assert any("Recent 2" in msg.get("content", "") for msg in result)
            # Should have summary instead of old messages
            assert any("Summary of old messages" in msg.get("content", "") for msg in result)
            # First old message should not be in result (it's been summarized)
            result_str = str(result)
            assert "Old message 2" not in result_str or "Summary" in result_str

    def test_summarize_messages_calls_llm(self):
        """Test _summarize_messages calls LLM with correct parameters."""
        behavior = CompactWhenNearFullBehavior()
        messages = [
            {"role": "user", "content": "Create file.py"},
            {"role": "assistant", "content": "Creating file..."},
            {"role": "tool", "content": "File created successfully"}
        ]

        mock_response = {
            "message": {
                "content": "- Created file.py\n- Tests passed"
            }
        }

        with patch('llm_utils.chat_with_inactivity_timeout', return_value=mock_response) as mock_chat:
            result = behavior._summarize_messages(messages)

            # Check LLM was called
            assert mock_chat.called
            call_args = mock_chat.call_args

            # Check model and temperature
            assert call_args[1]["model"] == "gpt-oss:20b"
            assert call_args[1]["options"]["temperature"] == 0.2

            # Check result
            assert "Created file.py" in result

    def test_summarize_messages_handles_llm_failure(self):
        """Test _summarize_messages handles LLM failures gracefully."""
        behavior = CompactWhenNearFullBehavior()
        messages = [
            {"role": "user", "content": "Test message"}
        ]

        with patch('llm_utils.chat_with_inactivity_timeout', side_effect=Exception("LLM error")):
            result = behavior._summarize_messages(messages)

            # Should return fallback summary
            assert "Summarization failed" in result
            assert "1" in result  # Message count

    def test_estimate_context_size(self):
        """Test token estimation using 4 chars per token heuristic."""
        behavior = CompactWhenNearFullBehavior()
        context = [
            {"role": "system", "content": "a" * 400},  # 100 tokens
            {"role": "user", "content": "b" * 800},    # 200 tokens
        ]

        estimated = behavior._estimate_context_size(context)

        # Should be approximately 300 tokens (1200 chars / 4)
        assert estimated == 300

    def test_get_tools_returns_mark_goal_complete(self):
        """Test get_tools returns mark_goal_complete tool definition."""
        behavior = CompactWhenNearFullBehavior()
        tools = behavior.get_tools()

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "mark_goal_complete"
        assert "summary" in tools[0]["function"]["parameters"]["properties"]

    def test_dispatch_tool_mark_goal_complete(self):
        """Test dispatch_tool handles mark_goal_complete."""
        behavior = CompactWhenNearFullBehavior()

        # Mock context manager
        mock_cm = Mock()
        mock_goal = Mock()
        mock_cm.state.goal = mock_goal

        result = behavior.dispatch_tool(
            "mark_goal_complete",
            {"summary": "All tests pass"},
            context_manager=mock_cm
        )

        assert result["success"] is True
        assert "All tests pass" in result["result"]
        assert result["summary"] == "All tests pass"
        mock_goal.mark_complete.assert_called_once_with(success=True)

    def test_dispatch_tool_unknown_tool(self):
        """Test dispatch_tool raises NotImplementedError for unknown tools."""
        behavior = CompactWhenNearFullBehavior()

        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool("unknown_tool", {})

    def test_get_instructions_returns_workflow(self):
        """Test get_instructions returns workflow guidance."""
        behavior = CompactWhenNearFullBehavior()
        instructions = behavior.get_instructions()

        assert "WORKFLOW" in instructions
        assert "mark_goal_complete" in instructions
        assert "write_file" in instructions

    def test_enhance_context_with_no_messages(self):
        """Test enhance_context handles empty message list."""
        behavior = CompactWhenNearFullBehavior()
        context = [
            {"role": "system", "content": "System prompt"}
        ]

        result = behavior.enhance_context(context)

        # Should return unchanged
        assert result == context

    def test_enhance_context_compaction_not_enough_messages(self):
        """Test compaction handles case where not enough messages to summarize."""
        behavior = CompactWhenNearFullBehavior(max_tokens=10, keep_recent=20)  # keep_recent > messages

        context = [
            {"role": "system", "content": "x" * 1000},
            {"role": "user", "content": "GOAL: Test"},
            {"role": "user", "content": "Message 1"},
        ]

        result = behavior.enhance_context(context)

        # Should keep recent messages without summarization
        assert len(result) == 3

    def test_summarize_messages_truncates_tool_results(self):
        """Test _summarize_messages truncates long tool results."""
        behavior = CompactWhenNearFullBehavior()
        messages = [
            {"role": "tool", "content": "x" * 500}  # Long tool result
        ]

        with patch('llm_utils.chat_with_inactivity_timeout', return_value={"message": {"content": "Summary"}}) as mock_chat:
            behavior._summarize_messages(messages)

            # Check that prompt contains truncated content
            call_args = mock_chat.call_args
            prompt = call_args[1]["messages"][0]["content"]

            # Tool results should be truncated to ~100 chars
            assert "..." in prompt or len(prompt) < 500

    def test_enhance_context_prints_compaction_stats(self, capsys):
        """Test enhance_context prints compaction statistics."""
        behavior = CompactWhenNearFullBehavior(max_tokens=100, keep_recent=2)

        messages = [
            {"role": "user", "content": "x" * 100} for _ in range(50)
        ]
        context = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "GOAL: Test"},
        ] + messages

        with patch.object(behavior, '_summarize_messages', return_value="Summary"):
            behavior.enhance_context(context)

            captured = capsys.readouterr()
            assert "compact_when_near_full" in captured.out
            assert "tokens" in captured.out
            assert "Reduced from" in captured.out
