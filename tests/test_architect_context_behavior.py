"""
Tests for ArchitectContextBehavior.

Tests:
- Architecture context injection
- Higher token limit
- Architecture-focused summarization
"""

import pytest
from unittest.mock import Mock, patch
from behaviors.architect_context import ArchitectContextBehavior


class TestArchitectContextBehavior:
    """Test suite for ArchitectContextBehavior."""

    def test_get_name(self):
        """Test behavior returns correct name."""
        behavior = ArchitectContextBehavior()
        assert behavior.get_name() == "architect_context"

    def test_initialization_with_defaults(self):
        """Test behavior initializes with default parameters."""
        behavior = ArchitectContextBehavior()
        assert behavior.max_tokens == 32000
        assert behavior.compact_threshold == 0.75
        assert behavior.keep_recent == 20

    def test_initialization_with_custom_params(self):
        """Test behavior initializes with custom parameters."""
        behavior = ArchitectContextBehavior(
            max_tokens=64000,
            compact_threshold=0.8,
            keep_recent=30
        )
        assert behavior.max_tokens == 64000
        assert behavior.compact_threshold == 0.8
        assert behavior.keep_recent == 30

    def test_enhance_context_injects_project_info(self):
        """Test 'PROJECT' header is injected."""
        behavior = ArchitectContextBehavior()

        # Mock context manager
        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Design microservices architecture"
        mock_cm.state.goal = mock_goal

        context = [{"role": "system", "content": "System prompt"}]

        result = behavior.enhance_context(context, context_manager=mock_cm)

        # Should have system + project info
        assert len(result) >= 2
        project_msg = result[1]
        assert "PROJECT: Design microservices architecture" in project_msg["content"]

    def test_enhance_context_no_context_manager(self):
        """Test enhance_context returns unchanged if no context manager."""
        behavior = ArchitectContextBehavior()

        context = [{"role": "system", "content": "System"}]

        result = behavior.enhance_context(context)

        assert result == context

    def test_enhance_context_triggers_compaction(self):
        """Test compaction triggers when context exceeds threshold."""
        behavior = ArchitectContextBehavior(max_tokens=100, keep_recent=2)

        # Create large context that will exceed threshold
        messages = [
            {"role": "user", "content": "x" * 1000} for _ in range(50)
        ]

        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test project"
        mock_cm.state.goal = mock_goal

        context = [
            {"role": "system", "content": "System"},
        ] + messages

        with patch.object(behavior, '_summarize_architecture_messages', return_value="Architecture summary"):
            result = behavior.enhance_context(context, context_manager=mock_cm)

            # Should have compacted
            assert any("Architecture summary" in msg.get("content", "") for msg in result)

    def test_summarize_architecture_messages_calls_llm(self):
        """Test _summarize_architecture_messages calls LLM with architecture focus."""
        behavior = ArchitectContextBehavior()
        messages = [
            {"role": "user", "content": "What architecture pattern should we use?"},
            {"role": "assistant", "content": "Microservices with event-driven communication"}
        ]

        mock_response = {
            "message": {
                "content": "- Microservices pattern chosen\n- Event-driven communication"
            }
        }

        with patch('behaviors.architect_context.chat_with_inactivity_timeout', return_value=mock_response) as mock_chat:
            result = behavior._summarize_architecture_messages(messages)

            # Check LLM was called with qwen3:14b
            assert mock_chat.called
            call_args = mock_chat.call_args
            assert call_args[1]["model"] == "qwen3:14b"

            # Check result
            assert "Microservices pattern" in result

    def test_summarize_architecture_messages_handles_failure(self):
        """Test _summarize_architecture_messages handles LLM failures."""
        behavior = ArchitectContextBehavior()
        messages = [{"role": "user", "content": "Test"}]

        with patch('behaviors.architect_context.chat_with_inactivity_timeout', side_effect=Exception("Error")):
            result = behavior._summarize_architecture_messages(messages)

            assert "Summarization failed" in result

    def test_estimate_context_size(self):
        """Test token estimation."""
        behavior = ArchitectContextBehavior()
        context = [
            {"role": "system", "content": "a" * 400},
            {"role": "user", "content": "b" * 800},
        ]

        estimated = behavior._estimate_context_size(context)

        assert estimated == 300

    def test_get_instructions_returns_architecture_workflow(self):
        """Test get_instructions returns architecture design workflow."""
        behavior = ArchitectContextBehavior()
        instructions = behavior.get_instructions()

        assert "ARCHITECTURE DESIGN WORKFLOW" in instructions
        assert "architecture" in instructions.lower()
        assert "artifacts" in instructions
        assert "write_architecture_doc" in instructions

    def test_enhance_context_uses_high_safety_threshold(self):
        """Test compaction uses 128K as upper bound for safety."""
        behavior = ArchitectContextBehavior(max_tokens=200000)  # Higher than 128K

        # Create very large context
        messages = [
            {"role": "user", "content": "x" * 1000} for _ in range(300)
        ]

        mock_cm = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test"
        mock_cm.state.goal = mock_goal

        context = [{"role": "system", "content": "System"}] + messages

        with patch.object(behavior, '_summarize_architecture_messages', return_value="Summary"):
            result = behavior.enhance_context(context, context_manager=mock_cm, capsys=None)

            # Should have triggered compaction (using 131072 as upper bound)
            assert any("Summary" in msg.get("content", "") for msg in result)

    def test_summarize_architecture_messages_keeps_longer_content(self):
        """Test architecture summarization keeps longer content than standard."""
        behavior = ArchitectContextBehavior()
        messages = [
            {"role": "user", "content": "x" * 500}  # Long content
        ]

        with patch('behaviors.architect_context.chat_with_inactivity_timeout', return_value={"message": {"content": "Summary"}}) as mock_chat:
            behavior._summarize_architecture_messages(messages)

            # Check that prompt contains longer content (400 chars for non-tool)
            call_args = mock_chat.call_args
            prompt = call_args[1]["messages"][0]["content"]

            # Architecture summaries should keep more content
            assert len(prompt) > 200  # Should have more than standard summarization
