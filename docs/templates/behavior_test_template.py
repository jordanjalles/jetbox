"""
Tests for {BEHAVIOR_CLASS_NAME}Behavior.

Tests:
- Behavior identifier
- Tool schemas (if applicable)
- Tool dispatch (if applicable)
- Context enhancement (if applicable)
- Event handlers (if applicable)
- No cross-behavior dependencies
"""

import pytest
from unittest.mock import Mock
from behaviors.{BEHAVIOR_MODULE} import {BEHAVIOR_CLASS_NAME}Behavior


class Test{BEHAVIOR_CLASS_NAME}Behavior:
    """Test suite for {BEHAVIOR_CLASS_NAME}Behavior."""

    def test_get_name(self):
        """Behavior returns correct identifier."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        assert behavior.get_name() == "{BEHAVIOR_NAME}"

    def test_initialization(self):
        """Behavior initializes without errors."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        assert behavior is not None

    # ADD TESTS BELOW BASED ON WHAT THE BEHAVIOR DOES

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_tool_schema(self):
        """Tool schemas are well-formed."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        tools = behavior.get_tools()

        assert len(tools) > 0
        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_tool_dispatch_success(self):
        """Tool dispatch returns expected result."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        mock_agent = Mock()

        result = behavior.dispatch_tool(
            agent=mock_agent,
            tool_name="{TOOL_NAME}",
            args={"{PARAM_NAME}": "{TEST_VALUE}"}
        )

        assert "result" in result or "success" in result

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_tool_dispatch_unknown_tool(self):
        """Unknown tools raise NotImplementedError."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        mock_agent = Mock()

        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool(
                agent=mock_agent,
                tool_name="unknown_tool",
                args={}
            )

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_initial_context_injection(self):
        """on_initial_context injects expected information (called ONCE)."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        mock_agent = Mock()
        mock_agent.goal = "Test goal"

        context = [
            {"role": "system", "content": "System prompt"}
        ]

        enhanced = behavior.on_initial_context(agent=mock_agent, context=context)

        # Verify injection
        assert len(enhanced) > len(context)
        assert any("Test goal" in msg.get("content", "") for msg in enhanced)

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_round_start_context_injection(self):
        """on_round_start injects dynamic information (called EVERY round)."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        mock_agent = Mock()

        context = [
            {"role": "system", "content": "System prompt"}
        ]

        enhanced = behavior.on_round_start(
            agent=mock_agent,
            round_number=1,
            context=context
        )

        # Verify injection or no modification depending on behavior
        assert isinstance(enhanced, list)
