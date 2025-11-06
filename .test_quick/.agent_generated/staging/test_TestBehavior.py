import pytest
from unittest.mock import Mock

from behaviors.testbehavior import TestbehaviorBehavior


class TestTestbehaviorBehavior:
    """Test suite for TestbehaviorBehavior."""

    def test_get_name(self):
        """Behavior returns correct identifier."""
        behavior = TestbehaviorBehavior()
        assert behavior.get_name() == "TestBehavior"

    def test_initialization(self):
        """Behavior initializes without errors."""
        behavior = TestbehaviorBehavior()
        assert behavior is not None

    def test_tool_schema(self):
        """Tool schemas are well‑formed."""
        behavior = TestbehaviorBehavior()
        tools = behavior.get_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0
        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            # Validate the tool name matches the expected tool
            assert tool["function"]["name"] == "test_tool"
            # Validate required parameter
            params = tool["function"]["parameters"]["properties"]
            assert "param1" in params
            assert params["param1"]["type"] == "string"
            assert params["param1"]["description"] == "Test param"
            assert "param1" in tool["function"]["parameters"]["required"]

    def test_tool_dispatch_success(self):
        """Tool dispatch returns expected result when called with valid arguments."""
        behavior = TestbehaviorBehavior()
        mock_agent = Mock()

        result = behavior.dispatch_tool(
            agent=mock_agent,
            tool_name="test_tool",
            args={"param1": "test value"}
        )

        assert isinstance(result, dict)
        # The behavior may return a 'result' or 'success' key
        assert "result" in result or "success" in result

    def test_tool_dispatch_unknown_tool(self):
        """Unknown tools raise NotImplementedError."""
        behavior = TestbehaviorBehavior()
        mock_agent = Mock()

        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool(
                agent=mock_agent,
                tool_name="unknown_tool",
                args={}
            )

    def test_tool_dispatch_missing_required_param(self):
        """Dispatching without required parameters raises ValueError."""
        behavior = TestbehaviorBehavior()
        mock_agent = Mock()

        with pytest.raises(ValueError):
            behavior.dispatch_tool(
                agent=mock_agent,
                tool_name="test_tool",
                args={}  # missing 'param1'
            )

    def test_tool_dispatch_wrong_param_type(self):
        """Dispatching with wrong parameter type raises TypeError."""
        behavior = TestbehaviorBehavior()
        mock_agent = Mock()

        with pytest.raises(TypeError):
            behavior.dispatch_tool(
                agent=mock_agent,
                tool_name="test_tool",
                args={"param1": 123}  # should be string
            )

    def test_initial_context_injection(self):
        """on_initial_context injects expected information (called ONCE)."""
        behavior = TestbehaviorBehavior()
        mock_agent = Mock()
        mock_agent.goal = "Test goal"

        context = [
            {"role": "system", "content": "System prompt"}
        ]

        enhanced = behavior.on_initial_context(agent=mock_agent, context=context)

        assert isinstance(enhanced, list)
        assert len(enhanced) > len(context)
        # Ensure the agent's goal is present in the injected messages
        assert any(
            "Test goal" in msg.get("content", "") for msg in enhanced
        )

    def test_round_start_context_injection(self):
        """on_round_start injects dynamic information (called EVERY round)."""
        behavior = TestbehaviorBehavior()
        mock_agent = Mock()

        context = [
            {"role": "system", "content": "System prompt"}
        ]

        enhanced = behavior.on_round_start(
            agent=mock_agent,
            round_number=1,
            context=context
        )

        assert isinstance(enhanced, list)
        # The behavior may or may not modify the context; ensure it returns a list
        assert len(enhanced) >= len(context)