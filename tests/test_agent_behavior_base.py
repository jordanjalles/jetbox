"""
Unit tests for AgentBehavior base class.

Tests the interface, default implementations, and abstract method enforcement.
"""

import pytest
from typing import Any

from behaviors import AgentBehavior


class MinimalBehavior(AgentBehavior):
    """Minimal concrete implementation with only get_name() implemented."""

    def get_name(self) -> str:
        return "minimal"


class FullBehavior(AgentBehavior):
    """Full implementation overriding all methods."""

    def __init__(self) -> None:
        self.events_received: list[str] = []
        self.context_enhance_count = 0
        self.tool_dispatch_count = 0

    def get_name(self) -> str:
        return "full"

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.context_enhance_count += 1
        # Add a marker to prove we were called
        if context and context[0]["role"] == "system":
            context[0]["content"] += "\n[Enhanced by FullBehavior]"
        return context

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg": {"type": "string"}
                        },
                        "required": ["arg"]
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        self.tool_dispatch_count += 1
        if tool_name == "test_tool":
            return {"result": f"Processed: {args.get('arg', 'none')}"}
        return super().dispatch_tool(tool_name, args, **kwargs)

    def get_instructions(self) -> str:
        return "Test instructions for FullBehavior"

    def on_goal_start(self, goal: str, **kwargs: Any) -> None:
        self.events_received.append(f"goal_start:{goal}")

    def on_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        **kwargs: Any
    ) -> None:
        self.events_received.append(f"tool_call:{tool_name}")

    def on_round_end(self, round_number: int, **kwargs: Any) -> None:
        self.events_received.append(f"round_end:{round_number}")

    def on_timeout(self, elapsed_seconds: float, **kwargs: Any) -> None:
        self.events_received.append(f"timeout:{elapsed_seconds}")

    def on_goal_complete(self, success: bool, **kwargs: Any) -> None:
        status = "success" if success else "failure"
        self.events_received.append(f"goal_complete:{status}")


class TestAgentBehaviorBase:
    """Test AgentBehavior base class."""

    def test_cannot_instantiate_abstract_base(self) -> None:
        """Cannot instantiate AgentBehavior directly."""
        with pytest.raises(TypeError):
            AgentBehavior()  # type: ignore[abstract]

    def test_minimal_behavior_requires_get_name(self) -> None:
        """Subclass must implement get_name()."""
        # This should fail because get_name() is not implemented
        with pytest.raises(TypeError):
            class IncompleteBehavior(AgentBehavior):  # noqa: B903
                pass
            IncompleteBehavior()  # type: ignore[abstract]

    def test_minimal_behavior_creation(self) -> None:
        """Can create minimal behavior with just get_name()."""
        behavior = MinimalBehavior()
        assert behavior.get_name() == "minimal"

    def test_default_enhance_context(self) -> None:
        """Default enhance_context returns context unchanged."""
        behavior = MinimalBehavior()
        context = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"}
        ]
        result = behavior.enhance_context(context)
        assert result == context
        assert result is context  # Should return same object

    def test_default_get_tools(self) -> None:
        """Default get_tools returns empty list."""
        behavior = MinimalBehavior()
        assert behavior.get_tools() == []

    def test_default_dispatch_tool_raises(self) -> None:
        """Default dispatch_tool raises NotImplementedError."""
        behavior = MinimalBehavior()
        with pytest.raises(NotImplementedError) as exc_info:
            behavior.dispatch_tool("unknown_tool", {})
        assert "minimal" in str(exc_info.value)
        assert "unknown_tool" in str(exc_info.value)

    def test_default_get_instructions(self) -> None:
        """Default get_instructions returns empty string."""
        behavior = MinimalBehavior()
        assert behavior.get_instructions() == ""

    def test_default_event_handlers_do_nothing(self) -> None:
        """Default event handlers can be called and do nothing."""
        behavior = MinimalBehavior()

        # Should not raise
        behavior.on_goal_start("test goal")
        behavior.on_tool_call("tool", {}, {})
        behavior.on_round_end(1)
        behavior.on_timeout(10.5)
        behavior.on_goal_complete(True)

    def test_full_behavior_get_name(self) -> None:
        """FullBehavior returns correct name."""
        behavior = FullBehavior()
        assert behavior.get_name() == "full"

    def test_full_behavior_enhance_context(self) -> None:
        """FullBehavior can modify context."""
        behavior = FullBehavior()
        context = [
            {"role": "system", "content": "Original prompt"},
            {"role": "user", "content": "Message"}
        ]
        result = behavior.enhance_context(context)

        assert behavior.context_enhance_count == 1
        assert "[Enhanced by FullBehavior]" in result[0]["content"]
        assert "Original prompt" in result[0]["content"]

    def test_full_behavior_get_tools(self) -> None:
        """FullBehavior returns tool definitions."""
        behavior = FullBehavior()
        tools = behavior.get_tools()

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "test_tool"
        assert "description" in tools[0]["function"]
        assert "parameters" in tools[0]["function"]

    def test_full_behavior_dispatch_tool(self) -> None:
        """FullBehavior can dispatch its tools."""
        behavior = FullBehavior()
        result = behavior.dispatch_tool("test_tool", {"arg": "hello"})

        assert behavior.tool_dispatch_count == 1
        assert result["result"] == "Processed: hello"

    def test_full_behavior_dispatch_unknown_tool(self) -> None:
        """FullBehavior raises for unknown tools."""
        behavior = FullBehavior()
        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool("unknown_tool", {})

    def test_full_behavior_get_instructions(self) -> None:
        """FullBehavior returns instructions."""
        behavior = FullBehavior()
        instructions = behavior.get_instructions()
        assert instructions == "Test instructions for FullBehavior"

    def test_full_behavior_on_goal_start(self) -> None:
        """FullBehavior receives goal start events."""
        behavior = FullBehavior()
        behavior.on_goal_start("Create a calculator")

        assert "goal_start:Create a calculator" in behavior.events_received

    def test_full_behavior_on_tool_call(self) -> None:
        """FullBehavior receives tool call events."""
        behavior = FullBehavior()
        behavior.on_tool_call(
            "write_file",
            {"path": "test.py"},
            {"success": True}
        )

        assert "tool_call:write_file" in behavior.events_received

    def test_full_behavior_on_round_end(self) -> None:
        """FullBehavior receives round end events."""
        behavior = FullBehavior()
        behavior.on_round_end(5)

        assert "round_end:5" in behavior.events_received

    def test_full_behavior_on_timeout(self) -> None:
        """FullBehavior receives timeout events."""
        behavior = FullBehavior()
        behavior.on_timeout(123.45)

        assert "timeout:123.45" in behavior.events_received

    def test_full_behavior_on_goal_complete_success(self) -> None:
        """FullBehavior receives goal complete (success) events."""
        behavior = FullBehavior()
        behavior.on_goal_complete(True)

        assert "goal_complete:success" in behavior.events_received

    def test_full_behavior_on_goal_complete_failure(self) -> None:
        """FullBehavior receives goal complete (failure) events."""
        behavior = FullBehavior()
        behavior.on_goal_complete(False)

        assert "goal_complete:failure" in behavior.events_received

    def test_kwargs_acceptance_enhance_context(self) -> None:
        """enhance_context accepts arbitrary kwargs."""
        behavior = FullBehavior()
        context = [{"role": "system", "content": "test"}]

        # Should not raise even with unexpected kwargs
        result = behavior.enhance_context(
            context,
            agent="mock_agent",
            workspace="/tmp/test",
            round_number=5,
            unexpected_param="should be ignored"
        )
        assert result is not None

    def test_kwargs_acceptance_dispatch_tool(self) -> None:
        """dispatch_tool accepts arbitrary kwargs."""
        behavior = FullBehavior()

        # Should not raise even with unexpected kwargs
        result = behavior.dispatch_tool(
            "test_tool",
            {"arg": "test"},
            agent="mock_agent",
            workspace="/tmp/test",
            unexpected_param="should be ignored"
        )
        assert result["result"] == "Processed: test"

    def test_kwargs_acceptance_events(self) -> None:
        """Event handlers accept arbitrary kwargs."""
        behavior = FullBehavior()

        # Should not raise even with unexpected kwargs
        behavior.on_goal_start(
            "test goal",
            agent="mock",
            workspace="/tmp",
            extra="ignored"
        )
        behavior.on_tool_call(
            "tool",
            {},
            {},
            agent="mock",
            extra="ignored"
        )
        behavior.on_round_end(1, agent="mock", extra="ignored")
        behavior.on_timeout(10.0, agent="mock", extra="ignored")
        behavior.on_goal_complete(True, agent="mock", extra="ignored")

        # All events should have been received
        assert len(behavior.events_received) == 5

    def test_multiple_enhance_context_calls(self) -> None:
        """enhance_context can be called multiple times."""
        behavior = FullBehavior()
        context = [{"role": "system", "content": "Original"}]

        result1 = behavior.enhance_context(context)
        result2 = behavior.enhance_context(result1)

        assert behavior.context_enhance_count == 2
        # Should have two enhancements
        assert result2[0]["content"].count("[Enhanced by FullBehavior]") == 2

    def test_event_sequence(self) -> None:
        """Events can be called in sequence."""
        behavior = FullBehavior()

        behavior.on_goal_start("test goal")
        behavior.on_round_end(1)
        behavior.on_tool_call("tool1", {}, {})
        behavior.on_tool_call("tool2", {}, {})
        behavior.on_round_end(2)
        behavior.on_goal_complete(True)

        expected = [
            "goal_start:test goal",
            "round_end:1",
            "tool_call:tool1",
            "tool_call:tool2",
            "round_end:2",
            "goal_complete:success"
        ]
        assert behavior.events_received == expected
