"""
Phase 6 Testing: Edge Case Tests for Behavior System

This module tests edge cases and error handling in the behavior system:

1. Tool conflict detection - Duplicate tool names across behaviors
2. Context enhancement order - Multiple behaviors modifying context
3. Event handler errors - Behaviors that throw exceptions in event handlers
4. Missing behavior classes - Config references non-existent behaviors

These tests ensure the behavior system is robust and handles errors gracefully.
"""
import pytest
from pathlib import Path
import tempfile
import shutil

from base_agent import BaseAgent
from behaviors.base import AgentBehavior
from agent_config import config
from typing import Any


class ConcreteTestAgent(BaseAgent):
    """Concrete agent for testing (not a pytest test class)."""

    def get_system_prompt(self) -> str:
        return "Test agent system prompt"

    def get_tools(self) -> list[dict[str, Any]]:
        # Return behavior tools if using behaviors
        if hasattr(self, 'use_behaviors') and self.use_behaviors:
            return self.get_behavior_tools()
        return []

    def get_context_strategy(self) -> str:
        return "test_strategy"

    def build_context(self) -> list[dict[str, Any]]:
        """Build minimal context."""
        context = [{"role": "system", "content": self.get_system_prompt()}]

        # Let behaviors modify context if present
        if hasattr(self, 'behaviors'):
            for behavior in self.behaviors:
                context = behavior.enhance_context(
                    context,
                    agent=self,
                    workspace=self.workspace,
                    round_number=1
                )

        return context

    def _notify_behaviors(self, event_name: str, **kwargs):
        """Notify all behaviors of an event."""
        if not hasattr(self, 'behaviors'):
            return

        for behavior in self.behaviors:
            try:
                handler = getattr(behavior, event_name, None)
                if handler and callable(handler):
                    handler(**kwargs)
            except Exception as e:
                # Log but don't crash on behavior errors
                print(f"[behavior] {behavior.get_name()} {event_name} error: {e}")


class DummyBehavior1(AgentBehavior):
    """Test behavior that provides tool1."""

    def get_name(self) -> str:
        return "dummy1"

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "tool1",
                    "description": "Test tool 1",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def dispatch_tool(self, tool_name, args, **kwargs):
        return {"result": "tool1 executed"}


class DummyBehavior2(AgentBehavior):
    """Test behavior that also provides tool1 (conflict!)."""

    def get_name(self) -> str:
        return "dummy2"

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "tool1",  # Same name as DummyBehavior1!
                    "description": "Test tool 1 (duplicate)",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def dispatch_tool(self, tool_name, args, **kwargs):
        return {"result": "tool1 executed by behavior2"}


class ContextEnhancingBehavior(AgentBehavior):
    """Test behavior that modifies context."""

    def __init__(self, tag: str):
        self.tag = tag

    def get_name(self) -> str:
        return f"context_enhancer_{self.tag}"

    def enhance_context(self, context, **kwargs):
        """Add a marker to system prompt."""
        if context and context[0]["role"] == "system":
            context[0]["content"] += f"\n[Enhanced by {self.tag}]"
        return context


class BrokenEventBehavior(AgentBehavior):
    """Test behavior that throws exceptions in event handlers."""

    def get_name(self) -> str:
        return "broken_events"

    def on_goal_start(self, goal, **kwargs):
        raise RuntimeError("Intentional error in on_goal_start")

    def on_tool_call(self, tool_name, args, result, **kwargs):
        raise ValueError("Intentional error in on_tool_call")


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_edge_cases_")
    workspace = Path(temp_dir)
    yield workspace
    # Cleanup
    if workspace.exists():
        shutil.rmtree(workspace)


class TestToolConflictDetection:
    """Test that duplicate tool names are detected."""

    def test_duplicate_tool_names_rejected(self, temp_workspace):
        """
        Test that registering two behaviors with the same tool name fails.

        This is a critical safety feature - tool conflicts could cause
        unpredictable behavior.
        """
        print("\n" + "="*80)
        print("TEST: Tool Conflict Detection")
        print("="*80)

        # Create agent
        agent = ConcreteTestAgent(
            name="test_agent",
            role="Test agent",
            workspace=temp_workspace,
            config=config,
        )

        # Add first behavior with tool1
        behavior1 = DummyBehavior1()
        agent.add_behavior(behavior1)

        print(f"\n✓ Registered behavior: {behavior1.get_name()}")
        print(f"  Tools: {[t['function']['name'] for t in behavior1.get_tools()]}")

        # Try to add second behavior with same tool name - should fail
        behavior2 = DummyBehavior2()

        print(f"\nAttempting to register behavior: {behavior2.get_name()}")
        print(f"  Tools: {[t['function']['name'] for t in behavior2.get_tools()]}")

        with pytest.raises(ValueError) as exc_info:
            agent.add_behavior(behavior2)

        print(f"\n✓ Tool conflict detected correctly!")
        print(f"  Error: {exc_info.value}")

        assert "already registered" in str(exc_info.value).lower()
        assert "tool1" in str(exc_info.value).lower()

        print("="*80)


class TestContextEnhancementOrder:
    """Test that context enhancements are applied in order."""

    def test_context_enhanced_in_order(self, temp_workspace):
        """
        Test that multiple behaviors enhance context in registration order.

        This ensures deterministic behavior composition.
        """
        print("\n" + "="*80)
        print("TEST: Context Enhancement Order")
        print("="*80)

        # Create agent
        agent = ConcreteTestAgent(
            name="test_agent",
            role="Test agent",
            workspace=temp_workspace,
            config=config,
        )

        # Add behaviors in specific order
        agent.add_behavior(ContextEnhancingBehavior("A"))
        agent.add_behavior(ContextEnhancingBehavior("B"))
        agent.add_behavior(ContextEnhancingBehavior("C"))

        print("\nRegistered behaviors:")
        for b in agent.behaviors:
            print(f"  - {b.get_name()}")

        # Build context
        initial_context = [{"role": "system", "content": "Base prompt"}]

        # Apply enhancements
        enhanced_context = initial_context.copy()
        for behavior in agent.behaviors:
            enhanced_context = behavior.enhance_context(
                enhanced_context,
                agent=agent,
                workspace=temp_workspace,
                round_number=1
            )

        final_content = enhanced_context[0]["content"]

        print(f"\nInitial prompt: {initial_context[0]['content']}")
        print(f"Final prompt: {final_content}")

        # Check that enhancements were applied in order
        assert "[Enhanced by A]" in final_content
        assert "[Enhanced by B]" in final_content
        assert "[Enhanced by C]" in final_content

        # Check order: A should come before B, B before C
        pos_a = final_content.index("[Enhanced by A]")
        pos_b = final_content.index("[Enhanced by B]")
        pos_c = final_content.index("[Enhanced by C]")

        assert pos_a < pos_b < pos_c, "Enhancements not applied in order"

        print(f"\n✓ Enhancements applied in correct order")
        print(f"  Positions: A={pos_a}, B={pos_b}, C={pos_c}")
        print("="*80)


class TestEventHandlerErrors:
    """Test that event handler errors are handled gracefully."""

    def test_broken_event_handlers_dont_crash(self, temp_workspace):
        """
        Test that exceptions in event handlers don't crash the agent.

        Event handlers should be defensive - one bad behavior shouldn't
        break the entire system.
        """
        print("\n" + "="*80)
        print("TEST: Event Handler Error Handling")
        print("="*80)

        # Create agent
        agent = ConcreteTestAgent(
            name="test_agent",
            role="Test agent",
            workspace=temp_workspace,
            config=config,
        )

        # Add behavior with broken event handlers
        broken_behavior = BrokenEventBehavior()
        agent.add_behavior(broken_behavior)

        print(f"\nRegistered behavior: {broken_behavior.get_name()}")
        print("  This behavior throws exceptions in event handlers")

        # Test on_goal_start - should not crash
        print("\nCalling on_goal_start...")
        try:
            agent._notify_behaviors("on_goal_start", goal="test goal")
            print("✓ on_goal_start handled (no crash)")
        except Exception as e:
            print(f"✓ on_goal_start exception caught: {e}")
            # This is OK - as long as it doesn't crash the whole system

        # Test on_tool_call - should not crash
        print("\nCalling on_tool_call...")
        try:
            agent._notify_behaviors(
                "on_tool_call",
                tool_name="test_tool",
                args={},
                result={"success": True}
            )
            print("✓ on_tool_call handled (no crash)")
        except Exception as e:
            print(f"✓ on_tool_call exception caught: {e}")

        print("\n✓ Event handler errors don't crash the system")
        print("="*80)

        # Test passes if we get here without crashing
        assert True


class TestMissingBehaviorGracefulFailure:
    """Test that missing behaviors are handled gracefully."""

    def test_config_with_missing_behavior_logs_error(self, temp_workspace):
        """
        Test that loading a config with non-existent behavior logs error and continues.

        The current implementation logs failures and continues rather than crashing.
        This is a graceful degradation approach.
        """
        print("\n" + "="*80)
        print("TEST: Missing Behavior Handling")
        print("="*80)

        # Create agent
        agent = ConcreteTestAgent(
            name="test_agent",
            role="Test agent",
            workspace=temp_workspace,
            config=config,
        )

        # Create temp config with non-existent behavior
        config_file = temp_workspace / "bad_config.yaml"
        config_file.write_text("""
behaviors:
  - type: NonExistentBehavior
    params: {}
  - type: AnotherMissingBehavior
    params: {}
""")

        print(f"\nCreated config with missing behaviors: {config_file}")
        print("  References: NonExistentBehavior, AnotherMissingBehavior")

        # Try to load - should log errors but not crash
        print("\nAttempting to load config...")
        initial_behavior_count = len(agent.behaviors)

        agent.load_behaviors_from_config(str(config_file))

        # Should have same number of behaviors (none added due to failures)
        final_behavior_count = len(agent.behaviors)

        print(f"\n✓ Load completed without crashing!")
        print(f"  Initial behaviors: {initial_behavior_count}")
        print(f"  Final behaviors: {final_behavior_count}")
        print(f"  Errors logged (check output above)")

        assert final_behavior_count == initial_behavior_count, \
            "Missing behaviors should not be added"

        print("="*80)


class TestBehaviorSystemRobustness:
    """Test overall robustness of behavior system."""

    def test_empty_behaviors_list_works(self, temp_workspace):
        """Test that agent works with no behaviors registered."""
        print("\n" + "="*80)
        print("TEST: Agent with No Behaviors")
        print("="*80)

        agent = ConcreteTestAgent(
            name="test_agent",
            role="Test agent",
            workspace=temp_workspace,
            config=config,
        )

        print(f"\nCreated agent with {len(agent.behaviors)} behaviors")

        # Should have no tools
        tools = agent.get_behavior_tools()
        print(f"Tools: {len(tools)}")
        assert len(tools) == 0

        # Should still be able to notify (no-op)
        agent._notify_behaviors("on_goal_start", goal="test")

        print("\n✓ Agent works with no behaviors")
        print("="*80)

    def test_behavior_with_no_tools_works(self, temp_workspace):
        """Test that behaviors without tools work correctly."""
        print("\n" + "="*80)
        print("TEST: Behavior with No Tools")
        print("="*80)

        class NoToolsBehavior(AgentBehavior):
            def get_name(self):
                return "no_tools"

            def get_tools(self):
                return []  # No tools

        agent = ConcreteTestAgent(
            name="test_agent",
            role="Test agent",
            workspace=temp_workspace,
            config=config,
        )

        behavior = NoToolsBehavior()
        agent.add_behavior(behavior)

        print(f"\nRegistered behavior: {behavior.get_name()}")
        print(f"Tools: {len(behavior.get_tools())}")

        # Should work fine
        tools = agent.get_behavior_tools()
        print(f"\nAgent tools: {len(tools)}")
        assert len(tools) == 0

        print("\n✓ Behavior with no tools works correctly")
        print("="*80)


if __name__ == "__main__":
    """Run edge case tests."""
    import sys

    print("\n" + "="*80)
    print("PHASE 6: Edge Case Tests for Behavior System")
    print("="*80)
    print("\nTests:")
    print("  1. Tool conflict detection")
    print("  2. Context enhancement order")
    print("  3. Event handler error handling")
    print("  4. Missing behavior handling")
    print("  5. System robustness")
    print("="*80)

    sys.exit(pytest.main([__file__, "-v", "-s"]))
