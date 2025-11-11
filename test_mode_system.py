"""
Test the event-based mode system implementation.

Tests:
1. Chat mode starts active by default
2. set_goal() activates execution mode
3. Execution mode enforces tool usage (empty round detection)
4. mark_complete() transitions back to chat mode
5. Event-based conflict resolution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from behaviors.execution_mode import ExecutionModeBehavior
from behaviors.chatbot import ChatbotBehavior
from unittest.mock import Mock, MagicMock


def test_initial_state():
    """Test that behaviors initialize with correct default state."""
    print("\n=== Test 1: Initial State ===")

    # ExecutionModeBehavior should start inactive
    exec_mode = ExecutionModeBehavior()
    assert exec_mode.is_active == False, "ExecutionMode should start inactive"
    print("✓ ExecutionModeBehavior starts inactive")

    # ChatbotBehavior should start active
    chatbot = ChatbotBehavior()
    assert chatbot.is_active == True, "Chatbot should start active"
    print("✓ ChatbotBehavior starts active")


def test_chat_to_execution_transition():
    """Test transition from chat mode to execution mode."""
    print("\n=== Test 2: Chat → Execution Transition ===")

    # Setup
    exec_mode = ExecutionModeBehavior()
    chatbot = ChatbotBehavior()

    # Mock agent
    agent = Mock()
    agent.behaviors = [exec_mode, chatbot]
    agent.event_system = Mock()
    agent.event_system.fire_custom_event = Mock()
    agent.add_message = Mock()
    agent.goal = None

    # Attach behaviors to agent
    exec_mode.agent = agent
    chatbot.agent = agent

    # Initial state: chat active, execution inactive
    assert chatbot.is_active == True
    assert exec_mode.is_active == False
    print("✓ Starting state: chat active, execution inactive")

    # Activate execution mode
    result = exec_mode.activate(agent)
    assert result["success"] == True
    assert exec_mode.is_active == True
    print("✓ ExecutionMode activated successfully")

    # Verify activation message was added
    assert agent.add_message.called
    activation_msg = agent.add_message.call_args[0][0]
    assert "EXECUTION MODE ACTIVATED" in activation_msg["content"]
    print("✓ Activation message appended to history")

    # Verify event was fired
    assert agent.event_system.fire_custom_event.called
    event_call = agent.event_system.fire_custom_event.call_args
    assert event_call[1]["mode_name"] == "execution"
    print("✓ 'mode_activated' event fired with mode_name='execution'")

    # Simulate event delivery to chatbot
    chatbot.on_custom_event(agent, "mode_activated", mode_name="execution")

    # Verify chatbot auto-deactivated
    assert chatbot.is_active == False
    print("✓ ChatbotBehavior auto-deactivated on conflict")

    # Verify deactivation message was added
    deactivation_call = [c for c in agent.add_message.call_args_list if "CHAT MODE" in str(c)]
    assert len(deactivation_call) > 0
    print("✓ Deactivation message appended to history")


def test_execution_to_chat_transition():
    """Test transition from execution mode to chat mode."""
    print("\n=== Test 3: Execution → Chat Transition ===")

    # Setup
    exec_mode = ExecutionModeBehavior()
    chatbot = ChatbotBehavior()

    # Mock agent
    agent = Mock()
    agent.behaviors = [exec_mode, chatbot]
    agent.add_message = Mock()
    agent.event_system = Mock()
    agent.event_system.fire_custom_event = Mock()

    # Start in execution mode
    exec_mode.is_active = True
    chatbot.is_active = False

    # Deactivate execution mode (simulate mark_complete)
    result = exec_mode.deactivate(agent, reason="complete")
    assert result["success"] == True
    assert exec_mode.is_active == False
    print("✓ ExecutionMode deactivated successfully")

    # Verify completion message
    assert agent.add_message.called
    completion_msg = agent.add_message.call_args[0][0]
    assert "EXECUTION MODE COMPLETE" in completion_msg["content"]
    print("✓ Completion message appended to history")

    # Activate chat mode
    agent.add_message.reset_mock()
    result = chatbot.activate(agent)
    assert result["success"] == True
    assert chatbot.is_active == True
    print("✓ ChatbotBehavior activated successfully")

    # Verify activation message
    assert agent.add_message.called
    chat_msg = agent.add_message.call_args[0][0]
    assert "CHAT MODE ACTIVATED" in chat_msg["content"]
    print("✓ Chat activation message appended to history")


def test_empty_round_detection():
    """Test that execution mode detects empty rounds."""
    print("\n=== Test 4: Empty Round Detection ===")

    exec_mode = ExecutionModeBehavior(max_empty_rounds=3)

    # Mock agent with state
    agent = Mock()
    agent.behaviors = [exec_mode]
    agent.state = Mock()
    agent.state.messages = []

    # Activate execution mode
    exec_mode.is_active = True

    # Simulate 3 rounds with no tools called
    for i in range(3):
        # Don't call on_tool_call (simulates no tools called)
        exec_mode.on_round_end(agent, round_number=i+1)

    # Should have a pending nudge after 3 empty rounds
    assert exec_mode.pending_nudge is not None
    assert "Empty round detected" in exec_mode.pending_nudge
    assert "MUST call tools" in exec_mode.pending_nudge
    print("✓ Empty round violation detected after 3 rounds")
    print(f"✓ Warning message: {exec_mode.pending_nudge[:80]}...")

    # Simulate a round with tools called - should reset
    exec_mode.on_tool_call(agent, "write_file", {"path": "test.py", "content": "..."}, {"success": True})
    exec_mode.on_tool_call(agent, "run_bash", {"command": "pytest"}, {"success": True})
    exec_mode.on_round_end(agent, round_number=4)
    assert exec_mode.consecutive_empty_rounds == 0
    print("✓ Empty round counter resets when tools are called")


def test_chat_mode_no_empty_round_detection():
    """Test that chat mode does NOT detect empty rounds."""
    print("\n=== Test 5: Chat Mode - No Empty Round Detection ===")

    exec_mode = ExecutionModeBehavior(max_empty_rounds=3)

    # Mock agent with state
    agent = Mock()
    agent.behaviors = [exec_mode]
    agent.state = Mock()
    agent.state.messages = []

    # Keep execution mode inactive (chat mode)
    exec_mode.is_active = False

    # Simulate 10 rounds with no tools called
    for i in range(10):
        # Don't call on_tool_call (simulates no tools called)
        exec_mode.on_round_end(agent, round_number=i+1)

    # Should NOT have a pending nudge (execution mode not active)
    assert exec_mode.pending_nudge is None
    assert exec_mode.consecutive_empty_rounds == 0  # Not tracking
    print("✓ No empty round detection in chat mode (10 rounds with no tools)")


def test_completion_nudging():
    """Test that completion signals trigger nudges."""
    print("\n=== Test 6: Completion Signal Detection ===")

    exec_mode = ExecutionModeBehavior(
        enable_completion_nudging=True,
        min_rounds_before_nudge=3
    )

    # Mock agent with state and messages
    agent = Mock()
    agent.behaviors = [exec_mode]
    agent.state = Mock()
    agent.state.messages = [
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": "The task is now completed successfully. All tests pass."}
    ]

    # Activate execution mode
    exec_mode.is_active = True

    # Simulate tools being called
    exec_mode.on_tool_call(agent, "write_file", {}, {"success": True})
    exec_mode.on_tool_call(agent, "run_bash", {}, {"success": True})

    # Simulate round end with completion signal in last message
    exec_mode.on_round_end(agent, round_number=4)  # After min_rounds_before_nudge

    # Should have a completion nudge
    assert exec_mode.pending_nudge is not None
    assert "completed the task" in exec_mode.pending_nudge
    assert "mark_complete" in exec_mode.pending_nudge
    print("✓ Completion signal detected")
    print(f"✓ Nudge message: {exec_mode.pending_nudge[:80]}...")


def test_set_goal_tool():
    """Test that set_goal tool activates execution mode."""
    print("\n=== Test 7: set_goal() Tool ===")

    exec_mode = ExecutionModeBehavior()
    chatbot = ChatbotBehavior()

    # Mock agent with both behaviors
    agent = Mock()
    agent.behaviors = [exec_mode, chatbot]
    agent.add_message = Mock()
    agent.event_system = Mock()
    agent.event_system.fire_custom_event = Mock()
    agent.set_goal = Mock()
    agent.goal = None

    # Attach behaviors
    exec_mode.agent = agent
    chatbot.agent = agent

    # Initial state
    assert chatbot.is_active == True
    assert exec_mode.is_active == False

    # Call set_goal
    result = chatbot.dispatch_tool(
        agent,
        "set_goal",
        {"goal": "Build a calculator", "requirements": "Add and multiply"}
    )

    # Verify agent.set_goal was called
    assert agent.set_goal.called
    print("✓ agent.set_goal() was called")

    # Verify execution mode was activated
    assert exec_mode.is_active == True
    print("✓ ExecutionMode activated by set_goal")

    # Verify success response
    assert result["success"] == True
    assert result["mode_transition"] == "chat → execution"
    print("✓ set_goal returned success with mode transition")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Event-Based Mode System Implementation")
    print("=" * 60)

    try:
        test_initial_state()
        test_chat_to_execution_transition()
        test_execution_to_chat_transition()
        test_empty_round_detection()
        test_chat_mode_no_empty_round_detection()
        test_completion_nudging()
        test_set_goal_tool()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nMode system is working correctly:")
        print("  ✓ Chat mode starts active by default")
        print("  ✓ set_goal() activates execution mode")
        print("  ✓ Event-based conflict resolution works")
        print("  ✓ Empty round detection in execution mode")
        print("  ✓ No empty round detection in chat mode")
        print("  ✓ Completion signal detection works")
        print("  ✓ Mode transitions append messages")
        print("\n")
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
