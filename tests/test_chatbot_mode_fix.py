#!/usr/bin/env python3
"""
Test script to verify the chatbot vs goal mode fix.

Tests:
1. Pure chatbot (no goal) - should NOT have mark_complete/mark_failed
2. Goal-oriented agent - should HAVE mark_complete/mark_failed
"""
from pathlib import Path
from task_executor_agent import TaskExecutorAgent


def test_pure_chatbot_no_completion_tools():
    """Test that pure chatbot (no goal) doesn't have completion tools."""
    print("=" * 60)
    print("TEST 1: Pure Chatbot (No Goal)")
    print("=" * 60)

    workspace = Path('.agent_workspace/chatbot_test_pure')
    workspace.mkdir(parents=True, exist_ok=True)

    # Create simple-chatbot WITHOUT a goal
    agent = TaskExecutorAgent(
        workspace=workspace,
        config_file='config/agents/simple-chatbot.yaml',
        goal=None,  # No goal = pure conversational mode
        timeout_seconds=60
    )

    print(f"✓ Agent created: {agent.name}")
    print(f"  Goal: {agent.goal}")
    print(f"  Behaviors: {len(agent.behaviors)}")

    # Get tools
    tools = agent.get_tools()
    tool_names = [t.get("function", {}).get("name") for t in tools]

    print(f"  Tools ({len(tools)} total):")
    for name in tool_names:
        print(f"    - {name}")

    # Check for completion tools
    has_mark_complete = "mark_complete" in tool_names
    has_mark_failed = "mark_failed" in tool_names

    print()
    print("ASSERTIONS:")
    print(f"  mark_complete present: {has_mark_complete} (expected: False)")
    print(f"  mark_failed present: {has_mark_failed} (expected: False)")

    if has_mark_complete or has_mark_failed:
        print("❌ FAILED: Completion tools should NOT be present for pure chatbot")
        return False
    else:
        print("✅ PASSED: No completion tools (correct for pure chatbot)")
        return True


def test_goal_oriented_agent_has_completion_tools():
    """Test that goal-oriented agent HAS completion tools."""
    print("\n" + "=" * 60)
    print("TEST 2: Goal-Oriented Agent")
    print("=" * 60)

    workspace = Path('.agent_workspace/chatbot_test_goal')
    workspace.mkdir(parents=True, exist_ok=True)

    # Create agent WITH a goal
    agent = TaskExecutorAgent(
        workspace=workspace,
        config_file='config/agents/task_executor.yaml',
        goal="Test goal for validation",  # Goal set = execution mode
        timeout_seconds=60
    )

    print(f"✓ Agent created: {agent.name}")
    print(f"  Goal: {agent.goal}")
    print(f"  Behaviors: {len(agent.behaviors)}")

    # Get tools
    tools = agent.get_tools()
    tool_names = [t.get("function", {}).get("name") for t in tools]

    print(f"  Tools ({len(tools)} total):")
    for name in tool_names:
        print(f"    - {name}")

    # Check for completion tools
    has_mark_complete = "mark_complete" in tool_names
    has_mark_failed = "mark_failed" in tool_names

    print()
    print("ASSERTIONS:")
    print(f"  mark_complete present: {has_mark_complete} (expected: True)")
    print(f"  mark_failed present: {has_mark_failed} (expected: True)")

    if not has_mark_complete or not has_mark_failed:
        print("❌ FAILED: Completion tools SHOULD be present for goal-oriented agent")
        return False
    else:
        print("✅ PASSED: Completion tools present (correct for goal-oriented agent)")
        return True


def test_chatbot_transition_gains_completion_tools():
    """Test that setting a goal dynamically adds completion tools."""
    print("\n" + "=" * 60)
    print("TEST 3: Chat Mode → Goal Mode Transition")
    print("=" * 60)

    workspace = Path('.agent_workspace/chatbot_test_transition')
    workspace.mkdir(parents=True, exist_ok=True)

    # Start without goal
    agent = TaskExecutorAgent(
        workspace=workspace,
        config_file='config/agents/task_executor.yaml',
        goal=None,  # Start in chat mode
        timeout_seconds=60
    )

    print(f"✓ Agent created: {agent.name}")
    print(f"  Initial goal: {agent.goal}")

    # Check tools before goal set
    tools_before = agent.get_tools()
    tool_names_before = [t.get("function", {}).get("name") for t in tools_before]
    has_completion_before = "mark_complete" in tool_names_before

    print(f"  Tools before goal ({len(tools_before)} total): {', '.join(tool_names_before[:5])}...")
    print(f"  Has completion tools: {has_completion_before} (expected: False)")

    # Set goal
    agent.goal = "Test goal for transition"
    print(f"\n  Goal set to: {agent.goal}")

    # Check tools after goal set
    tools_after = agent.get_tools()
    tool_names_after = [t.get("function", {}).get("name") for t in tools_after]
    has_completion_after = "mark_complete" in tool_names_after

    print(f"  Tools after goal ({len(tools_after)} total): {', '.join(tool_names_after[:5])}...")
    print(f"  Has completion tools: {has_completion_after} (expected: True)")

    print()
    print("ASSERTIONS:")
    print(f"  Before: No completion tools = {not has_completion_before}")
    print(f"  After: Has completion tools = {has_completion_after}")

    if has_completion_before:
        print("❌ FAILED: Should not have completion tools before goal set")
        return False
    elif not has_completion_after:
        print("❌ FAILED: Should have completion tools after goal set")
        return False
    else:
        print("✅ PASSED: Tools correctly added when goal set")
        return True


def main():
    """Run all tests."""
    print("\n🔧 Testing Chatbot vs Goal Mode Fix\n")

    results = []

    # Test 1: Pure chatbot
    try:
        results.append(("Pure Chatbot", test_pure_chatbot_no_completion_tools()))
    except Exception as e:
        print(f"❌ Test 1 ERROR: {e}")
        results.append(("Pure Chatbot", False))

    # Test 2: Goal-oriented agent
    try:
        results.append(("Goal-Oriented Agent", test_goal_oriented_agent_has_completion_tools()))
    except Exception as e:
        print(f"❌ Test 2 ERROR: {e}")
        results.append(("Goal-Oriented Agent", False))

    # Test 3: Transition
    try:
        results.append(("Chat→Goal Transition", test_chatbot_transition_gains_completion_tools()))
    except Exception as e:
        print(f"❌ Test 3 ERROR: {e}")
        results.append(("Chat→Goal Transition", False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
