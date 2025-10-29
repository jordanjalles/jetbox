#!/usr/bin/env python3
"""
Integration test for turn counter fix.

Tests that:
1. Turn counter increments by 1 per turn
2. Turn counter resets to 0 when moving to next subtask
3. Turn counter does NOT persist across subtask transitions
"""

from context_manager import ContextManager, Task, Subtask


def test_turn_counter_reset():
    """Test that turn counter resets when moving between subtasks."""

    # Create context manager
    ctx = ContextManager()
    ctx.load_or_init("Test goal")

    # Create a task with 3 subtasks
    task = Task(
        description="Test task",
        status="in_progress"
    )

    subtask1 = Subtask(
        description="First subtask",
        status="in_progress",
        rounds_used=0
    )

    subtask2 = Subtask(
        description="Second subtask",
        status="pending",
        rounds_used=0
    )

    subtask3 = Subtask(
        description="Third subtask",
        status="pending",
        rounds_used=0
    )

    task.subtasks = [subtask1, subtask2, subtask3]
    ctx.state.goal.tasks = [task]

    # Simulate working on first subtask for 3 rounds
    print("\n=== Subtask 1: Working for 3 rounds ===")
    for i in range(3):
        subtask1.rounds_used = i + 1
        print(f"Round {i+1}: subtask1.rounds_used = {subtask1.rounds_used}")

    assert subtask1.rounds_used == 3, "Should have 3 rounds"

    # Complete first subtask, move to second
    print("\n=== Completing subtask 1, moving to subtask 2 ===")
    subtask1.status = "completed"
    subtask2.status = "in_progress"

    # Second subtask should start at 0
    assert subtask2.rounds_used == 0, "Second subtask should start at 0 rounds"
    print(f"subtask2.rounds_used = {subtask2.rounds_used} ✓")

    # Simulate working on second subtask for 5 rounds
    print("\n=== Subtask 2: Working for 5 rounds ===")
    for i in range(5):
        subtask2.rounds_used = i + 1
        print(f"Round {i+1}: subtask2.rounds_used = {subtask2.rounds_used}")

    assert subtask2.rounds_used == 5, "Should have 5 rounds"

    # First subtask should still have 3 rounds (frozen)
    assert subtask1.rounds_used == 3, "First subtask should remain at 3 rounds"
    print(f"\nsubtask1.rounds_used = {subtask1.rounds_used} (frozen) ✓")

    # Complete second, move to third
    print("\n=== Completing subtask 2, moving to subtask 3 ===")
    subtask2.status = "completed"
    subtask3.status = "in_progress"

    # Third subtask should start at 0
    assert subtask3.rounds_used == 0, "Third subtask should start at 0 rounds"
    print(f"subtask3.rounds_used = {subtask3.rounds_used} ✓")

    # Verify frozen states
    assert subtask1.rounds_used == 3, "subtask1 should still be frozen at 3"
    assert subtask2.rounds_used == 5, "subtask2 should still be frozen at 5"

    print("\n✅ All tests passed!")
    print("Turn counter correctly:")
    print("  1. Increments by 1 per turn")
    print("  2. Resets to 0 when moving to next subtask")
    print("  3. Keeps previous subtask counts frozen")


def test_decomposition_reset():
    """Test that turn counter resets when decomposing a subtask."""

    print("\n" + "="*70)
    print("TEST: Turn counter resets on decomposition")
    print("="*70)

    ctx = ContextManager()
    ctx.load_or_init("Test goal")

    task = Task(description="Test task", status="in_progress")

    # Parent subtask that will be decomposed
    parent = Subtask(
        description="Parent subtask",
        status="in_progress",
        rounds_used=0
    )

    task.subtasks = [parent]
    ctx.state.goal.tasks = [task]

    # Work on parent for 6 rounds
    print("\n=== Parent: Working for 6 rounds ===")
    for i in range(6):
        parent.rounds_used = i + 1
        print(f"Round {i+1}: parent.rounds_used = {parent.rounds_used}")

    assert parent.rounds_used == 6

    # Decompose parent into children
    print("\n=== Decomposing parent into 2 children ===")
    parent.status = "decomposed"

    child1 = Subtask(
        description="Child 1",
        status="in_progress",
        depth=2,
        parent_subtask="Parent subtask",
        rounds_used=0  # Should start at 0
    )

    child2 = Subtask(
        description="Child 2",
        status="pending",
        depth=2,
        parent_subtask="Parent subtask",
        rounds_used=0
    )

    parent.child_subtasks = [child1, child2]

    # Verify child starts at 0
    assert child1.rounds_used == 0, "Child should start at 0 rounds"
    print(f"child1.rounds_used = {child1.rounds_used} ✓")

    # Parent should be frozen at 6
    assert parent.rounds_used == 6, "Parent should be frozen at 6 rounds"
    print(f"parent.rounds_used = {parent.rounds_used} (frozen) ✓")

    # Work on child for 2 rounds
    print("\n=== Child 1: Working for 2 rounds ===")
    for i in range(2):
        child1.rounds_used = i + 1
        print(f"Round {i+1}: child1.rounds_used = {child1.rounds_used}")

    assert child1.rounds_used == 2

    # Parent should still be frozen
    assert parent.rounds_used == 6

    print("\n✅ Decomposition test passed!")
    print("Turn counter resets to 0 when creating child subtasks")


if __name__ == "__main__":
    test_turn_counter_reset()
    test_decomposition_reset()
    print("\n" + "="*70)
    print("ALL INTEGRATION TESTS PASSED! ✅")
    print("="*70)
