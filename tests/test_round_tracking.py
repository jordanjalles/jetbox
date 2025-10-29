"""Test that round tracking only applies to the deepest active subtask."""

from context_manager import Task, Subtask, ContextManager


def test_rounds_only_tracked_for_deepest_subtask():
    """
    Test that when a subtask is decomposed, its round count freezes
    and rounds are only tracked for the active child subtask.
    """
    # Create context manager
    ctx = ContextManager()

    # Create a task with a parent subtask
    task = Task(description="Test task")

    parent = Subtask(
        description="Parent subtask",
        status="in_progress",
        depth=1,
        rounds_used=5  # Parent has used 5 rounds
    )

    task.subtasks = [parent]

    # Verify parent is active and has 5 rounds
    active = task.active_subtask()
    assert active.description == "Parent subtask"
    assert active.rounds_used == 5
    print(f"✓ Parent subtask active with {active.rounds_used} rounds used")

    # Now decompose parent into children
    child1 = Subtask(
        description="Child subtask 1",
        status="in_progress",
        depth=2,
        parent_subtask="Parent subtask",
        rounds_used=0  # Child starts fresh
    )

    child2 = Subtask(
        description="Child subtask 2",
        status="pending",
        depth=2,
        parent_subtask="Parent subtask",
        rounds_used=0
    )

    # Parent is now decomposed
    parent.status = "decomposed"
    parent.child_subtasks = [child1, child2]

    # Verify that active_subtask now returns the child, not the parent
    active = task.active_subtask()
    assert active is not None, "Should find an active subtask"
    assert active.description == "Child subtask 1", f"Expected child, got '{active.description}'"
    assert active.depth == 2
    assert active.rounds_used == 0, f"Child should start with 0 rounds, got {active.rounds_used}"

    # Parent's rounds should remain frozen at 5
    assert parent.rounds_used == 5, f"Parent rounds should stay at 5, got {parent.rounds_used}"
    assert parent.status == "decomposed", "Parent should be decomposed"

    print(f"✓ After decomposition:")
    print(f"  - Parent (decomposed) frozen at {parent.rounds_used} rounds")
    print(f"  - Child 1 (active) has {active.rounds_used} rounds")

    # Simulate incrementing rounds on the active (child) subtask
    active.rounds_used += 1
    active.rounds_used += 1
    active.rounds_used += 1

    # Get active again to verify it's still the child
    active = task.active_subtask()
    assert active.description == "Child subtask 1"
    assert active.rounds_used == 3, f"Child should have 3 rounds, got {active.rounds_used}"

    # Parent should still be frozen at 5
    assert parent.rounds_used == 5, f"Parent should still be at 5 rounds, got {parent.rounds_used}"

    print(f"✓ After 3 more rounds:")
    print(f"  - Parent (decomposed) still frozen at {parent.rounds_used} rounds")
    print(f"  - Child 1 (active) now has {active.rounds_used} rounds")

    # Now decompose the child further (grandchild)
    grandchild = Subtask(
        description="Grandchild subtask",
        status="in_progress",
        depth=3,
        parent_subtask="Child subtask 1",
        rounds_used=0
    )

    child1.status = "decomposed"
    child1.child_subtasks = [grandchild]

    # Verify active is now the grandchild
    active = task.active_subtask()
    assert active.description == "Grandchild subtask"
    assert active.rounds_used == 0, "Grandchild should start fresh"

    # Parent and child should both be frozen
    assert parent.rounds_used == 5, "Parent should remain at 5"
    assert child1.rounds_used == 3, "Child should remain at 3"

    print(f"✓ After further decomposition:")
    print(f"  - Parent (decomposed) frozen at {parent.rounds_used} rounds")
    print(f"  - Child 1 (decomposed) frozen at {child1.rounds_used} rounds")
    print(f"  - Grandchild (active) has {active.rounds_used} rounds")

    print("\n✓ All tests passed! Rounds only tracked for deepest active subtask.")


def test_round_limit_applies_to_deepest_only():
    """Test that round limit checking only applies to the deepest active subtask."""
    # Create task with multi-level hierarchy
    task = Task(description="Test task")

    # Parent at depth 1 with 10 rounds (over limit of 6)
    parent = Subtask(
        description="Parent",
        status="decomposed",  # Already decomposed, so rounds don't matter
        depth=1,
        rounds_used=10
    )

    # Child at depth 2 with only 2 rounds (under limit)
    child = Subtask(
        description="Child",
        status="in_progress",
        depth=2,
        parent_subtask="Parent",
        rounds_used=2
    )

    parent.child_subtasks = [child]
    task.subtasks = [parent]

    # Active should be the child with 2 rounds
    active = task.active_subtask()
    assert active.description == "Child"
    assert active.rounds_used == 2

    # Simulation: In the agent loop, we would check if current_subtask_rounds >= MAX_ROUNDS_PER_SUBTASK
    # Since active is the child with 2 rounds, it should NOT trigger escalation
    # Parent's 10 rounds should be ignored since it's decomposed

    MAX_ROUNDS_PER_SUBTASK = 6

    # This check would happen in the main agent loop
    should_escalate = (active.rounds_used >= MAX_ROUNDS_PER_SUBTASK)
    assert not should_escalate, "Child with 2 rounds should not escalate"

    print(f"✓ Parent (decomposed) has {parent.rounds_used} rounds - IGNORED")
    print(f"✓ Child (active) has {child.rounds_used} rounds - checked against limit")
    print(f"✓ No escalation triggered (child under {MAX_ROUNDS_PER_SUBTASK} round limit)")

    print("\n✓ Round limit correctly applies only to deepest active subtask!")


if __name__ == "__main__":
    test_rounds_only_tracked_for_deepest_subtask()
    print()
    test_round_limit_applies_to_deepest_only()
