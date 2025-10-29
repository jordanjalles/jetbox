"""Test that subtask round limits apply only to the deepest active subtask."""

from context_manager import Task, Subtask


def test_active_subtask_finds_deepest():
    """Test that active_subtask() returns the deepest in-progress subtask."""
    # Create a task with a hierarchical subtask structure
    task = Task(description="Test task")

    # Parent subtask at depth 1
    parent = Subtask(
        description="Parent subtask",
        status="decomposed",  # Decomposed means it has children
        depth=1
    )

    # Child subtask at depth 2 (in progress)
    child = Subtask(
        description="Child subtask",
        status="in_progress",
        depth=2,
        parent_subtask="Parent subtask"
    )

    # Grandchild subtask at depth 3 (pending)
    grandchild = Subtask(
        description="Grandchild subtask",
        status="pending",
        depth=3,
        parent_subtask="Child subtask"
    )

    # Build hierarchy
    child.child_subtasks = [grandchild]
    parent.child_subtasks = [child]
    task.subtasks = [parent]

    # Test: active_subtask should return the child (depth 2), not parent (depth 1)
    active = task.active_subtask()
    assert active is not None, "Should find an active subtask"
    assert active.description == "Child subtask", f"Expected 'Child subtask', got '{active.description}'"
    assert active.depth == 2, f"Expected depth 2, got {active.depth}"

    print("✓ Test 1 passed: active_subtask() correctly finds child at depth 2")

    # Test 2: When child is also decomposed, find grandchild
    child.status = "decomposed"
    grandchild.status = "in_progress"

    active = task.active_subtask()
    assert active is not None, "Should find an active subtask"
    assert active.description == "Grandchild subtask", f"Expected 'Grandchild subtask', got '{active.description}'"
    assert active.depth == 3, f"Expected depth 3, got {active.depth}"

    print("✓ Test 2 passed: active_subtask() correctly finds grandchild at depth 3")

    # Test 3: When no child is active, return top-level in_progress
    parent2 = Subtask(
        description="Another parent",
        status="in_progress",
        depth=1
    )
    task.subtasks.append(parent2)

    active = task.active_subtask()
    # Should still return grandchild from first hierarchy since it's still in_progress
    assert active.description == "Grandchild subtask"

    print("✓ Test 3 passed: active_subtask() prioritizes deeper subtasks correctly")

    # Test 4: Verify rounds_used is independent per subtask
    parent.rounds_used = 5
    child.rounds_used = 2
    grandchild.rounds_used = 1

    active = task.active_subtask()
    assert active.rounds_used == 1, f"Expected rounds_used=1 for grandchild, got {active.rounds_used}"
    assert parent.rounds_used == 5, "Parent rounds_used should remain unchanged"
    assert child.rounds_used == 2, "Child rounds_used should remain unchanged"

    print("✓ Test 4 passed: rounds_used is tracked independently per subtask level")

    print("\n✓ All tests passed! Subtask hierarchy and round tracking work correctly.")


if __name__ == "__main__":
    test_active_subtask_finds_deepest()
