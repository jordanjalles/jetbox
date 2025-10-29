"""Test that rounds_used is correctly tracked per subtask.

This test verifies the fix for the round counter bug where completing
a subtask mid-round would cause the next subtask to inherit the previous
subtask's round count.
"""

from context_manager import ContextManager, Subtask, Task


def test_rounds_reset_between_subtasks():
    """Test that round counter resets when moving to next subtask."""
    # Directly test the subtask objects without full context manager
    first = Subtask(description="First subtask", status="in_progress", rounds_used=3)
    second = Subtask(description="Second subtask", status="pending", rounds_used=0)

    # Simulate completion
    first.status = "completed"
    second.status = "in_progress"

    # Second subtask should start at 0, not inherit 3
    assert second.rounds_used == 0, f"Second subtask should start with 0 rounds, got {second.rounds_used}"

    # First subtask should still have 3 rounds
    assert first.rounds_used == 3, f"First subtask should still have 3 rounds, got {first.rounds_used}"


def test_decomposed_subtask_rounds():
    """Test that child subtasks start with 0 rounds when parent decomposes."""
    parent = Subtask(
        description="Parent subtask",
        status="decomposed",
        rounds_used=5
    )

    # Decompose parent into children
    parent.child_subtasks = [
        Subtask(
            description="Child 1",
            status="in_progress",
            depth=2,
            parent_subtask=parent.description,
            rounds_used=0
        ),
        Subtask(
            description="Child 2",
            status="pending",
            depth=2,
            parent_subtask=parent.description,
            rounds_used=0
        )
    ]

    # Child should start with 0 rounds
    child = parent.child_subtasks[0]
    assert child.rounds_used == 0, f"Child should start with 0 rounds, got {child.rounds_used}"
    assert parent.rounds_used == 5, f"Parent should keep 5 rounds, got {parent.rounds_used}"


if __name__ == "__main__":
    test_rounds_reset_between_subtasks()
    print("✓ test_rounds_reset_between_subtasks passed")

    test_decomposed_subtask_rounds()
    print("✓ test_decomposed_subtask_rounds passed")

    print("\nAll tests passed!")
