"""Test completion nudging integration in SubAgentModeBehavior."""
from behaviors.subagent_mode import SubAgentModeBehavior


def test_completion_nudging_detects_signals():
    """Test that completion nudging detects completion signals."""
    behavior = SubAgentModeBehavior(
        is_subagent=True,
        enable_completion_nudging=True,
        min_rounds_before_nudge=1  # Allow nudging on round 1 for testing
    )

    # Set up goal
    behavior.goal = "Create a test file"

    # Simulate round 1 with completion signal
    llm_response = "I have successfully created all the required files and all tests passed."
    tool_calls = [
        {
            "function": {"name": "write_file"},
            "arguments": {"path": "test.py", "content": "# test"}
        }
    ]

    # Call on_round_end
    behavior.on_round_end(
        round_number=1,
        llm_response=llm_response,
        tool_calls=tool_calls
    )

    # Should have set a pending nudge
    assert behavior.pending_nudge is not None
    assert "mark_complete" in behavior.pending_nudge
    assert "REMINDER" in behavior.pending_nudge

    print("✓ Completion signal detected and nudge created")


def test_completion_nudging_respects_min_rounds():
    """Test that completion nudging respects min_rounds_before_nudge."""
    behavior = SubAgentModeBehavior(
        is_subagent=True,
        enable_completion_nudging=True,
        min_rounds_before_nudge=3
    )

    behavior.goal = "Create a test file"

    # Simulate round 1 with completion signal
    llm_response = "All tests passed successfully."
    tool_calls = []

    behavior.on_round_end(
        round_number=1,
        llm_response=llm_response,
        tool_calls=tool_calls
    )

    # Should NOT have set a nudge (round < min_rounds)
    assert behavior.pending_nudge is None

    # Simulate round 3 with completion signal
    behavior.on_round_end(
        round_number=3,
        llm_response=llm_response,
        tool_calls=tool_calls
    )

    # Should have set a nudge (round >= min_rounds)
    assert behavior.pending_nudge is not None

    print("✓ min_rounds_before_nudge respected")


def test_completion_nudging_skips_if_mark_complete_called():
    """Test that nudging is skipped if agent already called mark_complete."""
    behavior = SubAgentModeBehavior(
        is_subagent=True,
        enable_completion_nudging=True,
        min_rounds_before_nudge=1
    )

    behavior.goal = "Create a test file"

    # Simulate round with completion signal AND mark_complete call
    llm_response = "Task is complete!"
    tool_calls = [
        {
            "function": {"name": "mark_complete"},
            "arguments": {"summary": "Created test file"}
        }
    ]

    behavior.on_round_end(
        round_number=1,
        llm_response=llm_response,
        tool_calls=tool_calls
    )

    # Should NOT set a nudge (agent already called mark_complete)
    assert behavior.pending_nudge is None

    print("✓ Nudging skipped when mark_complete already called")


def test_completion_nudging_can_be_disabled():
    """Test that completion nudging can be disabled."""
    behavior = SubAgentModeBehavior(
        is_subagent=True,
        enable_completion_nudging=False,  # Disabled
        min_rounds_before_nudge=1
    )

    behavior.goal = "Create a test file"

    # Simulate round with completion signal
    llm_response = "All tests passed successfully."
    tool_calls = []

    behavior.on_round_end(
        round_number=1,
        llm_response=llm_response,
        tool_calls=tool_calls
    )

    # Should NOT set a nudge (nudging disabled)
    assert behavior.pending_nudge is None

    print("✓ Completion nudging can be disabled")


def test_pending_nudge_injected_into_context():
    """Test that pending nudges are injected into context."""
    behavior = SubAgentModeBehavior(
        is_subagent=True,
        enable_completion_nudging=True,
        min_rounds_before_nudge=1
    )

    behavior.goal = "Create a test file"

    # Set a pending nudge manually
    behavior.pending_nudge = "💡 REMINDER: Call mark_complete"

    # Create context
    context = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Do the task."}
    ]

    # Enhance context
    enhanced = behavior.enhance_context(context)

    # Should have injected goal header AND nudge
    # Format: system + goal_header + original_user + nudge
    assert len(enhanced) == 4
    assert enhanced[1]["role"] == "user"  # Goal header
    assert "GOAL" in enhanced[1]["content"] or "DELEGATED GOAL" in enhanced[1]["content"]
    assert enhanced[-1]["role"] == "user"  # Nudge
    assert "REMINDER" in enhanced[-1]["content"]

    # Should have cleared pending nudge
    assert behavior.pending_nudge is None

    print("✓ Pending nudge injected into context and cleared")


if __name__ == "__main__":
    test_completion_nudging_detects_signals()
    test_completion_nudging_respects_min_rounds()
    test_completion_nudging_skips_if_mark_complete_called()
    test_completion_nudging_can_be_disabled()
    test_pending_nudge_injected_into_context()
    print("\n✅ All completion nudging tests passed!")
