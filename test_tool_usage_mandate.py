"""Quick test to verify ToolUsageMandateBehavior injects instruction correctly."""

from behaviors.tool_usage_mandate import ToolUsageMandateBehavior


def test_tool_usage_mandate_injection():
    """Test that the behavior injects tool usage instruction."""
    behavior = ToolUsageMandateBehavior()

    # Mock initial context (system prompt only)
    context = [
        {"role": "system", "content": "You are a coding agent."}
    ]

    # Mock agent
    class MockAgent:
        pass

    agent = MockAgent()

    # Inject instruction
    enhanced = behavior.on_initial_context(agent, context)

    # Verify injection
    assert len(enhanced) == 2, f"Expected 2 messages, got {len(enhanced)}"
    assert enhanced[0]["role"] == "system", "First message should be system"
    assert enhanced[1]["role"] == "user", "Second message should be user"
    assert "MUST use the available tools" in enhanced[1]["content"], "Should contain tool mandate"
    assert "Never respond with explanatory text only" in enhanced[1]["content"], "Should contain text-only prohibition"

    print("✓ Tool usage mandate injected correctly")
    print(f"✓ Instruction: {enhanced[1]['content'][:100]}...")


if __name__ == "__main__":
    test_tool_usage_mandate_injection()
    print("\n✅ All tests passed!")
