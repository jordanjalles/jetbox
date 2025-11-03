#!/usr/bin/env python3
"""Test loop detection in ContextStrategy."""

from context_strategies import SubAgentStrategy

def test_loop_detection():
    """Test that loop detection triggers after repeated actions."""
    strategy = SubAgentStrategy()

    # Simulate same action repeated 5 times
    for i in range(7):
        warning = strategy.record_action(
            tool_name="run_bash",
            args={"command": "pytest -q"},
            result={"error": "ModuleNotFoundError: No module named 'api_client'", "rc": 2},
            success=False
        )

        print(f"Attempt {i+1}:")
        if warning:
            print(f"  ⚠️  LOOP DETECTED: {warning['warning']}")
            print(f"  Suggestion: {warning['suggestion'][:80]}...")
        else:
            print("  No loop detected yet")

    print(f"\n✅ Loop detection test complete")
    print(f"Action history length: {len(strategy.action_history)}")
    print(f"Loop warnings: {strategy.loop_warnings}")

if __name__ == "__main__":
    test_loop_detection()
