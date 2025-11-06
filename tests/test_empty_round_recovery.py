#!/usr/bin/env python3
"""
Test the enhanced empty round recovery mechanism with architect.

This test uses the architect agent (which has been showing empty rounds
in L7 evaluations) to verify that the recovery mechanism works correctly.
"""

from pathlib import Path
from architect_agent import ArchitectAgent

def test_empty_round_recovery():
    """Test recovery prompts for architect in empty round scenario."""

    print("="*80)
    print("EMPTY ROUND RECOVERY TEST - Architect Agent")
    print("="*80)
    print("\nThis test will:")
    print("  1. Create architect agent with a complex delegated goal")
    print("  2. Run for limited rounds (expect some empty rounds with gpt-oss:20b)")
    print("  3. Verify recovery prompts are injected after 3 empty rounds")
    print("  4. Check that prompts show goal, tools, and completion signals")
    print()

    # Create architect agent with complex task (like L7 delegation)
    workspace = Path(".agent_workspaces/tests/architect_empty_round_test")
    agent = ArchitectAgent(
        workspace=workspace,
        goal="Design architecture for a full-stack Flask application with user authentication, posts, and comments using SQLite"
    )

    print("Created architect with complex architecture goal")
    print(f"Workspace: {workspace}")
    print()

    # Run for limited rounds - expect to see empty rounds and recovery
    print("Running architect (max 10 rounds to observe recovery behavior)...")
    print("="*80)
    print()

    result = agent.run(max_rounds=10)

    print("\n" + "="*80)
    print("TEST RESULT")
    print("="*80)
    print(f"Status: {result.get('status')}")
    print(f"Rounds completed: {result.get('round', 'N/A')}")
    print()

    # Check loop detection behavior state
    loop_behavior = None
    for behavior in agent._behaviors:
        if behavior.get_name() == 'loop_detection':
            loop_behavior = behavior
            break

    if loop_behavior:
        print("LOOP DETECTION STATS:")
        print(f"  Empty rounds detected: {loop_behavior.consecutive_empty_rounds}")
        print(f"  Recovery prompt injected: {loop_behavior.recovery_prompt_injected}")
        print(f"  Total actions tracked: {len(loop_behavior.action_history)}")
        print()

        # Analyze results
        if loop_behavior.consecutive_empty_rounds >= 3:
            print("✓ Empty rounds detected (as expected with gpt-oss:20b on complex tasks)")
            if loop_behavior.recovery_prompt_injected:
                print("✓ Recovery prompt was injected")
                print()
                print("RECOVERY MECHANISM WORKING:")
                print("  ✓ Empty rounds detected")
                print("  ✓ Recovery prompt injected into context")
                print("  ✓ Prompt should include goal, tools, and completion signals")
            else:
                print("✗ WARNING: Recovery prompt NOT injected (check threshold)")
        elif len(loop_behavior.action_history) > 0:
            print("✓ No empty rounds - architect called tools successfully!")
            print(f"  Tools used: {len(loop_behavior.action_history)} total actions")
        else:
            print("⚠️  No actions tracked - model may have produced only text responses")
    else:
        print("✗ ERROR: LoopDetectionBehavior not found on architect agent")

    print()
    print("="*80)
    print("WHAT TO LOOK FOR IN LOGS ABOVE:")
    print("  1. '[loop_detection] ⚠️  Empty round #N' - empty round detection")
    print("  2. '[loop_detection] LLM response: ...' - response preview")
    print("  3. '[loop_detection] Injecting empty round recovery' - recovery triggered")
    print("  4. '🚨 WARNING/CRITICAL' in subsequent rounds - recovery prompt in context")
    print("="*80)

if __name__ == "__main__":
    test_empty_round_recovery()
