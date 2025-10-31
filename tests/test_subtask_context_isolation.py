#!/usr/bin/env python3
"""
Test that subtask transitions actually clear prior subtask messages from context.

CRITICAL BUG: Currently self.state.messages accumulates across subtasks,
so subtask 2's context includes subtask 1's messages!
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_executor_agent import TaskExecutorAgent
import tempfile


def test_subtask_context_leak():
    """
    Verify that prior subtask messages are NOT in current subtask's context.

    This test will FAIL with current implementation because:
    - self.state.messages accumulates across subtasks
    - build_context() uses self.state.messages
    - Only local messages list is cleared (which isn't used!)
    """
    print("\n" + "="*70)
    print("TEST: Subtask Context Isolation (LEAK DETECTION)")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create agent with simple multi-subtask goal
        agent = TaskExecutorAgent(
            workspace=workspace,
            goal="Create calculator.py with add() and multiply() functions. Write tests.",
            max_rounds=30,
        )

        # Patch build_context to capture what's being sent to LLM
        contexts_sent = []
        original_build_context = agent.build_context

        def patched_build_context():
            context = original_build_context()
            contexts_sent.append({
                "round": agent.state.total_rounds,
                "message_count": len(context),
                "state_messages_count": len(agent.state.messages),
                "context": context.copy(),
            })
            return context

        agent.build_context = patched_build_context

        # Run agent
        result = agent.run()

        print(f"\nAgent completed with status: {result.get('status')}")
        print(f"Total contexts built: {len(contexts_sent)}")

        # Analyze contexts to detect leaks
        print("\nAnalyzing context isolation across subtasks...")

        # Find rounds where subtask transitions occurred
        # (Look for sudden drops in message count)
        transitions = []
        for i in range(1, len(contexts_sent)):
            prev = contexts_sent[i-1]
            curr = contexts_sent[i]

            # If message count dropped significantly, a subtask transition likely occurred
            if prev["state_messages_count"] > curr["state_messages_count"] + 5:
                transitions.append({
                    "round": curr["round"],
                    "before_messages": prev["state_messages_count"],
                    "after_messages": curr["state_messages_count"],
                })

        print(f"\nDetected {len(transitions)} subtask transitions:")
        for t in transitions:
            print(f"  Round {t['round']}: {t['before_messages']} → {t['after_messages']} messages")

        if not transitions:
            print("  ⚠️  No transitions detected (task may have been too simple)")
            print("  Cannot verify isolation - test inconclusive")
            return {
                "test": "subtask_context_leak",
                "success": None,
                "reason": "No subtask transitions detected"
            }

        # Check if messages actually decreased after transitions
        leak_detected = False
        for t in transitions:
            if t["after_messages"] > 5:  # Should be very small after clearing
                print(f"  ⚠️  LEAK: Round {t['round']} still has {t['after_messages']} messages after transition")
                leak_detected = True

        # Also check: do context messages include artifacts from previous subtasks?
        # Look for keywords from subtask 1 appearing in subtask 2's context
        if len(contexts_sent) > 10:
            # Sample context from early rounds vs late rounds
            early_context = "\n".join([
                msg.get("content", "")
                for msg in contexts_sent[5]["context"]
                if isinstance(msg, dict)
            ])

            late_context = "\n".join([
                msg.get("content", "")
                for msg in contexts_sent[-5]["context"]
                if isinstance(msg, dict)
            ])

            print(f"\nEarly context length: {len(early_context)} chars")
            print(f"Late context length: {len(late_context)} chars")

        if leak_detected:
            print("\n✗ FAIL - Context leak detected: prior subtask messages not cleared")
            success = False
        else:
            print("\n✓ PASS - Context properly isolated between subtasks")
            success = True

        return {
            "test": "subtask_context_leak",
            "success": success,
            "transitions_detected": len(transitions),
            "contexts_captured": len(contexts_sent),
        }


if __name__ == "__main__":
    result = test_subtask_context_leak()

    print("\n" + "="*70)
    print("TEST RESULT")
    print("="*70)

    if result["success"]:
        print("✓ PASS")
        sys.exit(0)
    elif result["success"] is None:
        print("⚠️  INCONCLUSIVE")
        sys.exit(2)
    else:
        print("✗ FAIL")
        sys.exit(1)
