"""
Test script to demonstrate context leakage between subtasks.

This runs the agent with a simple goal that should have two sequential subtasks,
then analyzes the diagnostic log to verify whether information leaks between them.
"""
import subprocess
import sys
import os
from pathlib import Path
import json

# Set a fast model for testing
os.environ["OLLAMA_MODEL"] = "qwen2.5-coder:3b"

def run_test_goal():
    """Run agent with a goal that requires two sequential subtasks."""
    goal = "Create hello.py with print('hello'), then create world.py with print('world')"

    print(f"Running agent with goal: {goal}")
    print("="*70)

    # Run agent
    result = subprocess.run(
        ["python", "agent.py", goal],
        capture_output=True,
        text=True,
        timeout=120
    )

    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print("\nReturn code:", result.returncode)

    return result.returncode == 0


def analyze_diagnostic_log():
    """Analyze the diagnostic log for information leakage."""
    log_path = Path(".agent_context/context_diagnostic.jsonl")

    if not log_path.exists():
        print("❌ No diagnostic log found!")
        return False

    print("\n" + "="*70)
    print("CONTEXT LEAKAGE ANALYSIS")
    print("="*70 + "\n")

    entries = []
    with log_path.open("r") as f:
        for line in f:
            entries.append(json.loads(line))

    # Track subtask transitions
    last_subtask = None
    leaks_detected = []

    for entry in entries:
        turn = entry["turn"]
        subtask = entry["subtask"]
        msg_history = entry["context_structure"]["message_history_count"]

        # Detect subtask transition
        if last_subtask and subtask != last_subtask and subtask != "(none)":
            print(f"Turn {turn}: SUBTASK TRANSITION")
            print(f"  From: {last_subtask}")
            print(f"  To:   {subtask}")
            print(f"  Message history count: {msg_history}")

            if msg_history > 0:
                print(f"  ⚠️  LEAK DETECTED: Message history should be 0 after transition")
                print(f"       but found {msg_history} messages from previous subtask\n")
                leaks_detected.append({
                    "turn": turn,
                    "from": last_subtask,
                    "to": subtask,
                    "leaked_messages": msg_history
                })
            else:
                print(f"  ✓ Clean transition: Message history correctly cleared\n")

        last_subtask = subtask

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total turns analyzed: {len(entries)}")
    print(f"Leaks detected: {len(leaks_detected)}")

    if leaks_detected:
        print("\n❌ CONTEXT LEAKAGE CONFIRMED")
        print("\nDetails:")
        for leak in leaks_detected:
            print(f"  • Turn {leak['turn']}: {leak['leaked_messages']} messages leaked")
            print(f"    from '{leak['from']}'")
            print(f"    to   '{leak['to']}'")
        return False
    else:
        print("\n✓ NO LEAKAGE: Context isolation working correctly")
        return True


def main():
    """Run test and analyze results."""
    print("CONTEXT LEAKAGE TEST")
    print("="*70)
    print("This test will:")
    print("1. Run agent with a multi-subtask goal")
    print("2. Analyze diagnostic logs for context leakage")
    print("3. Report whether messages leak between subtasks")
    print("="*70 + "\n")

    # Clean up old diagnostic log
    log_path = Path(".agent_context/context_diagnostic.jsonl")
    if log_path.exists():
        log_path.unlink()
        print("Cleaned up old diagnostic log\n")

    # Run test
    success = run_test_goal()

    if not success:
        print("\n❌ Agent failed to complete goal")
        return 1

    # Analyze results
    no_leaks = analyze_diagnostic_log()

    if no_leaks:
        print("\n✅ TEST PASSED: No context leakage detected")
        return 0
    else:
        print("\n❌ TEST FAILED: Context leakage detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
