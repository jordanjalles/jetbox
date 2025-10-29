#!/usr/bin/env python3
"""
Standalone script to analyze context diagnostic logs for information leakage.

Usage:
    python analyze_context_log.py
"""
import json
from pathlib import Path

DIAGNOSTIC_LOG = Path(".agent_context/context_diagnostic.jsonl")


def main():
    """Analyze the diagnostic log to detect information leakage between subtasks."""
    if not DIAGNOSTIC_LOG.exists():
        print("❌ No diagnostic log found. Run agent with diagnostic logging enabled first.")
        print(f"   Expected location: {DIAGNOSTIC_LOG}")
        return 1

    entries = []
    with DIAGNOSTIC_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️  Skipping malformed line: {line[:80]}")

    if not entries:
        print("❌ Diagnostic log is empty")
        return 1

    print(f"\n{'='*70}")
    print(f"CONTEXT LEAKAGE ANALYSIS")
    print(f"{'='*70}\n")

    # Track subtask transitions
    last_subtask = None
    leaks_detected = []
    clean_transitions = []

    for entry in entries:
        turn = entry["turn"]
        subtask = entry["subtask"]
        msg_history = entry["context_structure"]["message_history_count"]
        notes = entry.get("notes", "")

        # Detect subtask transition
        if last_subtask and subtask != last_subtask and subtask != "(none)":
            print(f"Turn {turn}: SUBTASK TRANSITION")
            print(f"  From: {last_subtask[:60]}")
            print(f"  To:   {subtask[:60]}")
            print(f"  Message history count: {msg_history}")

            if notes == "SUBTASK_TRANSITION_CLEAR":
                print(f"  ℹ️  Transition clear marker detected")

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
                clean_transitions.append({
                    "turn": turn,
                    "from": last_subtask,
                    "to": subtask
                })

        last_subtask = subtask

    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total turns analyzed: {len(entries)}")
    print(f"Clean transitions: {len(clean_transitions)}")
    print(f"Leaks detected: {len(leaks_detected)}")

    if clean_transitions:
        print(f"\n✅ CLEAN TRANSITIONS:")
        for trans in clean_transitions:
            print(f"  • Turn {trans['turn']}: Successfully isolated transition")

    if leaks_detected:
        print(f"\n❌ CONTEXT LEAKAGE CONFIRMED")
        print(f"\nDetails:")
        for leak in leaks_detected:
            print(f"  • Turn {leak['turn']}: {leak['leaked_messages']} messages leaked")
            print(f"    from '{leak['from'][:60]}'")
            print(f"    to   '{leak['to'][:60]}'")
        return 1
    else:
        print(f"\n✅ NO LEAKAGE: Context isolation working correctly")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
