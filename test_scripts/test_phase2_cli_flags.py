#!/usr/bin/env python3
"""
Test script for Phase 2 CLI flag system.

Tests parse_extra_behaviors() function with various input formats.
"""
import sys
from agent import parse_extra_behaviors


def test_parse_extra_behaviors():
    """Test CLI flag parsing with various formats."""

    test_cases = [
        # (input_argv, expected_behaviors, expected_remaining)
        (
            ["agent.py", "--ContextInspector", "my goal"],
            ["ContextInspectorBehavior"],
            ["agent.py", "my goal"]
        ),
        (
            ["agent.py", "--ContextInspectorBehavior", "my goal"],
            ["ContextInspectorBehavior"],
            ["agent.py", "my goal"]
        ),
        (
            ["agent.py", "--StatusDisplay", "--ContextInspector", "my goal"],
            ["StatusDisplayBehavior", "ContextInspectorBehavior"],
            ["agent.py", "my goal"]
        ),
        (
            ["agent.py", "--team", "solo", "--ContextInspector", "my goal"],
            ["ContextInspectorBehavior"],
            ["agent.py", "--team", "solo", "my goal"]
        ),
        (
            ["agent.py", "my goal"],
            [],
            ["agent.py", "my goal"]
        ),
        (
            ["agent.py", "--workspace", "/path", "--ContextInspector"],
            ["ContextInspectorBehavior"],
            ["agent.py", "--workspace", "/path"]
        ),
    ]

    all_passed = True
    for i, (input_argv, expected_behaviors, expected_remaining) in enumerate(test_cases, 1):
        behaviors, remaining = parse_extra_behaviors(input_argv)

        if behaviors != expected_behaviors:
            print(f"❌ Test {i} FAILED: behaviors")
            print(f"   Input: {input_argv}")
            print(f"   Expected behaviors: {expected_behaviors}")
            print(f"   Got behaviors: {behaviors}")
            all_passed = False
        elif remaining != expected_remaining:
            print(f"❌ Test {i} FAILED: remaining args")
            print(f"   Input: {input_argv}")
            print(f"   Expected remaining: {expected_remaining}")
            print(f"   Got remaining: {remaining}")
            all_passed = False
        else:
            print(f"✅ Test {i} passed")

    return all_passed


if __name__ == "__main__":
    print("Testing Phase 2 CLI flag parsing...\n")

    if test_parse_extra_behaviors():
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
