#!/usr/bin/env python3
"""
Validate that context management strategies keep context size bounded.

Tests both strategies:
1. Hierarchical (TaskExecutor): Last N messages + task context
2. Append-until-full (Orchestrator): Compaction when near limit
"""
import sys
import json
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from context_manager import ContextManager
from context_strategies import build_hierarchical_context, build_append_context
from agent_config import config


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Estimate token count for messages.

    Rough heuristic: 4 characters per token
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        total_chars += len(str(content))

        # Count tool calls
        if "tool_calls" in msg:
            for tc in msg.get("tool_calls", []):
                total_chars += len(tc["function"]["name"])
                total_chars += len(str(tc["function"]["arguments"]))

    return total_chars // 4


def test_hierarchical_context_bounded():
    """
    Test that hierarchical context stays bounded even with many messages.

    Expected: Context should contain:
    - System prompt
    - Task context (goal/task/subtask)
    - Last N messages only (not all history)
    """
    print("\n" + "="*70)
    print("TEST: Hierarchical Context - Bounded Growth")
    print("="*70)

    # Setup - just need goal initialized, tasks are optional
    ctx = ContextManager()
    ctx.load_or_init("Test goal: create a calculator")

    # Simulate many rounds of messages (should not all be included)
    messages = []
    for i in range(100):  # 100 message exchanges = 200 messages
        messages.append({
            "role": "assistant",
            "content": f"This is assistant message {i}" + ("x" * 100)  # Add padding
        })
        messages.append({
            "role": "user",
            "content": f"This is user message {i}" + ("x" * 100)
        })

    # Build context with hierarchical strategy
    system_prompt = "You are a test agent."
    context = build_hierarchical_context(
        context_manager=ctx,
        messages=messages,
        system_prompt=system_prompt,
        config=config,
        probe_state_func=None,
        workspace=None,
    )

    # Analyze
    total_messages = len(context)
    estimated_tokens = estimate_tokens(context)

    print(f"Total message history: {len(messages)} messages")
    print(f"Context sent to LLM: {total_messages} messages")
    print(f"Estimated tokens: {estimated_tokens}")

    # Expected: Should have system + task_info + last N messages only
    # Not all 200 messages!
    history_keep = config.context.history_keep if hasattr(config, 'context') else 12
    expected_max = 2 + (history_keep * 2)  # system + task_info + last N exchanges

    success = total_messages <= expected_max

    if success:
        print(f"✓ PASS - Context bounded to {total_messages} messages (expected ≤{expected_max})")
        print(f"  Kept last {history_keep} exchanges out of {len(messages)//2} total")
    else:
        print(f"✗ FAIL - Context too large: {total_messages} messages (expected ≤{expected_max})")

    return {
        "test": "hierarchical_bounded",
        "success": success,
        "total_history": len(messages),
        "context_size": total_messages,
        "expected_max": expected_max,
        "estimated_tokens": estimated_tokens,
    }


def test_hierarchical_context_isolation():
    """
    Test that hierarchical strategy limits context even with message history accumulation.

    The key insight: hierarchical strategy takes last N messages regardless of history size.
    This simulates what happens between subtasks - message list might grow, but context stays bounded.
    """
    print("\n" + "="*70)
    print("TEST: Hierarchical Context - Message History Independence")
    print("="*70)

    ctx = ContextManager()
    ctx.load_or_init("Test goal")

    # Simulate work with many messages accumulated
    messages_large_history = []
    for i in range(100):
        messages_large_history.append({"role": "assistant", "content": f"Message {i}"})
        messages_large_history.append({"role": "user", "content": f"Response {i}"})

    # Build context - should only include recent messages
    context1 = build_hierarchical_context(
        context_manager=ctx,
        messages=messages_large_history,
        system_prompt="System prompt",
        config=config,
    )

    large_history_size = len(context1)
    print(f"Context with large history (200 messages): {large_history_size} messages in context")

    # Now simulate with small history
    messages_small_history = []
    for i in range(5):
        messages_small_history.append({"role": "assistant", "content": f"Message {i}"})
        messages_small_history.append({"role": "user", "content": f"Response {i}"})

    context2 = build_hierarchical_context(
        context_manager=ctx,
        messages=messages_small_history,
        system_prompt="System prompt",
        config=config,
    )

    small_history_size = len(context2)
    print(f"Context with small history (10 messages): {small_history_size} messages in context")

    # Success: large history should not result in significantly larger context
    # Both should be bounded by history_keep parameter
    history_keep = config.context.history_keep if hasattr(config, 'context') else 12
    max_expected = 2 + (history_keep * 2)

    large_bounded = large_history_size <= max_expected
    small_bounded = small_history_size <= max_expected

    success = large_bounded and small_bounded

    if success:
        print(f"✓ PASS - Context bounded regardless of history size")
        print(f"  Large history context: {large_history_size} (≤{max_expected})")
        print(f"  Small history context: {small_history_size} (≤{max_expected})")
    else:
        print(f"✗ FAIL - Context not properly bounded")
        if not large_bounded:
            print(f"  Large history: {large_history_size} > {max_expected}")
        if not small_bounded:
            print(f"  Small history: {small_history_size} > {max_expected}")

    return {
        "test": "hierarchical_isolation",
        "success": success,
        "large_history_context": large_history_size,
        "small_history_context": small_history_size,
        "max_expected": max_expected,
    }


def test_orchestrator_compaction():
    """
    Test that orchestrator's append-until-full strategy compacts when needed.
    """
    print("\n" + "="*70)
    print("TEST: Orchestrator Context - Compaction Behavior")
    print("="*70)

    system_prompt = "You are an orchestrator."
    max_tokens = 8000

    # Simulate many messages that would exceed 80% of token limit
    messages = []
    for i in range(200):  # Many exchanges
        messages.append({
            "role": "user",
            "content": f"User message {i}: " + ("x" * 200)  # Padding to increase size
        })
        messages.append({
            "role": "assistant",
            "content": f"Assistant response {i}: " + ("x" * 200)
        })

    # Build context with append strategy
    context = build_append_context(
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        recent_keep=20,
    )

    # Analyze
    total_messages = len(context)
    estimated_tokens = estimate_tokens(context)

    print(f"Total message history: {len(messages)} messages")
    print(f"Context sent to LLM: {total_messages} messages")
    print(f"Estimated tokens: {estimated_tokens} (limit: {max_tokens})")

    # Check if compaction happened
    # If compaction worked, context should be much smaller than full history
    compaction_happened = total_messages < len(messages) * 0.3  # Should be < 30% of original

    # Check that we stayed under token limit
    under_limit = estimated_tokens < max_tokens

    success = compaction_happened and under_limit

    if success:
        print(f"✓ PASS - Compaction working")
        print(f"  Original: {len(messages)} messages")
        print(f"  Compacted: {total_messages} messages")
        print(f"  Tokens: {estimated_tokens}/{max_tokens} ({estimated_tokens/max_tokens*100:.1f}%)")
    else:
        if not compaction_happened:
            print(f"✗ FAIL - Compaction not triggered")
        if not under_limit:
            print(f"✗ FAIL - Exceeded token limit: {estimated_tokens}/{max_tokens}")

    return {
        "test": "orchestrator_compaction",
        "success": success,
        "original_messages": len(messages),
        "compacted_messages": total_messages,
        "estimated_tokens": estimated_tokens,
        "token_limit": max_tokens,
        "utilization": f"{estimated_tokens/max_tokens*100:.1f}%",
    }


def test_orchestrator_no_compaction_when_small():
    """
    Test that orchestrator doesn't compact when context is small.
    """
    print("\n" + "="*70)
    print("TEST: Orchestrator Context - No Compaction When Small")
    print("="*70)

    system_prompt = "You are an orchestrator."
    max_tokens = 8000

    # Small conversation that shouldn't trigger compaction
    messages = []
    for i in range(10):  # Just 10 exchanges
        messages.append({"role": "user", "content": f"Short message {i}"})
        messages.append({"role": "assistant", "content": f"Short response {i}"})

    context = build_append_context(
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        recent_keep=20,
    )

    # Should have: system + all messages (no compaction)
    expected = 1 + len(messages)  # system + all messages
    actual = len(context)

    success = actual == expected
    estimated_tokens = estimate_tokens(context)

    print(f"Message history: {len(messages)} messages")
    print(f"Context: {actual} messages (expected {expected})")
    print(f"Estimated tokens: {estimated_tokens} (under 80% threshold)")

    if success:
        print(f"✓ PASS - No unnecessary compaction")
    else:
        print(f"✗ FAIL - Compaction triggered unnecessarily")

    return {
        "test": "orchestrator_no_compaction",
        "success": success,
        "messages": len(messages),
        "context_size": actual,
        "estimated_tokens": estimated_tokens,
    }


def main():
    """Run all context size validation tests."""
    print("="*70)
    print("CONTEXT SIZE VALIDATION TESTS")
    print("="*70)
    print()
    print("Testing both context management strategies:")
    print("1. Hierarchical (TaskExecutor)")
    print("2. Append-until-full with compaction (Orchestrator)")

    results = []

    # Run hierarchical tests
    try:
        results.append(test_hierarchical_context_bounded())
    except Exception as e:
        print(f"✗ Error in hierarchical_bounded: {e}")
        import traceback
        traceback.print_exc()
        results.append({"test": "hierarchical_bounded", "success": False, "error": str(e)})

    try:
        results.append(test_hierarchical_context_isolation())
    except Exception as e:
        print(f"✗ Error in hierarchical_isolation: {e}")
        import traceback
        traceback.print_exc()
        results.append({"test": "hierarchical_isolation", "success": False, "error": str(e)})

    # Run orchestrator tests
    try:
        results.append(test_orchestrator_compaction())
    except Exception as e:
        print(f"✗ Error in orchestrator_compaction: {e}")
        import traceback
        traceback.print_exc()
        results.append({"test": "orchestrator_compaction", "success": False, "error": str(e)})

    try:
        results.append(test_orchestrator_no_compaction_when_small())
    except Exception as e:
        print(f"✗ Error in orchestrator_no_compaction: {e}")
        import traceback
        traceback.print_exc()
        results.append({"test": "orchestrator_no_compaction", "success": False, "error": str(e)})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r.get("success"))
    total = len(results)

    for result in results:
        status = "✓ PASS" if result.get("success") else "✗ FAIL"
        print(f"{status} - {result['test']}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Save results
    output_file = Path("context_size_validation_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "summary": {"passed": passed, "total": total},
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
