"""
Test context compaction effectiveness using real timeout dumps.

This script:
1. Loads contexts from timeout dumps
2. Measures token counts before compaction
3. Runs contexts through CompactWhenNearFullBehavior
4. Measures token counts after compaction
5. Reports effectiveness metrics
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from behaviors.compact_when_near_full import CompactWhenNearFullBehavior


def estimate_tokens_simple(context: list[dict[str, Any]]) -> int:
    """Simple token estimation using 4 chars per token."""
    total_chars = 0
    for msg in context:
        # Count content
        content = msg.get("content", "")
        if content:
            total_chars += len(str(content))

        # Count tool_calls (can be huge!)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            # Convert to string (may not be JSON serializable)
            total_chars += len(str(tool_calls))

        # Count tool call args if present
        if "function" in msg:
            # Convert to string (may not be JSON serializable)
            total_chars += len(str(msg["function"]))

        # Add overhead for role, structure
        total_chars += 20

    return total_chars // 4


def load_context_from_dump(dump_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load context and metadata from timeout dump."""
    with open(dump_path, 'r', encoding='utf-8') as f:
        dump = json.load(f)

    # Try both "context" and "messages" keys
    context = dump.get("messages", dump.get("context", []))

    metadata = {
        "agent": dump.get("agent_name", "unknown"),
        "timestamp": dump.get("timestamp", "unknown"),
        "timeout_type": dump.get("timeout_type", "unknown"),
        "timeout_after": dump.get("elapsed_time_seconds", 0),
        "context_stats": dump.get("context_stats", {}),
    }

    return context, metadata


def test_compaction(context: list[dict[str, Any]], max_tokens: int = 64000) -> dict[str, Any]:
    """
    Test compaction on a context.

    Returns dict with:
    - before_tokens: tokens before compaction
    - after_tokens: tokens after compaction
    - before_messages: message count before
    - after_messages: message count after
    - reduction_percent: percentage reduction
    - triggered_emergency: whether emergency mode triggered
    """
    # Create compaction behavior
    behavior = CompactWhenNearFullBehavior(
        max_tokens=max_tokens,
        compact_threshold=0.75,
        keep_recent=20
    )

    # Measure before
    before_tokens = estimate_tokens_simple(context)
    before_messages = len(context)
    before_percent = (before_tokens / max_tokens) * 100

    # Apply compaction
    compacted_context = behavior.enhance_context(context.copy())

    # Measure after
    after_tokens = estimate_tokens_simple(compacted_context)
    after_messages = len(compacted_context)
    after_percent = (after_tokens / max_tokens) * 100

    # Calculate metrics
    reduction_tokens = before_tokens - after_tokens
    reduction_percent = (reduction_tokens / before_tokens * 100) if before_tokens > 0 else 0
    triggered_emergency = before_percent > 100

    return {
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "before_messages": before_messages,
        "after_messages": after_messages,
        "before_percent": before_percent,
        "after_percent": after_percent,
        "reduction_tokens": reduction_tokens,
        "reduction_percent": reduction_percent,
        "triggered_emergency": triggered_emergency,
    }


def main():
    """Run compaction tests on timeout dumps."""
    dumps_dir = Path(".agent_context/timeout_dumps")

    if not dumps_dir.exists():
        print(f"❌ Dumps directory not found: {dumps_dir}")
        return

    # Find all dump files
    dump_files = sorted(dumps_dir.glob("*.json"))

    if not dump_files:
        print(f"❌ No dump files found in {dumps_dir}")
        return

    print(f"Found {len(dump_files)} timeout dumps")
    print("=" * 80)

    # Test each dump
    results = []
    max_tests = min(20, len(dump_files))

    for i, dump_path in enumerate(dump_files[:max_tests], 1):
        print(f"\n[{i}/{max_tests}] Testing: {dump_path.name}")

        try:
            # Load context
            context, metadata = load_context_from_dump(dump_path)

            if not context:
                print("  ⚠️  Empty context, skipping")
                continue

            # Test compaction
            result = test_compaction(context)
            result["dump_file"] = dump_path.name
            result["metadata"] = metadata
            results.append(result)

            # Print result
            print(f"  Agent: {metadata['agent']}")
            print(f"  Before: {result['before_tokens']:,} tokens ({result['before_percent']:.1f}%) in {result['before_messages']} messages")
            print(f"  After:  {result['after_tokens']:,} tokens ({result['after_percent']:.1f}%) in {result['after_messages']} messages")
            print(f"  Reduction: {result['reduction_tokens']:,} tokens ({result['reduction_percent']:.1f}%)")

            if result['triggered_emergency']:
                print(f"  🚨 Emergency mode triggered (>100% capacity)")

            if result['after_percent'] > 100:
                print(f"  ⚠️  STILL OVER LIMIT after compaction!")
            elif result['after_percent'] > 75:
                print(f"  ⚠️  Still near limit after compaction")
            else:
                print(f"  ✓ Successfully reduced to safe level")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Summary statistics
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        avg_before = sum(r['before_tokens'] for r in results) / len(results)
        avg_after = sum(r['after_tokens'] for r in results) / len(results)
        avg_reduction = sum(r['reduction_percent'] for r in results) / len(results)

        emergency_count = sum(1 for r in results if r['triggered_emergency'])
        still_over_count = sum(1 for r in results if r['after_percent'] > 100)
        safe_count = sum(1 for r in results if r['after_percent'] <= 75)

        print(f"\nTested: {len(results)} dumps")
        print(f"Average before: {avg_before:,.0f} tokens")
        print(f"Average after:  {avg_after:,.0f} tokens")
        print(f"Average reduction: {avg_reduction:.1f}%")
        print(f"\nEmergency mode triggered: {emergency_count}/{len(results)} ({emergency_count/len(results)*100:.1f}%)")
        print(f"Still over limit after compaction: {still_over_count}/{len(results)} ({still_over_count/len(results)*100:.1f}%)")
        print(f"Reduced to safe level (<75%): {safe_count}/{len(results)} ({safe_count/len(results)*100:.1f}%)")

        # Effectiveness assessment
        print("\n" + "=" * 80)
        print("ASSESSMENT")
        print("=" * 80)

        if still_over_count > 0:
            print("⚠️  ISSUE: Some contexts still over limit after compaction")
            print("   → Emergency mode and hard limit enforcement need verification")

        if avg_reduction < 20:
            print("⚠️  ISSUE: Low average reduction ({avg_reduction:.1f}%)")
            print("   → Compaction may not be aggressive enough")
        elif avg_reduction > 50:
            print("✓ GOOD: High average reduction ({avg_reduction:.1f}%)")
            print("   → Compaction is effective")

        if safe_count == len(results):
            print("✓ EXCELLENT: All contexts reduced to safe levels")
        elif safe_count / len(results) > 0.8:
            print("✓ GOOD: Most contexts reduced to safe levels")
        else:
            print("⚠️  ISSUE: Many contexts still near/over limit")


if __name__ == "__main__":
    main()
