#!/usr/bin/env python3
"""Test that append strategy also uses LLM-based compaction."""

from context_strategies import AppendUntilFullStrategy
from context_manager import ContextManager, Goal


def test_append_compaction():
    """Verify append strategy uses LLM summarization at 75% threshold."""

    # Setup with default recent_keep=20
    strategy = AppendUntilFullStrategy(recent_keep=20)
    cm = ContextManager()
    cm.state.goal = Goal(description="Test goal")

    # Create messages with realistic content
    messages = []

    # Simulate 100 read_file operations with large results
    large_file = "class BlogManager:\n" * 500  # ~8K chars per message

    for i in range(100):
        messages.append({
            "role": "assistant",
            "content": f"Reading file {i}...",
        })
        messages.append({
            "role": "tool",
            "content": f"Contents of file {i}:\n{large_file}"
        })

    print(f"Created {len(messages)} messages")

    # Calculate total size
    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
    print(f"Total: {total_chars:,} chars (~{total_chars // 4:,} tokens)")

    # Build context with large system prompt to exceed 75%
    large_system = "System prompt. " * 5000  # ~20K tokens

    print("\nBuilding context...")
    context = strategy.build_context(
        context_manager=cm,
        messages=messages,
        system_prompt=large_system,
        config=None
    )

    # Check if compaction triggered
    summary_found = any("Previous work summary" in str(msg.get("content", "")) for msg in context)

    print(f"\nResults:")
    print(f"  Context messages: {len(context)}")
    print(f"  Summary found: {summary_found}")
    print(f"  Final tokens: ~{strategy.estimate_context_size(context):,}")

    if summary_found:
        print("\n✓ Append strategy compaction is working!")
    else:
        print("\n✗ Compaction did not trigger")


if __name__ == "__main__":
    test_append_compaction()
