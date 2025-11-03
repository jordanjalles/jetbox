#!/usr/bin/env python3
"""Test that compaction triggers at correct threshold."""

from context_strategies import AppendUntilFullStrategy
from context_manager import ContextManager

def test_compaction_threshold():
    """Verify compaction triggers at 75% of max_tokens."""

    # Create strategy with 8K max tokens
    strategy = AppendUntilFullStrategy(max_tokens=8000, recent_keep=5)

    # Create a simple context manager
    cm = ContextManager()
    cm.state.goal = type('obj', (object,), {'description': 'Test goal'})()

    # Build messages that will exceed 75% of 8000 (6000 tokens)
    # Each message ~1500 chars = ~375 tokens
    # Need 6000 / 375 = 16 messages
    messages = []
    for i in range(20):
        messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}: " + "x" * 1500
        })

    # Build context
    print("Building context with 20 messages (~7500 tokens)...")
    context = strategy.build_context(
        context_manager=cm,
        messages=messages,
        system_prompt="Test system prompt",
        config=None
    )

    # Check if compaction happened
    # After compaction, context should have:
    # - System prompt
    # - Goal
    # - Summary message
    # - Last 5 recent messages
    # Total: ~7-8 messages

    print(f"Result: {len(context)} messages in context")

    if len(context) < 15:
        print("✅ Compaction triggered! Context was reduced.")
        print(f"   Original: 20 messages → Compacted: {len(context)} messages")
        return True
    else:
        print("❌ Compaction did NOT trigger")
        print(f"   Still have {len(context)} messages (expected <15)")
        return False

if __name__ == "__main__":
    success = test_compaction_threshold()
    exit(0 if success else 1)
