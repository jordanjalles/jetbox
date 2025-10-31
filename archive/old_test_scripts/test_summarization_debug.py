#!/usr/bin/env python3
"""Debug context summarization to see what's being kept/removed."""

from context_strategies import HierarchicalStrategy
from context_manager import ContextManager, Goal, Task, Subtask

def test_summarization_debug():
    """Debug the summarization process."""

    # Setup
    strategy = HierarchicalStrategy(history_keep=12)
    cm = ContextManager()
    cm.state.goal = Goal(description="Test goal")
    cm.state.goal.tasks.append(Task(description="Test task"))
    cm.state.goal.tasks[0].subtasks.append(Subtask(description="Test subtask"))

    # Create messages that exceed 75% threshold
    messages = []
    large_content = "x" * 50000  # ~12500 tokens per message

    for i in range(10):  # 20 messages total
        messages.append({
            "role": "assistant",
            "content": f"Working on task {i}",
            "tool_calls": [f"write_file(path='file{i}.py', content='{large_content}')"]
        })
        messages.append({
            "role": "tool",
            "content": f"File file{i}.py written.\n\n{large_content}"
        })

    print(f"Created {len(messages)} messages")
    print(f"Estimated tokens in messages: {strategy.estimate_context_size([{'content': str(m)} for m in messages]):,}")

    # Build context
    print("\n" + "="*70)
    print("Building context...")
    print("="*70)

    context = strategy.build_context(
        context_manager=cm,
        messages=messages,
        system_prompt="System prompt " * 100,
        config=None
    )

    print("\n" + "="*70)
    print("CONTEXT BREAKDOWN")
    print("="*70)

    for i, msg in enumerate(context):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        tokens = len(content) // 4

        # Check if this is the summary message
        is_summary = "Previous work summary" in content

        if is_summary:
            print(f"\n{i}. [{role}] SUMMARY MESSAGE")
            print(f"   Length: {len(content):,} chars ({tokens:,} tokens)")
            print(f"   First 200 chars: {content[:200]}")
            print(f"   Word count: ~{len(content.split())} words")
        elif len(content) > 500:
            print(f"{i}. [{role}] {len(content):,} chars ({tokens:,} tokens)")
        else:
            print(f"{i}. [{role}] {len(content)} chars: {content[:80]}...")

    print("\n" + "="*70)
    print("TOTALS")
    print("="*70)
    total_tokens = strategy.estimate_context_size(context)
    print(f"Total context messages: {len(context)}")
    print(f"Total context tokens: {total_tokens:,}")
    print(f"Percentage of 128K: {total_tokens/131072*100:.1f}%")

if __name__ == "__main__":
    test_summarization_debug()
