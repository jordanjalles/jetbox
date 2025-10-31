#!/usr/bin/env python3
"""Test context summarization with realistic agentic activity."""

from context_strategies import HierarchicalStrategy
from context_manager import ContextManager, Goal, Task, Subtask


def create_realistic_messages(num_exchanges: int = 20) -> list[dict]:
    """
    Create realistic agent-LLM exchanges.

    Realistic pattern:
    - Assistant thinks/plans (100-300 chars)
    - Assistant calls tool
    - Tool responds (varies: 50 chars for success, 2-5K for file reads)

    Args:
        num_exchanges: Number of assistant-tool pairs to create

    Returns:
        List of message dicts with realistic sizes
    """
    messages = []

    # Create realistic file contents (larger to simulate real files)
    models_code = """from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class Comment:
    author: str
    content: str
    created_at: datetime

@dataclass
class Post:
    title: str
    content: str
    author: str
    created_at: datetime
    comments: List[Comment]
    id: Optional[str] = None
""" * 20  # ~2.5K chars

    manager_code = """class BlogManager:
    def __init__(self):
        self.posts = {}
        self.next_id = 1

    def add_post(self, title, content, author):
        post = Post(
            title=title,
            content=content,
            author=author,
            created_at=datetime.now(),
            comments=[],
            id=str(self.next_id)
        )
        self.posts[post.id] = post
        self.next_id += 1
        return post

    def add_comment(self, post_id, author, content):
        post = self.posts.get(post_id)
        if not post:
            return None
        comment = Comment(author=author, content=content, created_at=datetime.now())
        post.comments.append(comment)
        return comment

    def get_post(self, post_id):
        return self.posts.get(post_id)

    def list_posts(self):
        return list(self.posts.values())
""" * 15  # ~3.5K chars

    test_code = """import pytest
from blog.models import Post, Comment
from blog.manager import BlogManager
from datetime import datetime

def test_add_post():
    manager = BlogManager()
    post = manager.add_post('Title', 'Content', 'Author')
    assert post.title == 'Title'
    assert post.content == 'Content'
    assert post.author == 'Author'
    assert len(post.comments) == 0

def test_add_comment():
    manager = BlogManager()
    post = manager.add_post('Post 1', 'Content', 'Alice')
    comment = manager.add_comment(post.id, 'Bob', 'Great post!')
    assert comment.author == 'Bob'
    assert len(post.comments) == 1

def test_get_post():
    manager = BlogManager()
    post1 = manager.add_post('Post 1', 'Content', 'Alice')
    retrieved = manager.get_post(post1.id)
    assert retrieved.title == 'Post 1'

def test_list_posts():
    manager = BlogManager()
    manager.add_post('Post 1', 'Content 1', 'Alice')
    manager.add_post('Post 2', 'Content 2', 'Bob')
    posts = manager.list_posts()
    assert len(posts) == 2
""" * 10  # ~4K chars

    # Simulate realistic agent workflow
    tasks = [
        ("write_file", "blog/models.py", models_code, "File written successfully."),
        ("read_file", "blog/models.py", None, models_code),
        ("run_cmd", "pytest tests/", None, "===== test session starts =====\ncollected 5 items\n\ntests/test_blog.py .....    [100%]\n\n===== 5 passed in 0.23s ====="),
        ("write_file", "blog/manager.py", manager_code, "File written successfully."),
        ("read_file", "blog/manager.py", None, manager_code),
        ("run_cmd", "ruff check .", None, "All checks passed!"),
        ("write_file", "tests/test_blog.py", test_code, "File written successfully."),
        ("read_file", "tests/test_blog.py", None, test_code),
    ]

    for i in range(num_exchanges):
        task_idx = i % len(tasks)
        tool_name, arg1, arg2, tool_result = tasks[task_idx]

        # Assistant message (planning/thinking)
        thinking = f"I need to {tool_name.replace('_', ' ')} to make progress on the current subtask. "
        if tool_name == "write_file":
            thinking += f"Creating {arg1} with the necessary implementation."
        elif tool_name == "read_file":
            thinking += f"Let me check the contents of {arg1} to verify."
        elif tool_name == "run_cmd":
            thinking += f"Running {arg1} to validate the code."

        messages.append({
            "role": "assistant",
            "content": thinking,
            "tool_calls": [{
                "name": tool_name,
                "arguments": {"path": arg1, "content": arg2} if arg2 else {"path": arg1} if tool_name == "read_file" else {"cmd": arg1}
            }]
        })

        # Tool result
        messages.append({
            "role": "tool",
            "content": tool_result
        })

    return messages


def test_realistic_summarization():
    """Test summarization with realistic agent activity."""

    # Setup - use large history_keep to force compaction trigger
    # Normal is 12, but we want to test compaction so we'll keep 80 (160 messages)
    strategy = HierarchicalStrategy(history_keep=80)
    cm = ContextManager()
    cm.state.goal = Goal(description="Create blog system with models and manager")
    cm.state.goal.tasks.append(Task(description="Implement BlogManager"))
    cm.state.goal.tasks[0].subtasks.append(Subtask(description="Write tests"))

    # Create realistic messages
    # With larger file contents (~3K per read), 100 exchanges = ~300K chars = ~75K tokens
    # Need to exceed 75% of 128K = 98K tokens total (system + messages)
    messages = create_realistic_messages(100)

    print("\n" + "="*70)
    print("REALISTIC SUMMARIZATION TEST")
    print("="*70)
    print(f"Created {len(messages)} messages")

    # Calculate sizes
    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
    estimated_tokens = total_chars // 4
    print(f"Total size: {total_chars:,} chars (~{estimated_tokens:,} tokens)")

    # Show sample messages
    print("\n" + "="*70)
    print("SAMPLE MESSAGES (first 3 exchanges)")
    print("="*70)
    for i, msg in enumerate(messages[:6]):  # First 3 exchanges
        role = msg.get("role")
        content = str(msg.get("content", ""))
        print(f"\n{i}. [{role}] {len(content)} chars:")
        print(f"   {content[:150]}{'...' if len(content) > 150 else ''}")

    # Add large system prompt to exceed 75% threshold
    # Need: 98K tokens total = 25K system + 73K messages
    # We have ~75K tokens in messages, need ~23K in system
    # 23K tokens = 92K chars
    print("\n" + "="*70)
    print("BUILDING CONTEXT (will exceed 75% threshold)")
    print("="*70)

    large_system_prompt = "System prompt with tools and context. " * 2300  # ~92K chars = ~23K tokens

    context = strategy.build_context(
        context_manager=cm,
        messages=messages,
        system_prompt=large_system_prompt,
        config=None
    )

    print("\n" + "="*70)
    print("CONTEXT BREAKDOWN")
    print("="*70)

    summary_found = False
    summary_content = ""

    for i, msg in enumerate(context):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        tokens = len(content) // 4

        # Check if this is the summary message
        is_summary = "Previous work summary" in content

        if is_summary:
            summary_found = True
            summary_content = content
            print(f"\n{i}. [{role}] ⭐ SUMMARY MESSAGE")
            print(f"   Length: {len(content):,} chars (~{tokens:,} tokens)")
            print(f"   Content preview:")
            print(f"   {content[:300]}...")
        elif len(content) > 500:
            print(f"{i}. [{role}] {len(content):,} chars (~{tokens:,} tokens)")
        else:
            truncated = content[:100] + "..." if len(content) > 100 else content
            print(f"{i}. [{role}] {len(content)} chars: {truncated}")

    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    # Verify summary was created
    if summary_found:
        print("✓ Summary message found in context")
        summary_tokens = len(summary_content) // 4
        print(f"✓ Summary size: ~{summary_tokens:,} tokens")

        # Check that summary is actually compact (should be <2000 tokens)
        if summary_tokens < 2000:
            print(f"✓ Summary is compact (<2000 tokens)")
        else:
            print(f"✗ Summary too large ({summary_tokens:,} tokens)")

        # Count recent messages kept
        recent_count = sum(1 for msg in context if msg.get("role") in ["assistant", "tool"] and "Previous work summary" not in str(msg.get("content", "")))
        print(f"✓ Kept {recent_count} recent messages")

    else:
        print("✗ No summary found - compaction may not have triggered")

    # Show final size
    final_tokens = strategy.estimate_context_size(context)
    print(f"\nFinal context: {final_tokens:,} tokens")
    print(f"Original messages: ~{estimated_tokens:,} tokens")
    if summary_found and estimated_tokens > 0:
        reduction = (1 - final_tokens/estimated_tokens) * 100
        print(f"Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    test_realistic_summarization()
