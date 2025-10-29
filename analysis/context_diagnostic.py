"""
Diagnostic tool to log and visualize context sent to Ollama at each turn.

This helps verify that hierarchical context management is working as expected.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Any

DIAGNOSTIC_LOG = Path(".agent_context/context_diagnostic.jsonl")

def log_context_sent(
    turn_number: int,
    full_context: list[dict[str, Any]],
    subtask_description: str | None = None,
    notes: str = ""
) -> None:
    """
    Log the complete context being sent to Ollama for this turn.

    Args:
        turn_number: Current turn/round number
        full_context: The full context list being sent to Ollama
        subtask_description: Description of current subtask (if any)
        notes: Additional notes (e.g., "subtask transition")
    """
    DIAGNOSTIC_LOG.parent.mkdir(exist_ok=True)

    # Analyze context structure
    system_msg = None
    user_msgs = []
    assistant_msgs = []
    tool_msgs = []

    for msg in full_context:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            system_msg = content
        elif role == "user":
            user_msgs.append(content)
        elif role == "assistant":
            assistant_msgs.append({
                "content": content,
                "tool_calls": len(msg.get("tool_calls", []))
            })
        elif role == "tool":
            tool_msgs.append(content[:200] if isinstance(content, str) else str(content)[:200])

    # Create diagnostic entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "turn": turn_number,
        "subtask": subtask_description or "(none)",
        "notes": notes,
        "stats": {
            "total_messages": len(full_context),
            "system_messages": 1 if system_msg else 0,
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "tool_messages": len(tool_msgs),
            "total_chars": sum(len(str(m.get("content", ""))) for m in full_context),
        },
        "context_structure": {
            "system_prompt_preview": system_msg[:200] if system_msg else None,
            "task_context": user_msgs[0][:300] if user_msgs else None,
            "message_history_count": len(user_msgs) + len(assistant_msgs) - 1,  # Exclude first user (task context)
        }
    }

    # Append to log
    with DIAGNOSTIC_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def print_context_summary(turn_number: int, full_context: list[dict[str, Any]]) -> None:
    """Print a human-readable summary of context for this turn."""
    print(f"\n{'='*70}")
    print(f"CONTEXT DIAGNOSTIC - Turn {turn_number}")
    print(f"{'='*70}")

    for i, msg in enumerate(full_context):
        role = msg.get("role")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Truncate content for display
        content_preview = str(content)[:150].replace("\n", " ")
        if len(str(content)) > 150:
            content_preview += "..."

        tool_info = f" [+{len(tool_calls)} tool calls]" if tool_calls else ""

        print(f"{i+1}. [{role:10}] {content_preview}{tool_info}")

    print(f"{'='*70}\n")


def analyze_context_log() -> None:
    """Analyze the diagnostic log to detect information leakage between subtasks."""
    if not DIAGNOSTIC_LOG.exists():
        print("No diagnostic log found. Run agent with context logging enabled first.")
        return

    entries = []
    with DIAGNOSTIC_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"\n{'='*70}")
    print(f"CONTEXT LEAKAGE ANALYSIS")
    print(f"{'='*70}\n")

    # Track subtask transitions
    last_subtask = None
    for entry in entries:
        turn = entry["turn"]
        subtask = entry["subtask"]
        msg_history = entry["context_structure"]["message_history_count"]
        notes = entry.get("notes", "")

        # Detect subtask transition
        if last_subtask and subtask != last_subtask:
            print(f"Turn {turn}: SUBTASK TRANSITION")
            print(f"  From: {last_subtask}")
            print(f"  To:   {subtask}")
            print(f"  Message history count: {msg_history}")

            if msg_history > 0:
                print(f"  ⚠️  LEAK DETECTED: Message history should be 0 after transition, but is {msg_history}")
            else:
                print(f"  ✓ Clean transition: Message history correctly cleared")
            print()

        last_subtask = subtask

    print(f"{'='*70}\n")
