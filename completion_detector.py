"""Heuristic completion detection for agent responses."""
from __future__ import annotations
import re
from typing import Any

COMPLETION_PATTERNS = [
    r'\b(task|subtask|goal|work)\s+(is\s+)?(complete|completed|done|finished|successful)\b',
    r'\b(successfully|correctly)\s+(created|implemented|completed|finished|done)\b',
    r'\bhas\s+been\s+(created|implemented|completed|finished|done|written)\s+successfully\b',
    r'\ball\s+(tests|checks|requirements)\s+(pass|passed|succeed|succeeded)\b',
    r'\b(everything|all)\s+(works?|is\s+working)\s+(correctly|as\s+expected|properly)\b',
    r'\b(works?|working)\s+(correctly|as\s+expected|properly|fine|well)\b',
    r'\bno\s+(errors?|issues?|problems?)\s+(found|detected|encountered)\b',
    r'\b(goal|task|objective|requirement)\s+(achieved|accomplished|fulfilled|completed|done)\b',
    r'\b(achieved|accomplished|fulfilled)\s+(the\s+)?(goal|task|objective|requirement)\b',
    r'\bready\s+(for|to)\s+(use|deploy|test|run)\b',
    r'\byou\s+can\s+now\s+(use|run|test|execute)\b',
    r'\b(the\s+)?(file|script|program|code)\s+is\s+ready\b',
    r'\b(test|tests)\s+(ran|run|executed)\s+successfully\b',
    r'\ball\s+\d+\s+tests?\s+passed\b',
    r'\b(created|wrote|generated)\s+.*\s+successfully\b',
    r'\ball\s+tasks\s+(complete|completed|done|finished)\b',
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in COMPLETION_PATTERNS]

def detect_completion_signal(text: str) -> tuple[bool, list[str]]:
    matches = []
    for pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(match.group(0))
    return len(matches) > 0, matches

def should_nudge_completion(llm_response: str, tool_calls: list[dict[str, Any]]) -> tuple[bool, str]:
    for call in tool_calls:
        if call.get("function", {}).get("name") == "mark_subtask_complete":
            return False, "already_marked_complete"
    has_signal, matches = detect_completion_signal(llm_response)
    if has_signal:
        return True, f"completion_signal_detected: {matches[0][:50]}"
    return False, "no_signal"

def generate_nudge_message(completion_phrases: list[str], current_subtask: str | None = None) -> str:
    example_phrase = completion_phrases[0] if completion_phrases else "task is complete"
    message = f"💡 REMINDER: You mentioned '{example_phrase}'. If the subtask is truly complete, please call mark_subtask_complete(success=True) to advance to the next step."
    if current_subtask:
        message += f"\n\nCurrent subtask: {current_subtask}"
    return message

def analyze_llm_response(response: str, tool_calls: list[dict[str, Any]], current_subtask: str | None = None) -> dict[str, Any]:
    has_signal, matches = detect_completion_signal(response)
    should_nudge, reason = should_nudge_completion(response, tool_calls)
    analysis = {
        "has_completion_signal": has_signal,
        "matched_phrases": matches,
        "should_nudge": should_nudge,
        "nudge_message": None,
        "reason": reason,
    }
    if should_nudge:
        analysis["nudge_message"] = generate_nudge_message(matches, current_subtask)
    return analysis
