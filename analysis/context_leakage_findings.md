# Context Leakage Analysis - Findings

## Problem Statement

The hierarchical context management in Jetbox is not working as designed. Information is leaking between sibling subtasks, which should be impossible if context isolation were functioning correctly.

## Root Cause

### The Issue

In `agent.py`, the `messages` list is initialized **once** at the start of the agent session (line 1523):

```python
# Message history (just the conversation, not context info)
messages: list[dict[str, Any]] = []
```

This list is **never cleared** when transitioning between subtasks. It continuously accumulates messages throughout the entire agent session, across all subtask transitions.

### How Context is Built

When `build_hierarchical_context()` is called (context_strategies.py:118), it:

1. Adds system prompt
2. Adds goal/task/subtask info
3. Adds last N messages from the accumulated `messages` list

```python
# Add last N message exchanges
recent = messages[-history_keep * 2:] if len(messages) > history_keep * 2 else messages
context.extend(recent)
```

Even though it only keeps the last N messages (default 12 exchanges = 24 messages), those messages span across multiple subtasks.

## Expected Behavior vs Actual Behavior

### Expected (from design document)

When transitioning from Subtask 1 to Subtask 2, the agent should see:

```
Agent context turn N (after transition):
- System prompt
- Goal: "create hello world python file"
- Current subtask: "run hello world python file to check it works"
- [EMPTY MESSAGE STACK - no history from previous subtask]
```

### Actual (current implementation)

When transitioning from Subtask 1 to Subtask 2, the agent sees:

```
Agent context turn N (after transition):
- System prompt
- Goal: "create hello world python file"
- Current subtask: "run hello world python file to check it works"
- Last 12 messages (including messages from Subtask 1!)
```

## Evidence

The example given in the task description demonstrates this:
- Subtask 1: "Understand project structure"
- Subtask 2: "Implement feature X"

Currently, Subtask 2 can reference the understanding from Subtask 1, which should be **impossible** if the message history were properly cleared.

## Impact

This breaks the hierarchical decomposition strategy because:

1. **Subtasks are not independent**: Later subtasks can "remember" what happened in earlier subtasks
2. **Context accumulation**: The context grows unbounded within the history window
3. **Design violation**: The agent is not forced to maintain shared state through explicit mechanisms (like writing to files)

## Solution

When a subtask transition occurs (via `mark_subtask_complete`), the `messages` list must be cleared:

```python
# In agent.py main loop, after tool execution:
if tool_name == "mark_subtask_complete" and "result" in tool_result:
    # Subtask transition occurred - clear message history
    messages = []
    log("Cleared message history after subtask transition")
```

This ensures that each subtask starts with a clean slate, seeing only:
- System prompt
- Goal description
- Current subtask description
- Empty message history

## Verification

I've created two tools to verify and fix this issue:

1. **context_diagnostic.py**: Logs every context sent to Ollama, tracking:
   - Turn number
   - Current subtask
   - Number of messages in history
   - Detects and reports leakage on subtask transitions

2. **test_context_leakage.py**: Automated test that:
   - Runs agent with a multi-subtask goal
   - Analyzes diagnostic logs
   - Reports whether information leaks between subtasks

## Next Steps

1. ✅ Identify root cause (completed)
2. ✅ Create diagnostic tools (completed)
3. ⏳ Implement fix (clear messages on subtask transition)
4. ⏳ Verify fix with test_context_leakage.py
5. ⏳ Test with real-world multi-subtask goals

## Files Modified

- `agent.py`: Added diagnostic logging import and call
- `context_diagnostic.py`: New diagnostic logging module
- `test_context_leakage.py`: New automated test script
- `analysis/context_leakage_findings.md`: This document

## Technical Details

### Where to Implement the Fix

The fix should be implemented in `agent.py` in the main loop, right after tool execution. Specifically, after detecting that `mark_subtask_complete` was called:

```python
# Around line 1913 in agent.py
# Check if goal completed via mark_subtask_complete
if isinstance(tool_result, dict) and tool_result.get("status") == "goal_complete":
    # ... existing code ...

# NEW CODE: Check if subtask transition occurred
if isinstance(tool_result, dict) and tool_result.get("status") in ["subtask_advanced", "task_advanced"]:
    # Clear message history for clean subtask transition
    messages = []
    log_context_sent(round_no, [], subtask_desc, notes="SUBTASK_TRANSITION_CLEAR")
    log(f"Cleared message history after subtask transition")
```

### Alternative Approaches Considered

1. **Clear in context_manager.mark_subtask_complete**: This wouldn't work because the context manager doesn't have access to the messages list

2. **Clear in build_hierarchical_context**: This wouldn't work because by then it's too late - the messages have already been passed in

3. **Track transitions and clear at start of next round**: This is the correct approach and is what we'll implement

## References

- Task description: `/workspace/tasks_for_claude/tasks_for_claude_new_2.txt`
- Original design: Subtasks should be completely isolated with no shared context
- Current implementation: `agent.py:1523` (messages initialization), `context_strategies.py:118` (context building)
