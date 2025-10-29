# Critical Bug Fix: Subtask Context Isolation

## Summary

Fixed a **critical bug** where prior subtask messages were bleeding into subsequent subtask contexts, violating the intended isolation boundary.

## The Bug

### What Was Happening

**Before the fix:**
```python
# Line 352 in task_executor_agent.py (WRONG!)
messages.clear()  # Cleared local list (never used for context!)
```

**The problem:**
1. Local `messages` list was cleared on subtask transitions
2. BUT `build_context()` uses `self.state.messages` (line 160)
3. `self.state.messages` was **NEVER cleared**
4. Result: Subtask 2's context included messages from Subtask 1!

### Code Flow (Buggy)

```
Subtask 1:
  - Accumulates 50 messages in self.state.messages
  - Completes

Subtask 1 → 2 Transition:
  - Line 352: messages.clear()  ← Clears wrong list!
  - self.state.messages still has 50 messages

Subtask 2:
  - build_context() called
  - Line 160: messages=self.state.messages (still has 50!)
  - Hierarchical strategy takes last 24 of those 50
  - Context includes Subtask 1's messages! ❌
```

### Why This Matters

**Context isolation is critical for:**
1. **Preventing confusion**: Subtask 2 shouldn't see Subtask 1's tool calls/results
2. **Memory efficiency**: Old messages should be discarded
3. **Clean slate**: Each subtask starts fresh without prior baggage

## The Fix

### Code Change

```python
# Line 352 in task_executor_agent.py (FIXED!)
old_count = len(self.state.messages)
self.clear_messages()  # Clear self.state.messages (used by build_context)
messages.clear()  # Also clear local list for consistency
print(f"[context_isolation] Cleared {old_count} messages after subtask transition")
```

### What Changed

- **Now clears `self.state.messages`** (the list actually used by build_context)
- Also clears local `messages` for consistency
- Counts from correct list (`self.state.messages`)

### Code Flow (Fixed)

```
Subtask 1:
  - Accumulates 50 messages in self.state.messages
  - Completes

Subtask 1 → 2 Transition:
  - Line 352: self.clear_messages()  ← Clears correct list!
  - self.state.messages now empty

Subtask 2:
  - build_context() called
  - Line 160: messages=self.state.messages (empty!)
  - Hierarchical strategy gets only new Subtask 2 messages
  - Context isolated! ✓
```

## Verification

### Test Created

`/workspace/tests/test_subtask_context_isolation.py`

### Test Results

```
[context_isolation] Cleared 8 messages after subtask transition
[context_isolation] Cleared 8 messages after subtask transition
[context_isolation] Cleared 10 messages after subtask transition

✓ PASS - Context properly isolated between subtasks
```

### What the Test Verifies

1. Patches `build_context()` to capture what's sent to LLM
2. Runs multi-subtask goal
3. Detects subtask transitions (message count drops)
4. Verifies that after transitions, message count is near zero
5. Confirms no leak of prior subtask messages

## Impact

### Before Fix

- ❌ Prior subtask messages leaked into current context
- ❌ Context could contain irrelevant tool calls from earlier subtasks
- ❌ Violated isolation boundary
- ❌ Potential confusion for LLM

### After Fix

- ✓ Clean subtask boundaries
- ✓ Each subtask starts with empty message history
- ✓ Only hierarchical context (goal/task/subtask info) persists
- ✓ Proper isolation maintained

## Why Was This Not Caught Earlier?

### Misleading Behavior

The log message said "Cleared N messages after subtask transition" which LOOKED correct, but:

1. It was counting from the wrong list (local `messages`)
2. The wrong list was being cleared
3. The actual context-building list (`self.state.messages`) was untouched

### The Hierarchical Strategy Masked It

Because hierarchical strategy only takes "last N" messages:
- With 50 accumulated messages, only last 24 sent to LLM
- Looked "bounded" but still included prior subtask messages
- The validation test checked bounding, not isolation

## Related Files

**Fixed:**
- `/workspace/task_executor_agent.py:352` - The critical fix

**Tests:**
- `/workspace/tests/test_subtask_context_isolation.py` - Verifies isolation
- `/workspace/tests/test_context_size_validation.py` - Verifies bounding

**Documentation:**
- `/workspace/analysis/CONTEXT_MANAGEMENT_VALIDATION.md` - Context strategy overview

## Lessons Learned

1. **Verify what's actually used**: The local `messages` list was redundant
2. **Test isolation, not just bounding**: Context can be bounded but still leak
3. **Check the full flow**: Trace from usage point (build_context) back to clearing

## Future Improvements

### Consider Removing Redundant Local List

The local `messages` list (line 253) is never used for context building:
- Could be removed entirely
- Or repurposed for a different tracking mechanism
- Currently just adds confusion

### Add Explicit Isolation Tests

All future context strategies should have:
- Bounding tests (size limits)
- Isolation tests (no cross-boundary leaks)
- Both are necessary for correct behavior

---

**Date:** 2025-10-29
**Bug severity:** Critical (affects all multi-subtask tasks)
**Fix verified:** Yes (test passing)
