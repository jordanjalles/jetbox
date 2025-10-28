# Turn Counter Fix

## Problem

The turn counter logic was causing occasional premature decomposition. The counter was trying to persist round counts across subtask transitions using a dictionary (`subtask_rounds`), which could lead to subtasks starting with non-zero round counts if they had been worked on previously.

## Root Cause

In `agent.py` lines 1680-1683, the code was:

```python
# Reset counter if subtask changed
if current_sig != last_subtask_sig:
    current_subtask_rounds = subtask_rounds.get(current_sig, 0)  # ❌ WRONG
    last_subtask_sig = current_sig
```

This would reload the old round count from the dictionary, meaning if a subtask was visited again, it would continue counting from where it left off instead of starting fresh.

## Solution

Simplified the logic to match the stated behavior:

**Turn counter always increments by one per turn.**
**Turn counter always resets when moving to the next task/subtask.**

### Changes Made

1. **Removed the `subtask_rounds` dictionary** - No longer needed since we don't persist counts
2. **Changed reset logic** to always reset to 0:

```python
# Reset counter if subtask changed
# Turn counter always resets to 0 when moving to next task/subtask
if current_sig != last_subtask_sig:
    current_subtask_rounds = 0  # ✓ CORRECT
    last_subtask_sig = current_sig
```

3. **Removed dictionary writes** at round increment points (removed `subtask_rounds[sig] = ...` lines)
4. **Added clarifying comments** at increment points

### Modified Files

- `agent.py`:
  - Line 1657-1661: Removed `subtask_rounds` dictionary, added clarifying comment
  - Line 1680-1684: Changed reset logic to always reset to 0
  - Line 1989-1992: Removed dictionary write (force escalation case)
  - Line 2053-2062: Removed dictionary write, added comment (first increment point)
  - Line 2085-2093: Removed dictionary write, added comment (second increment point)

## Testing

Created comprehensive tests to verify the fix:

1. **`test_round_counter_fix.py`** - Unit tests for reset behavior
2. **`test_round_tracking.py`** - Tests for hierarchical tracking
3. **`test_turn_counter_integration.py`** - Integration tests simulating real usage

All tests pass ✅

### Test Results

```
✓ Turn counter increments by 1 per turn
✓ Turn counter resets to 0 when moving to next subtask
✓ Turn counter resets to 0 when decomposing into child subtasks
✓ Previous subtask counts remain frozen (not affected by new subtask)
```

## Benefits

1. **Simpler logic** - No dictionary to maintain, just a single integer counter
2. **Predictable behavior** - Always starts at 0, always increments by 1
3. **No premature decomposition** - Subtasks won't trigger escalation from inherited counts
4. **Easier to reason about** - Clear invariant: each subtask gets a fresh count

## Behavior

- Each subtask now gets exactly `MAX_ROUNDS_PER_SUBTASK` (default: 6) rounds before escalation
- Moving between subtasks resets the counter to 0
- Decomposing a subtask into children starts the first child at 0 rounds
- The `rounds_used` field on subtask objects is still tracked and persisted for display/debugging purposes, but the in-memory counter that drives escalation always resets

This matches the intended design: "Turn counter always increments by one per turn. Turn counter always resets when moving to the next task/subtask."
