# Max Rounds Exceeded Bug - Root Cause Analysis

**Date**: 2025-11-01
**Issue**: TaskExecutor creates files successfully but exceeds max rounds without marking goal complete

---

## 🔍 Investigation Summary

### Symptom

All Level 1 evaluation tests showed:
- Files created: ✅ 1/1, 1/2, 3/3, 2/3
- Status: ❌ FAIL - Max rounds exceeded
- Agent called `mark_goal_complete` tool
- But goal never actually completed

### Root Cause Found

**File**: `behaviors/compact_when_near_full.py` **Line**: 264

```python
def dispatch_tool(self, tool_name: str, args: dict, **kwargs):
    if tool_name == "mark_goal_complete":
        # Signal goal completion
        context_manager = kwargs.get('context_manager')
        summary = args.get('summary', 'Goal completed')

        # Mark goal as complete in context manager if available
        if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
            context_manager.state.goal.mark_complete(success=True)  # ← BUG HERE!

        return {
            "success": True,
            "result": f"Goal marked complete: {summary}",
            "summary": summary
        }
```

**The bug**: Calling `goal.mark_complete()` but `Goal` class doesn't have this method!

**Result**: Tool throws exception `'Goal' object has no attribute 'mark_complete'`, which gets caught and wrapped as error response.

---

## 📊 Evidence

### Diagnostic Test Output

```bash
Round 1: write_file (creates hello.py) ✓
Round 2: run_bash (verifies it works) ✓
Round 3: mark_complete (marks subtask complete)
Round 4: mark_goal_complete ← CALLED THIS
Round 5: mark_complete (another subtask)

Status: FAIL - Max rounds exceeded
Files created: 1/1 ✓
```

Agent tried to call `mark_goal_complete` but it failed silently.

### Tool Dispatch Error

```python
Result: {'error': "Tool mark_goal_complete failed: 'Goal' object has no attribute 'mark_complete'"}
```

### Tool Registry

```
mark_goal_complete  -> compact_when_near_full (provides the tool)
mark_complete       -> subagent_context
mark_failed         -> subagent_context
```

When `use_behaviors=True`, behaviors provide tools (not legacy tools.py).

---

## 🔧 Fix Required

### Option 1: Add mark_complete() method to Goal class

```python
# In context_manager.py, Goal class
def mark_complete(self, success: bool = True):
    """Mark goal as complete."""
    self.status = "completed" if success else "failed"
```

### Option 2: Fix CompactWhenNearFullBehavior to set status directly

```python
# In behaviors/compact_when_near_full.py, line 264
# Change from:
context_manager.state.goal.mark_complete(success=True)

# To:
context_manager.state.goal.status = "completed"
context_manager._save_state()
```

### Option 3: Return proper status without touching Goal (recommended)

```python
# In behaviors/compact_when_near_full.py, dispatch_tool
if tool_name == "mark_goal_complete":
    summary = args.get('summary', 'Goal completed')

    # Return goal_complete status (agent run loop will handle exit)
    return {
        "status": "goal_complete",  # ← This is what agent checks for
        "message": "Goal completed!",
        "summary": summary
    }
```

The agent's run loop already checks for `status == "goal_complete"` and exits properly.

---

## 🎯 Why This Wasn't Caught

1. **Behavior tools are new**: CompactWhenNearFullBehavior was created during refactoring
2. **Silent failure**: Exception was caught and wrapped as error, didn't crash
3. **No unit tests**: CompactWhenNearFullBehavior.dispatch_tool not tested
4. **Status display doesn't show errors**: Tool errors not displayed in status output

---

## ✅ Recommended Fix

**Change file**: `behaviors/compact_when_near_full.py`
**Line**: 257-270

```python
def dispatch_tool(
    self,
    tool_name: str,
    args: dict[str, Any],
    **kwargs: Any
) -> dict[str, Any]:
    """Handle tool calls for this behavior."""
    if tool_name == "mark_goal_complete":
        summary = args.get('summary', 'Goal completed')

        # Return goal_complete status to trigger agent exit
        return {
            "status": "goal_complete",
            "message": "Goal completed!",
            "summary": summary
        }

    return super().dispatch_tool(tool_name, args, **kwargs)
```

This matches what legacy `mark_goal_complete()` in tools.py returns.

---

## 🧪 Testing

After fix, re-run diagnostic test:
```bash
python diagnose_completion_issue.py
```

Expected:
- Round 4: mark_goal_complete called
- Agent returns {"status": "success"}
- No "Max rounds exceeded" error

Then re-run full evaluation:
```bash
python safe_test_runner.py run_three_level_eval.py
```

Expected L1 results:
- L1: Simple File - ✅ SUCCESS (was FAIL)
- L2: File with Function - ✅ SUCCESS (was FAIL)
- L3: Multi-File Package - ✅ SUCCESS (was FAIL)
- L4: Package with Dependencies - ⚠️  PARTIAL SUCCESS (complex task)

---

## 📝 Additional Findings

### Why SubAgentContextBehavior's mark_complete works

```python
# In behaviors/subagent_context.py
def dispatch_tool(self, tool_name: str, args: dict, **kwargs):
    if tool_name == "mark_complete":
        return {
            "status": "goal_complete",  # ← Returns correct status
            "message": "Delegated task completed successfully",
            "summary": summary
        }
```

It returns `"status": "goal_complete"` which agent detects and exits.

### Why CompactWhenNearFullBehavior's implementation was wrong

It tried to call methods on context_manager objects instead of just returning status dict.

---

*Analysis Date: 2025-11-01*
*Status: Root cause identified, fix ready to implement*
