# Loop Detection Gap - Why It Didn't Trigger

## Root Cause Identified ✅

**The TaskExecutorAgent does NOT integrate with ContextManager's loop detection system.**

## Evidence

### 1. Loop Counts Empty

From `.agent_context/state.json`:
```json
"loop_counts": {},
```

Loop counts are empty because TaskExecutorAgent never calls `context_manager.record_action()`.

### 2. No Loop Detection Code in TaskExecutor

Searched `task_executor_agent.py` for loop detection:
```bash
$ grep -n "LoopDetector\|loop_counts\|check_loop\|record_action" task_executor_agent.py
# Result: NO MATCHES
```

TaskExecutorAgent only calls `self.status_display.record_action(success)` for statistics, NOT `self.context_manager.record_action()` for loop detection.

### 3. Loop Detection Only in ContextManager

The LoopDetector class exists in `context_manager.py:611-680`, but is only used by:
- HierarchicalStrategy (the old approach)
- Legacy agent code

**SubAgentStrategy does NOT use it.**

## Architecture Gap

### How Loop Detection SHOULD Work

```python
# In dispatch_tool() after executing a tool:
self.context_manager.record_action(
    name=tool_name,
    args=tool_args,
    result="success" if success else "error",
    error_msg=error_msg,
    result_content=str(tool_result)
)

# ContextManager then:
1. Creates Action object with signature
2. Checks LoopDetector.is_loop()
3. Updates state.loop_counts
4. Returns warning if loop detected
```

### How It ACTUALLY Works (TaskExecutor)

```python
# In dispatch_tool():
tool_result = tool_function(**args)

# Record to status display (for stats only)
self.status_display.record_action(success)

# NO call to context_manager.record_action()
# NO loop detection
# NO loop_counts updates
```

## Why This Wasn't Caught Earlier

1. **Context strategy refactoring**: When SubAgentStrategy was introduced, it inherited from ContextStrategy (base class), not from HierarchicalStrategy which has loop detection integrated.

2. **TaskExecutorAgent is stateless**: TaskExecutor doesn't maintain task/subtask hierarchy like the old agent, so it doesn't need the full ContextManager features. But it DOES need loop detection!

3. **Loop detection is coupled to hierarchy**: The LoopDetector is part of ContextManager, which is designed for hierarchical task decomposition. SubAgentStrategy is flat (no hierarchy), so it never integrated loop detection.

## The L6 Failure Pattern

**What happened in L6:**

```
Round 5-31:
  write_file(path="sitecustomize.py", content=variation1)
  run_bash(command="pytest -q") -> ERROR rc=2

  write_file(path="sitecustomize.py", content=variation2)
  run_bash(command="pytest -q") -> ERROR rc=2

  write_file(path="sitecustomize.py", content=variation3)
  run_bash(command="pytest -q") -> ERROR rc=2

  ... (repeat 15-20 times until timeout)
```

**What loop detection WOULD have caught:**

After 3-5 iterations of `pytest -q` failing with same error:
```
⚠️  LOOP DETECTION WARNING:
You appear to be repeating actions. Observed patterns:
  • Action repeated 5x: run_bash::{"command":"pytest -q"}
  • Result signature matches: ERROR rc=2

Consider:
- Trying a COMPLETELY DIFFERENT approach
- Reading error messages carefully
- Checking if assumptions are wrong
- If task is substantially complete, call mark_complete() anyway
```

**What actually happened:**

No warning. Agent looped until timeout (3min).

## Fix Required

### Option 1: Add Loop Detection to TaskExecutorAgent ⭐ RECOMMENDED

Integrate ContextManager's loop detection into TaskExecutorAgent.dispatch_tool():

```python
# In task_executor_agent.py:dispatch_tool()
def dispatch_tool(self, call: dict[str, Any]) -> dict[str, Any]:
    tool_name = call["function"]["name"]
    args = call["function"]["arguments"]

    # Execute tool
    result = tool_function(**args)

    # Record to status display
    success = not (isinstance(result, dict) and result.get("error"))
    if self.status_display:
        self.status_display.record_action(success)

    # NEW: Record to context_manager for loop detection
    if self.context_manager:
        should_continue = self.context_manager.record_action(
            name=tool_name,
            args=args,
            result="success" if success else "error",
            error_msg=str(result.get("error", "")) if isinstance(result, dict) else "",
            result_content=str(result)
        )

        if not should_continue:
            # Loop detected! Add warning to result
            result["_loop_warning"] = (
                "This action has been attempted multiple times with same results. "
                "Consider trying a different approach or calling mark_complete() "
                "if the core task is done."
            )

    return result
```

**Benefits:**
- Reuses existing LoopDetector logic
- Works with SubAgentStrategy
- Minimal code change

**Drawbacks:**
- TaskExecutorAgent doesn't use task/subtask hierarchy, so context_manager.record_action() might need adjustment

### Option 2: Standalone Loop Detector for TaskExecutor

Create a lightweight loop detector independent of ContextManager:

```python
class SimpleLoopDetector:
    def __init__(self, max_repeats=5):
        self.action_history = []
        self.max_repeats = max_repeats

    def check(self, tool_name, args, result):
        signature = f"{tool_name}::{json.dumps(args, sort_keys=True)}"
        result_sig = f"{signature}::{result[:100]}"

        # Count recent identical action+result pairs
        recent = self.action_history[-20:]  # Last 20 actions
        count = sum(1 for sig in recent if sig == result_sig)

        self.action_history.append(result_sig)

        if count >= self.max_repeats:
            return False, f"Action repeated {count} times with same result"

        return True, None
```

Use in TaskExecutorAgent:

```python
self.loop_detector = SimpleLoopDetector(max_repeats=5)

# In dispatch_tool():
ok, warning = self.loop_detector.check(tool_name, args, str(result))
if not ok:
    result["_loop_warning"] = warning
```

**Benefits:**
- Simpler, decoupled from ContextManager
- Easy to understand and maintain
- Works perfectly with flat (non-hierarchical) strategies

**Drawbacks:**
- Duplicates some logic from ContextManager.LoopDetector
- Needs separate configuration

### Option 3: Make SubAgentStrategy Use LoopDetector

Extend SubAgentStrategy to optionally use ContextManager's loop detection:

```python
class SubAgentStrategy(ContextStrategy):
    def __init__(self, use_loop_detection=True):
        super().__init__()
        self.use_loop_detection = use_loop_detection
        self.max_tokens = 128000
        self.compaction_threshold = 0.75

    def record_action_for_loop_detection(self, context_manager, tool_name, args, result):
        if self.use_loop_detection and context_manager:
            return context_manager.record_action(
                name=tool_name,
                args=args,
                result="success" if not result.get("error") else "error",
                error_msg=str(result.get("error", "")),
                result_content=str(result)
            )
        return True  # No blocking if detection disabled
```

**Benefits:**
- Keeps loop detection as a strategy concern
- Opt-in per strategy

**Drawbacks:**
- Strategies shouldn't know about tool execution (separation of concerns)
- Awkward API (agents would need to call strategy.record_action_for_loop_detection())

## Recommendation

**Option 1** (Add loop detection to TaskExecutorAgent.dispatch_tool) is the best approach:

1. ✅ Reuses proven LoopDetector logic
2. ✅ Works immediately with SubAgentStrategy
3. ✅ Minimal code change (~15 lines)
4. ✅ Centralizes loop detection in one place (agent dispatch layer)

The only adjustment needed is making `context_manager.record_action()` work without requiring a full task/subtask hierarchy (it should gracefully handle flat strategies).

## Testing After Fix

After implementing loop detection:

**L6 REST API test should:**
1. Create APIClient, AuthHandler, tests (rounds 1-4)
2. Run pytest, get error (round 5)
3. Try to fix import (rounds 6-8)
4. **NEW:** Get loop warning after 3-5 failed pytest attempts
5. **NEW:** Either try different approach OR call mark_complete() with existing code
6. Complete within 15 rounds instead of timing out at 31 rounds

**Expected log output:**
```
Round 8:
⚠️  LOOP DETECTION WARNING:
Action repeated 5x: run_bash::{"command":"pytest -q"}
Result: ERROR rc=2

Consider:
- Trying a COMPLETELY DIFFERENT approach
- If core implementation is complete, call mark_complete()
```

---

## Summary

**Why loop detection didn't trigger:**
- TaskExecutorAgent never calls `context_manager.record_action()`
- Loop detection only exists in ContextManager (designed for hierarchical strategies)
- SubAgentStrategy doesn't integrate with loop detection

**Fix:**
- Add loop detection integration to TaskExecutorAgent.dispatch_tool()
- This will catch repeated failures and warn the agent
- Agent can then choose to try different approach or mark complete

**Impact:**
- Should prevent "perfectionism loops" like L6
- Reduces timeouts and wasted LLM calls
- Makes agent more robust to environment issues
