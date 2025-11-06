# CRITICAL REGRESSION: Lifecycle API Migration Broke All Tool Calls

## Summary

The lifecycle API migration introduced a **critical API mismatch** between `base_agent.py` and all behavior implementations. This causes **100% test failure** in L1-L6 evaluation.

## Root Cause

### The Mismatch

**base_agent.py:1341-1350** calls behaviors with keyword arguments:
```python
result = behavior.dispatch_tool(
    tool_name=tool_name,        # ❌ KEYWORD ARG
    args=args,                  # ❌ KEYWORD ARG
    agent=self,                 # ❌ KEYWORD ARG
    workspace=self.workspace,   # ❌ KEYWORD ARG
    ...
)
```

**All behaviors** define positional parameters:
```python
def dispatch_tool(
    self,
    agent: Any,                 # ✅ POSITIONAL
    tool_name: str,             # ✅ POSITIONAL
    args: dict[str, Any]        # ✅ POSITIONAL
) -> dict[str, Any]:
```

### Why This Fails

When you call `behavior.dispatch_tool(tool_name="write_file", args={...}, agent=self)`:
- Python tries to match `tool_name` to the first positional parameter `agent`
- TypeError: "agent" got an unexpected keyword argument 'tool_name'

## Impact

**ALL 6 tests failed (0% pass rate):**

| Level | Status | Error |
|-------|--------|-------|
| L1 | ❌ FAIL | Unable to invoke write_file tool due to unexpected parameter errors |
| L2 | ❌ FAIL | All tool calls fail due to unexpected 'workspace' parameter |
| L3 | ❌ FAIL | Unable to invoke workspace tools due to unexpected keyword argument errors |
| L4 | ❌ FAIL | Max rounds exceeded (tool calls failing) |
| L5 | ❌ FAIL | Unable to list workspace contents due to tool argument mismatch |
| L6 | ❌ FAIL | Max rounds exceeded (tool calls failing) |

**Time wasted:** 959.7 seconds (16 minutes) of agent looping trying to fix tool calls

**Files created:** 0 (agents couldn't write any files)

## The Fix

Two options:

### Option 1: Fix base_agent.py (RECOMMENDED)

Change `dispatch_tool_to_behavior` to use positional args:

```python
result = behavior.dispatch_tool(
    self,          # agent (positional)
    tool_name,     # tool_name (positional)
    args,          # args (positional)
)
```

Remove all the extra kwargs (`workspace`, `context_manager`, etc.) - behaviors can access these via `agent.workspace`, `agent.context_manager`.

### Option 2: Fix all behaviors

Change every behavior's `dispatch_tool` signature to accept kwargs:

```python
def dispatch_tool(
    self,
    **kwargs
) -> dict[str, Any]:
    agent = kwargs["agent"]
    tool_name = kwargs["tool_name"]
    args = kwargs["args"]
    ...
```

**Option 1 is better** because:
1. Less code churn (1 file vs 10+ files)
2. Cleaner API (behaviors access agent attributes directly)
3. Matches the documented pattern in behaviors/base.py

## Verification Steps

After fix:
1. Run `python test_lifecycle_api_l1_l6.py`
2. Verify all 6 tests pass
3. Check files are created in workspaces
4. Verify tests run correctly

## Timeline

- **2025-11-06 04:01**: Started L1-L6 evaluation
- **2025-11-06 04:17**: All tests failed (16 minutes runtime)
- **2025-11-06 04:18**: Root cause identified (API mismatch)

## Lesson

When changing method signatures:
1. Search for ALL call sites (grep for the method name)
2. Update call sites AND implementations together
3. Add integration tests that actually call the methods
4. Don't rely on unit tests alone (they test methods in isolation)
