# Silent Failure Audit - Critical Issues Found

## Summary

Found **6 critical silent failure patterns** that can cause bugs without clear error messages.

---

## CRITICAL: Event System Doesn't Sort Behaviors (Bug #11)

**File**: `src/agent_events.py`

**Problem**: Only `trigger_round_start()` sorts behaviors by sequence number. All other event triggers use config file order, breaking the sequence number contract.

**Impact**: 
- ToolCallingSyntaxBehavior (seq 950) may run AFTER ContextInspectorBehavior (seq 999)
- Context snapshots may capture state BEFORE tool parsing completes
- Behavior dependencies break silently

**Evidence**:
```python
# Line 121-126: trigger_round_start - SORTS
sorted_behaviors = sorted(
    self.agent._behaviors,
    key=lambda b: getattr(b, 'get_sequence_number', lambda: 0)()
)

# Line 165: trigger_llm_response - NO SORTING
for behavior in self.agent._behaviors:  # ← Config order!
    response = behavior.on_llm_response(agent=self.agent, response=response)

# Lines 185, 221, 238, 254: trigger_tool_call, trigger_round_end, 
# trigger_goal_complete, trigger_timeout - ALL NO SORTING
```

**Fix**: Apply same sorting logic to all event triggers

**Severity**: HIGH - Breaks behavior chain guarantees

---

## CRITICAL: No Return Type Validation in Event Chain (Bug #12 - NEW)

**File**: `src/agent_events.py:168`

**Problem**: `on_llm_response` handlers can return None instead of dict, breaking the chain for subsequent behaviors.

**Evidence**:
```python
# Line 168
response = behavior.on_llm_response(agent=self.agent, response=response)
# No validation! If behavior returns None, next behavior gets None
```

**Impact**:
- If any behavior returns None, chain breaks silently
- Subsequent behaviors receive None instead of response dict
- Tools may fail to parse or execute

**Fix**: Validate return value is dict before continuing chain

**Severity**: HIGH - Can break tool execution chain

---

## MEDIUM: Parameter Validation Errors Not Detected (Bug #8)

**File**: `src/agent_lifecycle.py:368-376`

**Problem**: Tool dispatch returns `{status: "parameter_error"}` for invalid params, but caller doesn't check status - treats validation failures as successful tool calls.

**Evidence**:
```python
# tool_dispatch.py returns:
{
    "status": "parameter_error",
    "message": "Invalid parameters: ...",
    "invalid_params": ["bad_param"]
}

# agent_lifecycle.py line 368-376:
result = self.agent.dispatch_tool(tool_call)
tool_result_str = json.dumps(result)  # Serializes error as if it's success!
# NO check for result.get("status") == "parameter_error"
```

**Impact**:
- LLM sees validation error in tool result but may not understand it
- Agent continues as if tool succeeded
- Loop detection doesn't track parameter errors differently

**Fix**: Check `result.get("status")` and handle parameter errors explicitly

**Severity**: MEDIUM - Confuses LLM, wastes rounds

---

## LOW: Return Values Not Captured (Fixed in Previous Commit)

**Files**: 
- `behaviors/delegation.py:1230` ✅ FIXED
- `behaviors/workspace_task_notes.py:480` ✅ FIXED

**Status**: Fixed in commit 9e46e5d

---

## LOW: Exception Handlers May Swallow Critical Errors

**Pattern**: Broad exception handlers in event system

**Evidence**:
```python
# agent_events.py lines 169, 195, 226, 242, 258
except Exception as e:
    print(f"Behavior {behavior.get_name()} error: {e}")
    # Error printed but execution continues
```

**Impact**:
- Behavior failures don't stop execution
- Can accumulate broken state silently
- Hard to debug cascading failures

**Severity**: LOW - Intentional fault tolerance, but may hide bugs

---

## Silent Failure Patterns Identified

### Pattern 1: Missing Sorting
- **Affected**: trigger_llm_response, trigger_tool_call, trigger_round_end, trigger_goal_complete, trigger_timeout
- **Root cause**: Inconsistent event system design
- **Fix**: Extract sorting logic to helper method, apply to all triggers

### Pattern 2: No Return Type Validation
- **Affected**: All event chains (on_llm_response, on_round_start)
- **Root cause**: Trust-based design
- **Fix**: Add `isinstance(result, expected_type)` checks

### Pattern 3: Status Field Not Checked
- **Affected**: Tool dispatch results, validation errors
- **Root cause**: Inconsistent error signaling
- **Fix**: Explicit status checking or raise exceptions

### Pattern 4: Broad Exception Handling
- **Affected**: All event triggers
- **Root cause**: Fault tolerance design
- **Fix**: Log with stack traces, optionally re-raise

---

## Recommended Fix Priority

### P0 (Immediate - Breaks Tool Execution)
1. **Event System Sorting** - Apply to all triggers (Bug #11)
2. **Return Type Validation** - Validate dict returns (Bug #12)

### P1 (High - User Experience)
3. **Parameter Error Handling** - Check status field (Bug #8)

### P2 (Low - Monitoring)
4. **Enhanced Error Logging** - Add stack traces to broad handlers

---

## Fix Strategy

### Quick Fix (Bug #11 - Sorting)
Extract sorting logic to helper method:

```python
def _sort_behaviors_by_sequence(self) -> list:
    """Sort behaviors by sequence number (lower first)."""
    return sorted(
        self.agent._behaviors,
        key=lambda b: getattr(b, 'get_sequence_number', lambda: 0)()
    )

def trigger_llm_response(self, response: dict[str, Any]) -> dict[str, Any]:
    for behavior in self._sort_behaviors_by_sequence():  # ← FIX
        # ... existing code
```

### Return Validation (Bug #12)
```python
def trigger_llm_response(self, response: dict[str, Any]) -> dict[str, Any]:
    for behavior in self._sort_behaviors_by_sequence():
        try:
            if hasattr(behavior, 'on_llm_response'):
                result = behavior.on_llm_response(agent=self.agent, response=response)
                # Validate return type
                if not isinstance(result, dict):
                    print(f"WARNING: {behavior.get_name()}.on_llm_response() returned {type(result)}, expected dict")
                    continue  # Skip this behavior's result
                response = result
```

### Parameter Error Handling (Bug #8)
```python
result = self.agent.dispatch_tool(tool_call)

# Check for validation error
if result.get("status") == "parameter_error":
    # Log validation failure
    print(f"[{self.agent.name}] Parameter validation failed: {result.get('message')}")
    # Could inject warning into context here
    # Continue to next tool call

# Add tool result to messages
tool_result_str = json.dumps(result)
```

---

## Files to Fix

1. `src/agent_events.py` - Add sorting to all triggers (5 methods)
2. `src/agent_events.py` - Add return type validation (2 methods)
3. `src/agent_lifecycle.py` - Check parameter_error status (1 line)

**Total**: 3 files, ~20 lines of changes

**Time estimate**: 30 minutes to fix all P0/P1 issues
