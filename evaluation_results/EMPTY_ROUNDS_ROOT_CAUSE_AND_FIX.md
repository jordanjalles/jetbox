# Empty Rounds Root Cause Analysis and Fix

**Date**: 2025-11-02
**Issue**: Recurring empty rounds after major refactor to behavior system
**Status**: ✅ FIXED

## Executive Summary

**Root Cause**: ChatbotBehavior incorrectly activated chat mode for delegated agents, causing them to ask clarifying questions instead of executing with available tools.

**Impact**: Architect and potentially other delegated agents had empty rounds, asking questions like "To design a solid architecture, I'd like to clarify a few details first..." instead of calling tools like `write_architecture_doc`.

**Fix**: Updated ChatbotBehavior to check if SubAgentModeBehavior has set a goal, not just context_manager.state.goal.

## Investigation Timeline

### 1. User Observation

User reported: "this pattern of agents producing empty rounds didn't happen before our major refactor"

Expected context structure:
- sys prompt ✓
- tools ✓
- workspace_task_notes (if enabled) ✓
- delegatable agents blurbs and delegation tool spec ✓

### 2. Reproduction

Created `test_empty_round_reproduction.py` to test architect directly:

**Without goal parameter** (broken):
```
[architect] Round 1/3
[loop_detection] ⚠️  Empty round #1 - LLM did not call any tools
[loop_detection] LLM response: Sure! To design a solid architecture for your blog API, I'd like to clarify a few details first...
```

**With goal parameter** (fixed):
```
[subagent_mode] Goal set: Create architecture for a simple blog API.
[architect] Round 1/3
[architect] Executing 1 tool call(s)
[architect] -> write_architecture_doc
```

### 3. Root Cause Identified

**File**: `behaviors/chatbot.py`

**Problem**: ChatbotBehavior checks `agent.context_manager.state.goal` to determine if agent has a goal:

```python
# OLD CODE (BROKEN)
if hasattr(self.agent, 'context_manager') and self.agent.context_manager:
    if self.agent.context_manager.state.goal:
        # Goal already set - don't provide chat instructions
        return ""

# If no goal found, returns chat mode instructions
```

**Why it failed**:
1. Delegated agents use SubAgentModeBehavior, not context_manager
2. SubAgentModeBehavior stores goal in `self.goal`, not context_manager
3. ChatbotBehavior couldn't see SubAgentModeBehavior's goal
4. ChatbotBehavior incorrectly activated chat mode
5. LLM received chat mode instructions: "ask clarifying questions"
6. LLM asked questions instead of calling tools
7. Result: empty rounds

### 4. The Fix

**Files Modified**: `behaviors/chatbot.py`

Added SubAgentModeBehavior goal check in 3 methods:

**Method 1: `get_tools()` (lines 81-86)**
```python
# Check if SubAgentModeBehavior has set a goal (for delegated agents)
for behavior in self.agent._behaviors:
    if behavior.get_name() in ['subagent_mode', 'subagent_context']:
        if hasattr(behavior, 'goal') and behavior.goal:
            # Goal set by delegation - don't provide chatbot tools
            return []
```

**Method 2: `get_instructions()` (lines 219-224)**
```python
# Check if SubAgentModeBehavior has set a goal (for delegated agents)
for behavior in self.agent._behaviors:
    if behavior.get_name() in ['subagent_mode', 'subagent_context']:
        if hasattr(behavior, 'goal') and behavior.goal:
            # Goal set by delegation - don't provide chat instructions
            return ""
```

**Method 3: `on_agent_start()` (lines 276-282)**
```python
# Check if SubAgentModeBehavior has set a goal (for delegated agents)
if not has_goal and hasattr(agent, '_behaviors'):
    for behavior in agent._behaviors:
        if behavior.get_name() in ['subagent_mode', 'subagent_context']:
            if hasattr(behavior, 'goal') and behavior.goal:
                has_goal = True
                break
```

## Test Results

### Before Fix

```
Actions tracked: 0
Consecutive empty rounds: 3
LLM response: "Sure! To design a solid architecture, I'd like to clarify a few details first..."
```

### After Fix

```
Actions tracked: 3
Consecutive empty rounds: 0
Tools called:
  - write_architecture_doc: success=True
  - write_module_spec: success=True
  - write_module_spec: success=True
```

## Impact Assessment

### Affected Agents

Any agent with BOTH behaviors:
- ✅ **ChatbotBehavior** (enables chat mode)
- ✅ **SubAgentModeBehavior** (enables delegation)

From config files:
- **Architect**: ✅ Affected (has both behaviors)
- **TaskExecutor**: ✅ Affected (has both behaviors)
- **Orchestrator**: ⚠️ Partially affected (has ChatbotBehavior but usually excluded in autonomous mode)

### L7 Evaluation Impact

From L7 quick evaluation logs:

**Before fix**:
```
[architect] Round 1/50
[architect] Executing 1 tool call(s)
[architect] -> write_file  ← Trying to call wrong tool!
[loop_detection] ⚠️  Empty round #1

[architect] Round 2/50
[loop_detection] ⚠️  Empty round #2 - LLM did not call any tools
[loop_detection] LLM response: I'm sorry, but I don't have a tool that allows me to create or modify source code files...
```

**Expected after fix**:
- Architect will call correct tools (write_architecture_doc, write_module_spec)
- No more "asking clarifying questions" in delegated mode
- Significant reduction in empty rounds

## Next Steps

1. ✅ Fix implemented and tested
2. ⏭️ Re-run L7 evaluation to measure improvement
3. ⏭️ Monitor empty round rate in production
4. ⏭️ Consider adding explicit "execution_mode" flag to avoid ambiguity

## Lessons Learned

### Why This Happened

1. **Implicit state detection**: ChatbotBehavior used context_manager to detect goal
2. **Multiple goal storage locations**: context_manager vs SubAgentModeBehavior.goal
3. **No unified goal API**: No single method like `agent.has_goal()`

### Prevention for Future

**Recommendation**: Create unified goal detection method in BaseAgent:

```python
def has_goal(self) -> bool:
    """Check if agent has a goal set (any source)."""
    # Check context_manager
    if hasattr(self, 'context_manager') and self.context_manager:
        if self.context_manager.state.goal:
            return True

    # Check SubAgentModeBehavior
    for behavior in self._behaviors:
        if behavior.get_name() in ['subagent_mode', 'subagent_context']:
            if hasattr(behavior, 'goal') and behavior.goal:
                return True

    return False
```

Then ChatbotBehavior can simply call `self.agent.has_goal()`.

## Conclusion

The empty rounds were caused by ChatbotBehavior not recognizing that delegated agents had goals set by SubAgentModeBehavior. The fix adds SubAgentModeBehavior goal checking to all 3 ChatbotBehavior methods that determine execution mode.

**Expected improvement**: Significant reduction in empty rounds for L6-L7 tasks that delegate to architect or task executor.
