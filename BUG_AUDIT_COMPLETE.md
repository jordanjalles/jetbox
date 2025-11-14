# Complete Bug Audit - 25 Bugs Found

## Executive Summary

**Total Bugs Found**: 25
**Critical (Execution-Blocking)**: 5
**High (Data Loss/Corruption)**: 6
**Medium (Edge Cases/UX)**: 10
**Low (Minor Issues)**: 4

---

## CRITICAL BUGS (Must Fix Immediately)

### 1. **Loop Detection Warnings Never Shown to Agent**
- **File**: `behaviors/loop_detection.py:243`
- **Impact**: Agent repeats failing actions without feedback
- **Fix**: Capture return value from `inject_user_message_after_system()`
- **Status**: 🔴 BLOCKING L5 success

### 2. **JSON Parser Silently Fails on TypeError**
- **File**: `behaviors/tool_calling_syntax.py:188-196`
- **Impact**: Tool calls lost when arguments have wrong type
- **Fix**: Don't catch TypeError or validate arguments type
- **Status**: 🔴 BLOCKING tool execution

### 3. **Tool Result Serialized as String Instead of Dict**
- **File**: `src/agent_lifecycle.py:371-376`
- **Impact**: LLM receives string instead of structured data
- **Fix**: Remove `json.dumps()` - Ollama handles serialization
- **Status**: 🔴 May break LLM understanding

### 4. **No Arguments Dict Validation in Tool Dispatch**
- **File**: `src/tool_dispatch.py:202-203`
- **Impact**: String arguments passed to tools expecting dict
- **Fix**: Add type validation before dispatch
- **Status**: 🔴 BLOCKING tool execution

### 5. **Cross-Workspace Path Resolution Blocks Delegation**
- **File**: `src/workspace_manager.py:103-125`
- **Impact**: task_executor CANNOT read architect's files
- **Fix**: Skip containment validation for `.agent_workspaces/` paths
- **Status**: 🔴 **CRITICAL** - May explain why files weren't found

---

## HIGH PRIORITY BUGS

### 6. **Missing Arguments Type Check Before Returning**
- **File**: `behaviors/tool_calling_syntax.py:218-224`
- **Impact**: Parser returns string arguments as-is
- **Fix**: Validate `isinstance(parsed.get("arguments"), dict)`

### 7. **Write File Doesn't Verify Content Written Correctly**
- **File**: `behaviors/write_file_tools.py:359-381`
- **Impact**: Partial writes (disk full) reported as success
- **Fix**: Verify file size or read back content

### 8. **Parameter Validation Errors Not Handled by Caller**
- **File**: `src/tool_dispatch.py:91-94` + `agent_lifecycle.py:368-376`
- **Impact**: Validation failures treated as successful tool calls
- **Fix**: Check `result.get("status") == "parameter_error"`

### 9. **TimeBox Behavior Modifies State Instead of Context**
- **File**: `behaviors/time_box.py:156, 170`
- **Impact**: Breaks behavior chain contract, ordering bugs
- **Fix**: Append to context parameter, not `agent.state.messages`

### 10. **Workspace Parent Directory Check Too Strict**
- **File**: `behaviors/delegation.py:1112-1122`
- **Impact**: Prevents deep nested workspaces
- **Fix**: Remove check (Python's `mkdir(parents=True)` handles it)

### 11. **agent_events Doesn't Sort Behaviors in trigger_llm_response()**
- **File**: `src/agent_events.py:165-171`
- **Impact**: ContextInspector may capture BEFORE tool parsing
- **Fix**: Sort by `get_sequence_number()` like `trigger_round_start()`

---

## MEDIUM PRIORITY BUGS

### 12-21. Various edge cases and validation issues
- Brace counter doesn't validate JSON structure
- Regex pattern too restrictive for nested objects
- Escape sequence decoder heuristic can false-negative
- Directory list returns inconsistent error format
- Read file truncation uses wrong path in message
- Bash output truncation splits UTF-8 characters
- Event system doesn't validate return types
- Token estimation misses tool definitions
- write_file uses os.path instead of Path
- Auto-pollution may break user expectations

---

## LOW PRIORITY BUGS

### 22-25. Minor issues
- Context inspector JSON serialization errors
- Empty path not explicitly handled
- BaseAgent bypasses anti-pollution
- Command whitelist empty command handling

---

## Root Cause Patterns

1. **Silent Failure Pattern**: Exceptions caught without logging
2. **Missing Type Validation**: Assumes dicts, doesn't verify
3. **Inconsistent Error Formats**: String vs dict vs list
4. **No Return Type Checking**: Event handlers can break chain
5. **Insufficient Verification**: File writes, JSON parsing
6. **State Mutation**: Behaviors modify global state instead of parameters

---

## Immediate Action Plan

### Phase 1: Fix Execution-Blocking Bugs (Today)

1. ✅ **DONE**: JSON parser with trailing garbage (baa54f7)
2. ✅ **DONE**: Fix loop detection return value capture (439f4f9)
3. ❌ **FALSE POSITIVE**: Cross-workspace path resolution (workspace is shared)
4. ✅ **DONE**: Add arguments dict validation (439f4f9)
5. ❌ **FALSE POSITIVE**: Tool result serialization (design is correct)

### Phase 2: Fix High Priority Bugs (Next)

6. Fix TimeBox behavior state mutation
7. Sort behaviors in trigger_llm_response()
8. Add write file content verification
9. Handle parameter validation errors
10. Remove strict parent directory check

### Phase 3: Run New Evaluation

After fixing Phase 1+2 bugs:
- Re-run L5-L7 evaluation
- Expected: 40-60% L5 success (vs 0% currently)
- Verify files are created and accessible

---

## Why L5 Tasks Failed (Updated Understanding)

**Original theory**: Time nudges too late, reading too much
**After JSON parser fix**: Tools should execute now
**After audit**: Found 4 MORE execution-blocking bugs!

**Complete failure chain**:
1. LLM outputs JSON with trailing XML → Parser fails → Tool not executed ✅ **FIXED**
2. Loop warnings never shown → Agent repeats mistakes → Wasted time ❌ **BUG #1**
3. Cross-workspace paths blocked → Can't read architect files → Missing context ❌ **BUG #5**
4. Arguments validation missing → Wrong types crash tools → Silent failures ❌ **BUG #2, #4**
5. Tool results serialized wrong → LLM can't parse responses → Confusion ❌ **BUG #3**

**All 5 bugs must be fixed for L5 tasks to succeed.**

---

## Estimated Impact of Fixes

| Bug # | Fix | Expected Improvement |
|-------|-----|---------------------|
| 1 (JSON parser) | ✅ Done | Files can be created (was 0% success) |
| 2 (Loop warnings) | Todo | Stop infinite loops (20-30% improvement) |
| 5 (Cross-workspace) | Todo | Read architect files (30-40% improvement) |
| 3 (Tool results) | Todo | LLM understands results (10-20% improvement) |
| 4 (Args validation) | Todo | Tools execute correctly (10-20% improvement) |

**Combined expected**: 70-100% L5 success rate (from 0% currently)

---

## Files to Fix

1. `behaviors/loop_detection.py` - 1 line change
2. `src/workspace_manager.py` - 5 line change
3. `behaviors/tool_calling_syntax.py` - 3 line change
4. `src/tool_dispatch.py` - 3 line change
5. `src/agent_lifecycle.py` - 1 line change

**Total code changes**: ~15 lines across 5 files

**Time estimate**: 30 minutes to fix all critical bugs
