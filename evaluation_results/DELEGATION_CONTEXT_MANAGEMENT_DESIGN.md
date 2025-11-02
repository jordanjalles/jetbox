# Proper Delegation Context Management Design

**Date**: 2025-11-02
**Issue**: Primitive delegation lacks proper context management

## Current (Broken) Flow

```
Orchestrator Round N:
  messages: [sys, user, assistant, user, ...]  (N messages)
  → calls delegate_to_executor tool

DelegationBehavior._delegate_to_agent():
  target_agent = TaskExecutorAgent(workspace, goal)  ← Creates agent
  execution_result = target_agent.run(max_rounds=50) ← Runs it
  return {success, status, execution_result}         ← Returns raw result

Orchestrator Round N+1:
  messages: [sys, user, assistant, ..., tool_result]
  ← Tool result is complex dict, LLM confused about what to do
```

**Problems:**
1. No workspace coordination (subagent creates NEW workspace, orchestrator doesn't know where)
2. Subagent outcome not summarized (raw execution_result too verbose)
3. No jetbox notes loading (subagent's summary not retrieved)
4. Result format confusing (LLM doesn't understand execution_result dict)

## Proper Delegation Flow (User's Requirement)

```
1. Store Delegator State
   - Save orchestrator's current goal/context
   - Note: Already handled by agent.state.messages

2. Reset Ollama Context (NOT NEEDED)
   - Ollama chat API is stateless - each call gets full message list
   - No implicit context pollution

3. Instantiate Subagent with Goal/Context
   - Pass goal description ✅
   - Set workspace mode:
     * If orchestrator has workspace → subagent REUSES it
     * If orchestrator creating new → subagent uses SAME base workspace
   - Coordinate workspace paths ✅ (mostly working)

4. Prepare to Handle Subagent Ending
   - Capture execution result ✅
   - Get workspace location ✅
   - Read jetbox notes (summary) ❌ MISSING

5. Run Subagent
   - Execute target_agent.run(max_rounds) ✅

When Subagent Ends:

6. Get Success/Failure/Summary Outcome
   - Status: success/failure/max_rounds ✅
   - Summary: Read jetboxnotes.md from subagent workspace ❌ MISSING
   - Files created: List actual files ❌ MISSING

7. Reset Ollama Context (NOT NEEDED)
   - Stateless API, no reset required

8. Reload Delegator State + Subagent Outcome
   - Orchestrator continues automatically ✅
   - Tool result should contain:
     * Clear status message
     * Human-readable summary
     * Workspace location
     * Files created
     * Next steps suggestion

9. Run Delegator Again
   - Happens automatically when dispatch_tool returns ✅
```

## Required Fix

Update `behaviors/delegation.py` to:

### 1. Workspace Coordination

```python
# Determine workspace for subagent
if hasattr(calling_agent, 'workspace') and calling_agent.workspace:
    # Subagent works in SAME workspace as orchestrator
    workspace = calling_agent.workspace
else:
    # Create new workspace (fall back to current behavior)
    workspace = None
```

### 2. Capture Subagent Summary

```python
# After subagent completes, read its jetbox notes
subagent_workspace = target_agent.workspace
notes_file = subagent_workspace / "jetboxnotes.md"

if notes_file.exists():
    with open(notes_file) as f:
        summary = f.read().strip()
else:
    summary = f"Task execution {status}. No summary available."
```

### 3. List Files Created

```python
# List files created by subagent
files_created = []
if subagent_workspace and subagent_workspace.exists():
    for item in subagent_workspace.iterdir():
        if item.is_file() and not item.name.startswith('.'):
            files_created.append(item.name)
```

### 4. Return Clear Summary

```python
# Build human-readable result for orchestrator
if success:
    message = f"""Task completed successfully.

Summary:
{summary}

Workspace: {subagent_workspace}
Files created: {', '.join(files_created) if files_created else 'none'}

The task is complete. You can now proceed with next steps."""
else:
    message = f"""Task execution {status}.

Summary:
{summary}

Workspace: {subagent_workspace}

The task did not complete successfully. Consider:
- Breaking down into simpler subtasks
- Providing more specific requirements
- Trying a different approach"""

result = {
    "success": success,
    "status": status,
    "message": message,  # ← Clear, actionable summary
    "workspace": str(subagent_workspace),
    "files_created": files_created,
}
```

## Expected Improvement

**Before:**
- Orchestrator confused by complex execution_result dict
- No knowledge of what subagent actually did
- Doesn't know where files are
- Can't make informed decisions

**After:**
- Clear success/failure message
- Human-readable summary of work done
- Knowledge of workspace and files
- Actionable next steps

## Implementation Plan

1. ✅ Document proper flow (this file)
2. ⏳ Update `behaviors/delegation.py` with workspace coordination
3. ⏳ Add jetbox notes reading
4. ⏳ Add file listing
5. ⏳ Format clear result message
6. ⏳ Test with validation script
7. ⏳ Re-run L5-L7 evaluation

## Files to Modify

- `/workspace/behaviors/delegation.py` (lines 300-333)

## Expected Success Rate Improvement

| Level | Current | After Fix | Rationale |
|-------|---------|-----------|-----------|
| L5 | 22% | **60%+** | Orchestrator understands delegation results |
| L6 | 22% | **40%+** | Multi-agent coordination works |
| L7 | 11% | **25%+** | Complex workflows become viable |
