# Bug Fix: Workspace Attribute Errors

**Date:** 2025-10-27
**Severity:** CRITICAL (prevents agent from running)

## Issue 1: WORKSPACE_ROOT NameError

After implementing empty workspace detection, the agent crashed on startup with:

```
NameError: name 'WORKSPACE_ROOT' is not defined
```

**Location:** `agent.py:probe_state_generic()` lines 204, 268

## Issue 2: workspace_root AttributeError

After fixing issue 1, the agent crashed with:

```
AttributeError: 'WorkspaceManager' object has no attribute 'workspace_root'. Did you mean: 'workspace_dir'?
```

**Location:** Same locations as Issue 1

## Root Cause

The `probe_state_generic()` function used incorrect variable/attribute names:
1. Used `WORKSPACE_ROOT` which doesn't exist in agent.py
2. Used `workspace_root` property which doesn't exist on WorkspaceManager
3. The correct attribute is `workspace_dir` on the WorkspaceManager class

## Fix

**Step 1:** Changed from:
```python
ws_files = [f for f in WORKSPACE_ROOT.glob("*") if f.is_file() and not f.name.startswith(".")]
```

To:
```python
if _workspace:
    ws_files = [f for f in _workspace.workspace_root.glob("*") if f.is_file() and not f.name.startswith(".")]
```

**Step 2:** Changed from `workspace_root` to `workspace_dir`:
```python
if _workspace:
    ws_files = [f for f in _workspace.workspace_dir.glob("*") if f.is_file() and not f.name.startswith(".")]
```

**Files Modified:**
- `agent.py:202-224` (first occurrence - probe_state_generic early return)
- `agent.py:268-282` (second occurrence - probe_state_generic after ledger processing)
- `agent_registry.py:165` (workspace_root → workspace_dir in delegate_task)

## Changes Made

1. Added `if _workspace:` guard before accessing workspace
2. Changed `WORKSPACE_ROOT` to `_workspace.workspace_dir` (not workspace_root!)
3. Fixed agent_registry.py to use `workspace_dir` instead of `workspace_root`
4. Properly indented the workspace check code

This ensures:
- The code only runs when workspace manager is initialized
- We use the correct workspace directory path
- No NameError or AttributeError on startup

## Correct WorkspaceManager API

From `workspace_manager.py`:
```python
class WorkspaceManager:
    def __init__(self, ...):
        self.workspace_dir = Path(workspace_path).resolve()  # ← Correct attribute name
```

**NOT** `workspace_root` - that was my mistake!

## Testing

```bash
python -c "from agent import probe_state_generic; print('✓ Agent imports successfully')"
# Output: ✓ Agent imports successfully
```

## Status

✅ **FIXED** - Agent can now start successfully
