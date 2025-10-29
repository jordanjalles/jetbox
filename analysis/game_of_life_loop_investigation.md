# Game of Life Loop Investigation Report

**Date:** 2025-10-27
**Issue:** Coding agent got stuck in a loop when executing "update the game of life for mouse drag interaction" task

---

## Summary

The agent got stuck in an **infinite list_dir loop** due to a **workspace mismatch** between the orchestrator's expectation and the actual task executor workspace. The orchestrator delegated a task to modify an existing Game of Life web app, but failed to pass the workspace location containing the original files, causing the agent to work in an empty workspace.

---

## Root Cause Analysis

### Issue 1: Missing Workspace Parameter in Delegation

**Location:** `agent_registry.py:125-166` (delegate_task method)

The `delegate_task()` method signature is:
```python
def delegate_task(
    self,
    from_agent: str,
    to_agent: str,
    task_description: str,
    context: str = "",
) -> dict[str, Any]:
```

**Problem:** The method does **NOT** accept a `workspace` parameter, even though:
1. The orchestrator tool definition includes a `workspace` parameter (`orchestrator_agent.py:72-75`)
2. The orchestrator's system prompt instructs it to use workspace for existing projects (`orchestrator_agent.py:188`)
3. The tool call from orchestrator includes the workspace in arguments

**Evidence from logs:**
- The task was "Add drag-to-draw functionality to the Game of Life web app"
- This implies modification of an **existing** project
- But the agent started in a NEW empty workspace: `/workspace/.agent_workspace/modify-the-existing-game-of-life-web-application-s/`
- The actual Game of Life files were in: `/workspace/.agent_workspace/create-a-simple-web-application-implementing-conwa/`

### Issue 2: Agent Stuck in list_dir Loop

**Location:** `agent_v2.log` lines 21:01:43 - 21:02:08

**Sequence of events:**
1. Agent decomposed goal into tasks including "read_file src/components/Board.jsx"
2. Agent repeatedly called `list_dir` with `path: '.'` for 12 rounds (rounds 1-12)
3. Each call returned an empty directory listing (workspace was empty)
4. Forced decomposition kicked in after 12 rounds (`agent_config.yaml` default)
5. Agent decomposed "read_file src/components/Board.jsx" into child subtasks:
   - "Read src/components/Board.jsx and store its content in variable 'board_content'"
   - "Print the variable 'board_content' to the console"
6. Agent continued calling `list_dir` (rounds 14-18+), never finding the file

**State evidence** (`.agent_context/state.json`):
```json
{
  "goal": {
    "description": "Add drag-to-draw functionality to the Game of Life web app...",
    "tasks": [
      {
        "description": "Add drag-to-draw support to Board component",
        "subtasks": [
          {
            "description": "read_file src/components/Board.jsx",
            "status": "decomposed",
            "rounds_used": 12,  // <- Hit limit
            "child_subtasks": [
              {
                "description": "Read src/components/Board.jsx and store its content...",
                "status": "in_progress",
                "rounds_used": 0
              }
            ]
          }
        ]
      }
    ]
  }
}
```

**Loop detection evidence:**
- `loop_counts` shows `list_dir::{\"path\": \".\"}": 1` (only 1 because it's deduplicated after 3)
- `blocked_actions` includes `list_dir::{\"path\": \".\"}` (blocked after threshold)
- Agent kept trying `list_dir` despite it being blocked/unhelpful

### Issue 3: Agent Failed to Recognize Missing Files

**Problem:** The agent should have recognized that:
1. The workspace is empty (no files exist)
2. The task requires reading existing files (`src/components/Board.jsx`)
3. This is a mismatch → should escalate or fail gracefully

**Instead:** Agent kept searching for files that don't exist, creating a loop.

---

## Why Loop Detection Didn't Stop It

The loop detection system in `context_manager.py` has:
- Deduplication threshold: blocks actions after 3 identical calls
- `list_dir` was indeed blocked (in `blocked_actions`)

**However:**
- The LLM continued generating `list_dir` calls despite the block
- The dispatch mechanism wasn't enforcing the block strictly enough
- Agent didn't have a fallback strategy when blocked actions keep being attempted

---

## Actual Workspace Contents

### Expected workspace (with Game of Life files):
```bash
/workspace/.agent_workspace/create-a-simple-web-application-implementing-conwa/
├── index.html
├── script.js
└── style.css
```

### Actual workspace (empty):
```bash
/workspace/.agent_workspace/modify-the-existing-game-of-life-web-application-s/
└── (empty)
```

The orchestrator should have passed:
```python
workspace="/workspace/.agent_workspace/create-a-simple-web-application-implementing-conwa/"
```

---

## Recommended Fixes

### Fix 1: Add workspace parameter to delegate_task (HIGH PRIORITY)

**File:** `agent_registry.py:125`

**Change:**
```python
def delegate_task(
    self,
    from_agent: str,
    to_agent: str,
    task_description: str,
    context: str = "",
    workspace: str = "",  # ADD THIS
) -> dict[str, Any]:
    """
    Delegate a task from one agent to another.

    Args:
        from_agent: Source agent name
        to_agent: Target agent name
        task_description: Task to delegate
        context: Additional context
        workspace: Optional workspace path for existing projects
    """
    # ... existing checks ...

    # For TaskExecutor, set the goal
    if isinstance(target, TaskExecutorAgent):
        # If workspace specified, update TaskExecutor's workspace
        if workspace:
            workspace_path = Path(workspace)
            if workspace_path.exists():
                target.workspace = workspace_path
                # Also update workspace manager if it exists
                if hasattr(target, 'workspace_manager'):
                    target.workspace_manager.workspace_root = workspace_path

        target.set_goal(task_description)
        return {
            "success": True,
            "message": f"Task delegated to {to_agent}",
            "agent": to_agent,
            "workspace": str(target.workspace),
        }
```

**File:** `orchestrator_main.py:219`

**Change:**
```python
if tool_name == "delegate_to_executor":
    # Delegate to TaskExecutor and run it
    task_description = args.get("task_description", "")
    context = args.get("context", "")
    workspace = args.get("workspace", "")  # Already present

    try:
        # Set up the task
        result = registry.delegate_task(
            from_agent="orchestrator",
            to_agent="task_executor",
            task_description=task_description,
            context=context,
            workspace=workspace,  # ADD THIS
        )
```

### Fix 2: Improve Empty Workspace Detection (MEDIUM PRIORITY)

**File:** `agent.py` (probe_state_generic function)

Add explicit check for empty workspace with file-reading tasks:

```python
def probe_state_generic() -> dict:
    """Probe current filesystem state without goal-specific assumptions."""
    ws_files = list(WORKSPACE_ROOT.glob("*"))

    # Check if workspace is empty but task mentions reading files
    if not ws_files:
        current_subtask = ctx.get_current_subtask()
        if current_subtask and any(keyword in current_subtask.description.lower()
                                   for keyword in ["read", "modify", "update", "edit"]):
            return {
                "workspace_empty": True,
                "warning": "Workspace is empty but task requires reading/modifying files. "
                          "This may indicate a workspace mismatch. Consider escalating.",
                "files_written": [],
                "files_exist": [],
            }

    # ... rest of probe logic ...
```

### Fix 3: Stronger Loop Prevention (MEDIUM PRIORITY)

**File:** `agent.py` (dispatch function around line 313)

After detecting a blocked action, add stronger intervention:

```python
def dispatch(fn_name: str, args: dict, ctx: ContextManager) -> dict:
    """Execute a tool call and track action history."""

    # Check if action is blocked
    action_key = f"{fn_name}::{json.dumps(args, sort_keys=True)}"
    if action_key in ctx.state.blocked_actions:
        # Instead of just returning error, add to failure count
        ctx.add_blocked_action_attempt(action_key)

        # If too many blocked attempts, force escalation
        if ctx.get_blocked_attempt_count(action_key) > 3:
            return {
                "success": False,
                "error": f"Action blocked and repeatedly attempted. Escalating to force decomposition.",
                "force_escalate": True,  # Signal to main loop
            }

        return {
            "success": False,
            "error": f"Action blocked due to repetition (tried {ctx.loop_counts.get(action_key, 0)} times)",
        }
```

### Fix 4: Better System Feedback on Empty Workspace (LOW PRIORITY)

**File:** `agent.py` (build_context function)

Include workspace state in system feedback:

```python
def build_context(...) -> list[dict]:
    # ... existing context building ...

    # Add workspace warning if relevant
    if probe_state.get("workspace_empty"):
        system_feedback += f"\n\n⚠️ WORKSPACE EMPTY: {probe_state['warning']}"

    # ... rest of context ...
```

---

## Testing Recommendations

### Test 1: Workspace Parameter Passthrough
```python
# test_orchestrator_workspace_delegation.py
def test_workspace_parameter_passes_through():
    """Verify workspace parameter reaches TaskExecutor."""
    registry = AgentRegistry(config_path="agents.yaml", workspace=Path.cwd())

    # Create a workspace with files
    test_workspace = Path("/tmp/test_workspace")
    test_workspace.mkdir(exist_ok=True)
    (test_workspace / "test.txt").write_text("hello")

    # Delegate with workspace
    result = registry.delegate_task(
        from_agent="orchestrator",
        to_agent="task_executor",
        task_description="Read test.txt",
        workspace=str(test_workspace),
    )

    # Verify TaskExecutor workspace is set
    executor = registry.get_agent("task_executor")
    assert executor.workspace == test_workspace
    assert (executor.workspace / "test.txt").exists()
```

### Test 2: Empty Workspace Detection
```python
def test_empty_workspace_with_read_task():
    """Verify agent detects empty workspace mismatch."""
    # Set up empty workspace
    # Set goal that requires reading files
    # Run one round
    # Assert warning appears in probe_state
```

### Test 3: Loop Breaking on Blocked Actions
```python
def test_blocked_action_escalation():
    """Verify agent escalates after blocked action attempts."""
    # Mock blocked action
    # Attempt it 4 times
    # Verify force_escalate flag is set
```

---

## Conclusion

**Primary Issue:** Missing workspace parameter in the delegation chain caused the agent to work in the wrong directory.

**Secondary Issue:** Agent's loop detection and escalation strategy didn't handle the "searching for non-existent files" scenario well enough.

**Immediate Action:** Implement Fix 1 (workspace parameter passthrough) to resolve the root cause.

**Follow-up Actions:** Implement Fixes 2-4 to improve resilience and debugging for similar scenarios.

---

## Additional Context

### Relevant Code Locations
- **Orchestrator tool definition:** `orchestrator_agent.py:59-79`
- **Delegation execution:** `orchestrator_main.py:211-226`
- **Registry delegate_task:** `agent_registry.py:125-166`
- **TaskExecutor init:** `task_executor_agent.py:24-44`
- **Loop detection:** `context_manager.py` (ContextManager class)
- **Agent main loop:** `agent.py:1192-1281`

### Related Files Modified by User
- `.agent_context/state.json` - Shows task decomposition and rounds used
- `.agent_context/history.jsonl` - Full action history (too large to read easily)
- `agent_v2.log` - Timestamped execution log showing the loop
- `agent_ledger.log` - Audit trail (not examined in detail)

### Environment Info
- Workspace root: `/workspace`
- Agent workspaces: `/workspace/.agent_workspace/{goal-slug}/`
- Model: `gpt-oss:20b` (based on test files)
- OS: Linux/WSL2
