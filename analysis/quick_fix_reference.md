# Quick Fix Reference - Game of Life Loop Issue

## What Was The Problem?

Agent got stuck calling `list_dir` repeatedly in an empty workspace because:
1. Orchestrator didn't pass workspace parameter → TaskExecutor used wrong (empty) workspace
2. Agent couldn't find files it needed to read
3. Loop detection blocked the action but agent kept trying
4. No escalation or feedback to orchestrator

## What We Fixed

### 1. Workspace Parameter (CRITICAL FIX)
**Before:** `delegate_task(from, to, task, context)`
**After:** `delegate_task(from, to, task, context, workspace)`

Orchestrator can now tell TaskExecutor to use an existing workspace.

### 2. Empty Workspace Detection
Agent now checks: "Am I in empty workspace but task needs to read files?"
→ Shows warning
→ Sends message to orchestrator

### 3. Loop Prevention → Auto Escalation
**Before:** Blocked action → agent keeps trying → infinite loop
**After:** Blocked action → track attempts → 3rd attempt → FORCE ESCALATION

### 4. Better Warnings
Workspace mismatch warnings now appear prominently in agent's context.

### 5. TaskExecutor → Orchestrator Messages
TaskExecutor can send messages (info/warning/error) back to orchestrator via `.agent_context/messages_to_orchestrator.jsonl`

## Usage Examples

### Example 1: Modify Existing Project
```python
# Orchestrator delegates with workspace parameter
orchestrator.call_tool({
    "name": "delegate_to_executor",
    "arguments": {
        "task_description": "Add drag interaction to Game of Life",
        "workspace": "/workspace/.agent_workspace/game-of-life/"  # Specify existing workspace
    }
})
```

### Example 2: Agent Detects Mismatch
```
AGENT CONTEXT:
⚠️ WORKSPACE MISMATCH: Workspace is empty but task requires reading/modifying files.
This likely indicates the wrong workspace was used. Consider escalating.
```

### Example 3: Forced Escalation
```
[log] TOOL→ list_dir args={'path': '.'}
[log] TOOL✖ list_dir blocked (loop detected, attempt 1)
[log] TOOL→ list_dir args={'path': '.'}
[log] TOOL✖ list_dir blocked (loop detected, attempt 2)
[log] TOOL→ list_dir args={'path': '.'}
[log] TOOL✖ list_dir blocked and repeatedly attempted (3x) - FORCING ESCALATION
[log] Forced subtask rounds to 12 to trigger escalation
[escalation] Forcing decomposition (depth 1/5)
```

### Example 4: Messages to Orchestrator
```
Messages from TaskExecutor:
  [WARNING] Workspace mismatch detected: Empty workspace but task 'read_file src/components/Board.jsx' requires reading/modifying files. Current workspace: /workspace/.agent_workspace/modify-the-existing-game-of-life-web-application-s
```

## Quick Checklist

When delegating tasks that modify existing projects:

- [ ] Use `list_workspaces` tool to find existing workspace
- [ ] Pass `workspace` parameter in `delegate_to_executor`
- [ ] Check orchestrator messages after task completion
- [ ] Review executor messages for warnings

## Files That Changed

Core files:
- ✏️ `agent_registry.py` - workspace param
- ✏️ `orchestrator_main.py` - pass param + read messages
- ✏️ `context_manager.py` - blocked attempt tracking
- ✏️ `agent.py` - detection + escalation + messaging

## How to Test

**Scenario: Update existing web app**

1. Create initial web app:
   ```bash
   python orchestrator_main.py "Create a simple Game of Life web app"
   ```

2. Note the workspace path from output

3. Update it (WITH workspace):
   ```bash
   python orchestrator_main.py "Add mouse drag interaction to the Game of Life"
   ```
   Orchestrator should use `list_workspaces` and pass correct workspace

4. Verify no loop, no warnings, successful modification

**Scenario: Force empty workspace (test detection)**

1. Manually delegate with empty workspace path
2. Observe warning in agent context
3. Observe message to orchestrator
4. Observe escalation if agent loops

## State File Changes

`.agent_context/state.json` now tracks:
```json
{
  "blocked_actions": ["list_dir::{\"path\": \".\"}"],
  "blocked_attempt_counts": {
    "list_dir::{\"path\": \".\"}": 3
  }
}
```

## Troubleshooting

**Q: Agent still loops?**
A: Check if `blocked_attempt_counts` is incrementing. If yes, escalation should trigger at 3.

**Q: Workspace parameter not working?**
A: Verify `workspace` key is in tool call arguments. Check orchestrator_main.py line 215.

**Q: No messages from executor?**
A: Check if `.agent_context/messages_to_orchestrator.jsonl` was created. Verify send_message_to_orchestrator() was called.

**Q: Warning not showing?**
A: Verify probe_state_generic() detects empty workspace AND subtask has read/modify keywords.

## Next Steps

1. Test with real Game of Life scenario
2. Monitor for any edge cases
3. Consider adding workspace auto-detection
4. Add metrics for workspace mismatch frequency
