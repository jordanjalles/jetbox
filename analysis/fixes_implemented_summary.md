# Game of Life Loop Issue - Fixes Implemented

**Date:** 2025-10-27
**Issue:** Agent got stuck in infinite loop during "update the game of life for mouse drag interaction" task

---

## Summary

All 5 recommended fixes from the investigation have been successfully implemented:

1. ✅ **Workspace parameter passthrough** - Fixed delegation to use correct workspace
2. ✅ **Empty workspace detection** - Agent now detects workspace mismatches
3. ✅ **Stronger loop prevention** - Blocked actions trigger automatic escalation
4. ✅ **Workspace warnings** - System feedback includes mismatch warnings
5. ✅ **TaskExecutor messaging** - Agent can send messages back to orchestrator

---

## Detailed Changes

### Fix 1: Workspace Parameter Passthrough

**Files Modified:**
- `agent_registry.py:125-184`
- `orchestrator_main.py:219-225`

**Changes:**
```python
# agent_registry.py
def delegate_task(
    self,
    from_agent: str,
    to_agent: str,
    task_description: str,
    context: str = "",
    workspace: str = "",  # NEW PARAMETER
) -> dict[str, Any]:
    # ... validation code ...
    if isinstance(target, TaskExecutorAgent):
        # If workspace specified, update TaskExecutor's workspace
        if workspace:
            workspace_path = Path(workspace)
            if workspace_path.exists():
                target.workspace = workspace_path
                # Also update workspace manager if it exists
                if hasattr(target, 'workspace_manager') and target.workspace_manager:
                    target.workspace_manager.workspace_root = workspace_path
            else:
                return {
                    "success": False,
                    "message": f"Specified workspace does not exist: {workspace}",
                }
```

**Impact:** Orchestrator can now specify existing workspaces when delegating tasks that modify existing projects.

---

### Fix 2: Empty Workspace Detection

**Files Modified:**
- `agent.py:158-248`

**Changes:**
```python
def probe_state_generic() -> dict[str, Any]:
    state = {
        # ... existing fields ...
        "workspace_empty": False,  # NEW
        "warning": None,           # NEW
    }

    # Check for empty workspace with read/modify task
    ws_files = [f for f in WORKSPACE_ROOT.glob("*") if f.is_file() and not f.name.startswith(".")]
    if not ws_files and ctx:
        current_subtask = ctx.get_current_subtask()
        if current_subtask:
            desc_lower = current_subtask.description.lower()
            read_keywords = ["read", "modify", "update", "edit", "load", "open", "inspect"]
            if any(keyword in desc_lower for keyword in read_keywords):
                state["workspace_empty"] = True
                state["warning"] = (
                    "⚠️ WORKSPACE MISMATCH: Workspace is empty but task requires reading/modifying files. "
                    "This likely indicates the wrong workspace was used. Consider escalating or checking "
                    "if files should exist in a different workspace."
                )
```

**Impact:** Agent now detects when it's trying to read/modify files in an empty workspace.

---

### Fix 3: Stronger Loop Prevention with Escalation

**Files Modified:**
- `context_manager.py:146-547`
- `agent.py:1143-1184`
- `agent.py:1629-1650`

**Changes:**

1. Added blocked attempt tracking to ContextManager:
```python
@dataclass
class ContextState:
    # ... existing fields ...
    blocked_attempt_counts: dict[str, int] = field(default_factory=dict)  # NEW

class ContextManager:
    def add_blocked_attempt(self, action_sig: str) -> int:
        """Track attempt to execute a blocked action."""
        self.state.blocked_attempt_counts[action_sig] = (
            self.state.blocked_attempt_counts.get(action_sig, 0) + 1
        )
        self._save_state()
        return self.state.blocked_attempt_counts[action_sig]

    def get_blocked_attempt_count(self, action_sig: str) -> int:
        """Get number of attempts for a blocked action."""
        return self.state.blocked_attempt_counts.get(action_sig, 0)
```

2. Enhanced dispatch to check blocked actions and force escalation:
```python
def dispatch(call: dict[str, Any]) -> dict[str, Any]:
    # ... setup code ...

    # Check if action is blocked (loop detected)
    if ctx and action_sig in ctx.state.blocked_actions:
        # Track repeated blocked attempts
        attempt_count = ctx.add_blocked_attempt(action_sig)

        # If too many blocked attempts, force escalation
        if attempt_count >= 3:
            log(f"TOOL✖ {name} blocked and repeatedly attempted ({attempt_count}x) - FORCING ESCALATION")
            return {
                "error": f"Action blocked due to repetition (attempted {attempt_count} times after blocking). "
                         f"This indicates a fundamental issue. Please try a COMPLETELY different approach or escalate.",
                "force_escalate": True,
            }
```

3. Main loop handles force_escalate flag:
```python
# Execute tool
try:
    tool_result = dispatch(c)

    # Check if force_escalate flag is set
    if isinstance(tool_result, dict) and tool_result.get("force_escalate"):
        log(f"Force escalation triggered for blocked action")
        # Trigger escalation immediately
        current_task = _ctx._get_current_task()
        if current_task:
            current_subtask = current_task.active_subtask()
            if current_subtask:
                # Force rounds to max to trigger escalation
                current_subtask_rounds = MAX_ROUNDS_PER_SUBTASK
                subtask_rounds[current_sig] = current_subtask_rounds
                current_subtask.rounds_used = current_subtask_rounds
                _ctx._save_state()
```

**Impact:** Agent will automatically escalate after 3 attempts to use a blocked action, preventing infinite loops.

---

### Fix 4: Workspace Warnings in System Feedback

**Files Modified:**
- `agent.py:1125-1139`

**Changes:**
```python
def build_context(ctx: ContextManager, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # ... existing context building ...

    # Add generic filesystem state
    probe = probe_state_generic()

    # Add workspace warning if present (NEW)
    if probe.get("warning"):
        context_info.append("")
        context_info.append(probe["warning"])
        context_info.append("")

    # ... rest of context ...
```

**Impact:** Workspace mismatch warnings are now prominently displayed in the agent's context.

---

### Fix 5: TaskExecutor to Orchestrator Messaging

**Files Modified:**
- `agent.py:155-182` (new function)
- `agent.py:219-223` (call site)
- `orchestrator_main.py:258-315` (message reading)

**Changes:**

1. Added messaging function in agent.py:
```python
def send_message_to_orchestrator(message: str, severity: str = "info") -> None:
    """
    Send a message from TaskExecutor to Orchestrator.

    Messages are written to .agent_context/messages_to_orchestrator.jsonl
    and can be read by the orchestrator to understand issues.

    Args:
        message: Message content
        severity: "info", "warning", or "error"
    """
    from datetime import datetime

    msg_file = Path(".agent_context/messages_to_orchestrator.jsonl")
    msg_file.parent.mkdir(exist_ok=True)

    msg_entry = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity,
        "message": message,
    }

    # Append to file
    with open(msg_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg_entry) + "\n")

    log(f"MESSAGE_TO_ORCHESTRATOR [{severity}]: {message}")
```

2. Call when workspace mismatch detected:
```python
# Send message to orchestrator
send_message_to_orchestrator(
    f"Workspace mismatch detected: Empty workspace but task '{current_subtask.description}' "
    f"requires reading/modifying files. Current workspace: {WORKSPACE_ROOT}",
    severity="warning"
)
```

3. Orchestrator reads messages after task completion:
```python
# Read messages from TaskExecutor if any
messages_from_executor = []
msg_file = Path(".agent_context/messages_to_orchestrator.jsonl")
if msg_file.exists():
    try:
        with open(msg_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    messages_from_executor.append(json.loads(line))
        # Clear the file after reading
        msg_file.unlink()
    except Exception as e:
        print(f"[orchestrator] Warning: Failed to read executor messages: {e}")

# Display messages from executor
if messages_from_executor:
    print("Messages from TaskExecutor:")
    for msg in messages_from_executor:
        severity = msg.get("severity", "info").upper()
        content = msg.get("message", "")
        print(f"  [{severity}] {content}")
```

**Impact:** TaskExecutor can now communicate critical issues back to the Orchestrator, which can relay them to the user.

---

## Testing Recommendations

### Test 1: Workspace Parameter Works
1. Create a workspace with files
2. Use orchestrator to delegate task with workspace parameter
3. Verify TaskExecutor uses the specified workspace

### Test 2: Empty Workspace Detection
1. Delegate task that requires reading files
2. Don't specify workspace (or specify empty one)
3. Verify warning appears in agent context
4. Verify message sent to orchestrator

### Test 3: Loop Escalation
1. Mock a scenario where agent repeatedly calls blocked action
2. Verify escalation triggers after 3rd attempt
3. Verify subtask rounds forced to max

### Test 4: End-to-End Scenario
1. Create Game of Life web app workspace
2. Delegate "update for mouse drag" task WITHOUT workspace parameter
3. Verify agent detects mismatch, sends warning, escalates appropriately
4. Retry WITH workspace parameter
5. Verify agent successfully reads existing files

---

## Files Modified

1. **agent_registry.py** - Added workspace parameter to delegate_task()
2. **orchestrator_main.py** - Pass workspace parameter, read executor messages
3. **context_manager.py** - Added blocked attempt tracking
4. **agent.py** - Multiple improvements:
   - Empty workspace detection in probe_state_generic()
   - Workspace warnings in build_context()
   - Force escalation in dispatch()
   - Message sending function
   - Fixed duplicate import

---

## Migration Notes

### Breaking Changes
None - all changes are backward compatible.

### Configuration Changes
None required.

### State File Changes
The `.agent_context/state.json` file now includes a new field:
- `blocked_attempt_counts`: Dictionary tracking attempts on blocked actions

Old state files will load correctly (field defaults to empty dict).

### New Files Created
- `.agent_context/messages_to_orchestrator.jsonl` - Message queue from TaskExecutor to Orchestrator (created on-demand)

---

## Performance Impact

**Minimal impact expected:**
- Empty workspace check: O(n) where n = number of files in workspace (typically < 100)
- Blocked action check: O(1) dictionary lookup
- Message writing: Single file append per detection (rare event)

---

## Future Enhancements

1. **Workspace Auto-Detection**: Orchestrator could use `list_workspaces` tool to find relevant workspace automatically
2. **Workspace Suggestions**: When mismatch detected, suggest similar workspace names
3. **Interactive Correction**: Allow orchestrator to prompt user for correct workspace
4. **Metrics**: Track how often workspace mismatches occur to improve UX

---

## Conclusion

All fixes have been implemented successfully. The system now:
- ✅ Passes workspace parameters correctly through delegation chain
- ✅ Detects workspace mismatches early
- ✅ Prevents infinite loops through forced escalation
- ✅ Communicates issues clearly in context
- ✅ Sends messages from TaskExecutor to Orchestrator

The original Game of Life loop issue would now be prevented through multiple layers of defense:
1. Orchestrator would pass correct workspace parameter
2. If not, agent would detect empty workspace immediately
3. Agent would show clear warning in context
4. Agent would send message to orchestrator
5. If agent still looped on blocked action, escalation would trigger automatically

**Status: READY FOR TESTING**
