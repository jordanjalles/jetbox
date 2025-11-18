# TUI Integration Status

**Last Updated**: 2025-11-18
**Status**: ✅ Working (PlainDisplay only)

---

## Current State

### ✅ What Works

**PlainDisplay (Text Mode)**
- ✅ Integrated into all agents automatically
- ✅ Shows status updates each round: `[agent] Round 1/50 | 0m05s | Running`
- ✅ Shows completion summary with duration and files created
- ✅ Works with task execution (agents with goals)
- ✅ Works with delegation (orchestrator → task_executor)
- ✅ CLI flags work: `--no-tui` forces plain mode

**Example Output**:
```
[task_executor] Round 1/50 | 0m00s | Running
[task_executor] -> write_file(path=hello.py, ...)

[task_executor] Round 2/50 | 0m09s | Running
[task_executor] -> list_dir(path=., depth=0)

======================================================================
✅ TASK COMPLETED SUCCESSFULLY
======================================================================
Duration: 0m 28s

Created hello.py file that prints 'hello world'
======================================================================
```

### ⚠️ Known Limitations

**TextualDisplay (Interactive TUI)**
- ❌ Disabled in auto-detect (async integration issues)
- ❌ Crashes with "No screens on stack" error if forced
- ❌ Needs proper async/await integration with Textual library
- ℹ️ Shows warning if user tries `--tui` flag

**Chatbot Mode**
- ❌ No display updates (uses different code path)
- ℹ️ Chatbot uses `run_single_llm_round()` which doesn't call display lifecycle
- ℹ️ Display is created but never started/updated/stopped
- ℹ️ Not a critical issue - chatbots are conversational, don't need progress bars

### 🔧 Technical Details

**Where Display Works**:
- `AgentLifecycle.run()` - Full lifecycle integration
  - `display.start()` on setup
  - `display.update_status()` each round
  - `display.show_completion()` on completion
  - `display.stop()` on exit

**Where Display Doesn't Work**:
- `AgentLifecycle.run_single_llm_round()` - No lifecycle calls
- `AgentLifecycle.run_task_round_loop()` - No lifecycle calls

**Auto-Detection Logic**:
```python
# tui/display_factory.py:_auto_detect()
# TEMPORARY: Force PlainDisplay until TextualDisplay async is fixed
return PlainDisplay(verbose=verbose)
```

---

## Usage

### Task Execution (Works)

```bash
# Default (auto-detect → plain)
python agent.py --team solo "Create calculator"
# Output: [agent] Round 1/50 | 0m05s | Running

# Force plain (explicit)
python agent.py --team solo --no-tui "Create calculator"
# Output: [agent] Round 1/50 | 0m05s | Running

# Force TUI (shows warning, falls back to plain)
python agent.py --team solo --tui "Create calculator"
# Output: WARNING: TextualDisplay has async integration issues
#         Falling back to PlainDisplay
```

### Chatbot Mode (No Display)

```bash
python agent.py --team chatbot
# No status updates (uses different code path)
```

### Delegation (Works)

```bash
python agent.py --team default "Create Flask app"
# Orchestrator: No display (chatbot mode)
# Task Executor: [task_executor] Round 1/50 | 0m05s | Running
```

---

## What You See Now

### Solo/Default Team with Goal
```
[agent.py] Using team: Solo Executor
[task_executor] Starting run loop (max_rounds=50, model=qwen3:14b)

[task_executor] Round 1/50 | 0m00s | Running   ← NEW: Status update
[task_executor] Round 1/50
[task_executor] -> write_file(path=hello.py, ...)

[task_executor] Round 2/50 | 0m09s | Running   ← NEW: Status update
[task_executor] Round 2/50
[task_executor] -> list_dir(path=., depth=0)

======================================================================
✅ TASK COMPLETED SUCCESSFULLY                  ← NEW: Completion box
======================================================================
Duration: 0m 28s

Created hello.py file
======================================================================
```

### Chatbot Team
```
[agent.py] Using team: Simple Chatbot Team
[simple_chatbot] Chat mode - ask me anything!

You: Hello
simple_chatbot: Hi! How can I help you?
```
(No status updates - uses chat mode, not task mode)

### Default Team (Orchestrator)
```
[orchestrator] Chat mode - ask me anything!
You: Create a Flask app

[orchestrator] Executing 1 tool call(s)
[orchestrator] -> delegate_to_executor(task_description=...)
[delegation] Creating task_executor for task: Create Flask REST API

[task_executor] Round 1/50 | 0m00s | Running   ← NEW: Delegated agent shows status
[task_executor] -> write_file(path=app.py, ...)
```

---

## Future Work

### TextualDisplay Integration (Blocked)

**Problem**: Textual requires async event loop, agent loop is synchronous

**Attempted Fix**: Call `display.start()` synchronously
**Result**: `ScreenStackError: No screens on stack`

**Root Cause**: Textual's reactive properties trigger before app is fully initialized

**Possible Solutions**:
1. Run TextualDisplay in separate thread with queue communication
2. Convert agent loop to async/await (major refactor)
3. Use Textual's `run_worker()` for sync code (experimental)
4. Abandon TextualDisplay, use Rich library instead (simpler, sync-friendly)

**Recommendation**: Option 4 (Rich) - Simpler, better for CLI tools

### Chatbot Display (Low Priority)

**Why It Doesn't Work**: Chatbot uses `run_single_llm_round()` which doesn't call display lifecycle

**Fix**: Add display calls to `run_single_llm_round()`
```python
def run_single_llm_round(self, user_message: str) -> None:
    # Add display.start() before first call
    # Add display.update_status() periodically
    # Add display.stop() on exit
```

**Priority**: Low - Chatbots don't need progress bars

---

## Files Modified

1. **tui/display_factory.py**
   - Disabled TextualDisplay in auto-detect
   - Added warning for `--tui` flag
   - Always returns PlainDisplay for now

2. **base_agent.py**
   - Initialize display in `__init__()` (line 146-149)
   - Added `--tui/--no-tui` CLI flags (line 984-991)

3. **src/agent_lifecycle.py**
   - Added `display.start()` in `_setup_run()` (line 249)
   - Added `display.update_status()` in `_execute_round()` (line 309)
   - Added `display.show_completion()` + `stop()` on all exit paths
   - Added helper methods `_update_display_status()` and `_display_completion()`

4. **config/agents/*.yaml**
   - Removed TUIDisplayBehavior (now direct integration)
   - Added notes about CLI flags

---

## Rollback

If display causes issues:

```python
# base_agent.py:148 - Disable display
display_mode = "plain"  # Remove TUI temporarily
self.display = PlainDisplay(verbose=False)  # Silent mode
```

Or use CLI flag:
```bash
export JETBOX_TUI_MODE=plain
python agent.py "..."
```

---

## Summary

**Working**: PlainDisplay shows clean status updates for task execution
**Broken**: TextualDisplay (async issues), chatbot mode (different code path)
**Recommended**: Use PlainDisplay, it works great for monitoring progress
