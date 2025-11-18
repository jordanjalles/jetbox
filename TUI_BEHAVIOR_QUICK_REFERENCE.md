# TUIDisplayBehavior Quick Reference

## Import

```python
from behaviors import TUIDisplayBehavior
from behaviors.tui_display_behavior import TUIDisplayBehavior  # Direct import
```

## YAML Configuration

```yaml
behaviors:
  - type: TUIDisplayBehavior
    params:
      display_mode: "auto"  # auto | plain | textual
```

## Display Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `auto` | Auto-detects best mode | **Default** - Let system decide |
| `plain` | Plain text output | CI/CD, pipes, non-TTY |
| `textual` | Interactive TUI | Manual override for TUI |

## Lifecycle Events

| Event | When Called | What It Does |
|-------|-------------|--------------|
| `on_goal_start()` | Goal begins | Creates and starts display |
| `on_round_start()` | Before each LLM call | Updates status bar |
| `on_tool_call()` | After each tool | Logs tool call + result |
| `on_round_end()` | After tools execute | Checks for pause (TUI only) |
| `on_timeout()` | Goal times out | Logs timeout, stops display |
| `on_goal_complete()` | Goal finishes | Shows summary, stops display |

## Event Type Mapping

| Result | Event Type |
|--------|------------|
| `{"success": True}` | `EventType.SUCCESS` |
| `{"error": "..."}` | `EventType.ERROR` |
| `{"success": False}` | `EventType.ERROR` |
| `"Error: ..."` | `EventType.ERROR` |
| Other | `EventType.TOOL_RESULT` |

## Usage Example

```python
from behaviors.tui_display_behavior import TUIDisplayBehavior

# Create behavior
behavior = TUIDisplayBehavior(display_mode="auto")

# Add to agent
agent.add_behavior(behavior)

# Lifecycle (handled automatically by BaseAgent)
behavior.on_goal_start(agent, "Create calculator")
context = behavior.on_round_start(agent, 1, context)
behavior.on_tool_call(agent, "write_file", {...}, {...})
behavior.on_round_end(agent, 1)
behavior.on_goal_complete(agent, success=True, summary="Done!")
```

## Agent Attributes Used

| Attribute | Purpose | Fallback |
|-----------|---------|----------|
| `agent.goal` | Current goal | "Unknown goal" |
| `agent.name` | Agent name | `agent.__class__.__name__` |
| `agent.model` | Model name | "unknown" |
| `agent.max_rounds` | Max rounds | 50 |
| `agent.workspace` | Workspace path | None |
| `agent.workspace_manager.created_files` | Files list | Empty or directory scan |
| `agent.get_context_stats()` | Token usage | None (optional) |

## Command Line Flags

```bash
# Auto-detect mode (default)
python agent.py "Create calculator"

# Force plain text
python agent.py --display-mode plain "Create calculator"
# or
export JETBOX_TUI=plain
python agent.py "Create calculator"

# Force TUI
python agent.py --display-mode textual "Create calculator"
# or
export JETBOX_TUI=textual
python agent.py "Create calculator"
```

## Features

✅ **Multiple display modes** - Auto-detect or force plain/TUI
✅ **Status bar** - Real-time progress, tokens, rounds, elapsed time
✅ **Event logging** - Tool calls, successes, errors, warnings
✅ **Completion summary** - Files created, duration, success/fail
✅ **Pause/resume** - Interactive TUI supports 'p' to pause, 'r' to resume
✅ **Context-aware** - Extracts metrics from agent automatically
✅ **Error handling** - Graceful fallbacks for missing agent attributes
✅ **Security** - Empty Rule of Two properties (utility behavior)

## Rule of Two Properties

```python
rule_of_two_properties = set()  # [] Empty - utility behavior
```

**No security concerns:**
- ❌ No untrusted input processing
- ❌ No sensitive data access
- ❌ No external actions
- ✅ Pure display utility

## Migration from StatusDisplayBehavior

```yaml
# OLD (deprecated)
- type: StatusDisplayBehavior
  params:
    reset_stats: false
    show_hierarchical: true

# NEW (recommended)
- type: TUIDisplayBehavior
  params:
    display_mode: "auto"
```

## Testing

```bash
# Run tests
pytest tests/test_tui_display_behavior.py -v

# Run demo
python test_tui_integration_demo.py --plain

# Verify import
python -c "from behaviors import TUIDisplayBehavior; print('✓ OK')"
```

## Files

| File | Purpose |
|------|---------|
| `behaviors/tui_display_behavior.py` | Implementation (400+ lines) |
| `tests/test_tui_display_behavior.py` | Tests (20 tests, 100% pass) |
| `test_tui_integration_demo.py` | Demo script |
| `example_tui_config.yaml` | Example config |
| `TUI_BEHAVIOR_IMPLEMENTATION.md` | Full documentation |

## TUI Controls (Interactive Mode)

| Key | Action |
|-----|--------|
| `p` | Pause execution |
| `r` | Resume execution |
| `q` | Quit |

## Helper Methods

```python
# Format tool arguments
formatted = behavior._format_args({"path": "test.txt"}, max_length=50)
# Output: "path=test.txt"

# Analyze tool result
event_type, message, details = behavior._analyze_result("tool_name", result)
# Returns: (EventType.SUCCESS, "message", {...})
```

## Example Output (Plain Mode)

```
[test_agent] Round 1/10 | 0m00s | Running | Tokens: 5000/128000 (3%)
🔧 write_file(path=calculator.py, content=...)
✅ Success: File written

======================================================================
✅ TASK COMPLETED SUCCESSFULLY
======================================================================
Duration: 0m 2s

Created calculator with tests and documentation

Files created (3):
  - calculator.py
  - test_calculator.py
  - README.md
======================================================================
```

## See Also

- `/workspace/tui/INTEGRATION_GUIDE.md` - TUI integration guide
- `/workspace/tui/integration_example.py` - Direct TUI usage
- `/workspace/behaviors/base.py` - AgentBehavior base class
- `/workspace/TUI_BEHAVIOR_IMPLEMENTATION.md` - Full docs
