# Jetbox TUI System

Centralized, pluggable display system for Jetbox agent output.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 base_agent.py                   │
│  (Uses DisplayInterface, doesn't know details)  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│             DisplayInterface (ABC)              │
│  - update_status()                              │
│  - log_event()                                  │
│  - show_completion()                            │
│  - prompt_user()                                │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────┐         ┌──────────────────┐
│ PlainDisplay  │         │ TextualDisplay   │
│ (print-based) │         │ (interactive TUI)│
└───────────────┘         └──────────────────┘
```

## Usage

### From Agent Code

```python
from tui import DisplayFactory, AgentEvent, EventType

# Create display (auto-detects best mode)
display = DisplayFactory.create()

# Start display
display.start()

try:
    # Update status
    display.update_status(
        goal="Create calculator",
        agent_name="task_executor",
        model="qwen3:14b",
        current_round=5,
        max_rounds=50,
        elapsed_time=30.5,
        status="Running write_file",
        tokens_used=2500,
        tokens_max=128000,
    )

    # Log events
    display.log_event(AgentEvent(
        type=EventType.TOOL_CALL,
        message="write_file(path=calculator.py)",
    ))

    display.log_event(AgentEvent(
        type=EventType.SUCCESS,
        message="Created calculator.py",
        details={"size": "245 bytes"},
    ))

    # Show completion
    display.show_completion(
        success=True,
        summary="Created calculator with tests",
        duration=45.2,
        files_created=["calculator.py", "test_calculator.py"],
    )

finally:
    # Clean up
    display.stop()
```

### Force Specific Display Mode

```python
# Force plain display
display = DisplayFactory.create(force_mode="plain")

# Force TUI
display = DisplayFactory.create(force_mode="textual")
```

### Environment Variables

```bash
# Force plain display
export JETBOX_TUI=plain

# Force TUI
export JETBOX_TUI=textual

# Auto-detect (default)
export JETBOX_TUI=auto
```

### CLI Flags

```bash
# Force plain
python agent.py --no-tui "Create calculator"

# Force TUI
python agent.py --tui "Create calculator"

# Auto-detect (default)
python agent.py "Create calculator"
```

## Display Modes

### PlainDisplay

**When used**:
- Not a TTY (piped/redirected output)
- Terminal too small (<80 cols or <20 lines)
- TERM=dumb
- Textual not installed
- User forces with `--no-tui`

**Features**:
- Simple print() output
- Works everywhere
- No dependencies
- Verbose and minimal modes

**Example output**:
```
[task_executor] Round 5/50 | 0m30s | Running write_file | Tokens: 2500/128000 (2%)
🔧 write_file(path=calculator.py)
✅ Created calculator.py
   size: 245 bytes
```

### TextualDisplay

**When used**:
- TTY detected
- Terminal size >=80x20
- Textual installed
- Auto-detect mode (default)

**Features**:
- Interactive TUI dashboard
- Keyboard controls (p/r/s/c/q)
- Scrollable log history
- Live status updates
- Pause/resume

**Example output**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Jetbox Agent Monitor           ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ 🎯 Goal: Create calculator               ┃
┃ 🤖 Agent: task_executor | Model: qwen3   ┃
┃ ⏱️  Round: 5/50 | Time: 0m30s | Running  ┃
┃ [████████░░░░░░░░░░░░░░░░░░░░] 10%       ┃
┃ 📊 Tokens: 2,500/128,000 (2%)            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ [12:34:56] 🔧 write_file(calculator.py) ┃
┃ [12:34:57] ✅ Created calculator.py      ┃
┃            size: 245 bytes                ┃
┃                                           ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ p Pause │ r Resume │ s Step │ q Quit    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## Interactive Controls (TUI only)

| Key | Action | Description |
|-----|--------|-------------|
| `p` | Pause | Pause agent after current round |
| `r` | Resume | Resume paused agent |
| `s` | Step | Execute one round then pause |
| `c` | Context | Show context inspector (TODO) |
| `q` | Quit | Exit agent |

## Adding New Display Types

1. Create new file: `tui/my_display.py`
2. Implement `DisplayInterface`:
   ```python
   from .display_interface import DisplayInterface, AgentEvent

   class MyDisplay(DisplayInterface):
       def start(self): ...
       def stop(self): ...
       def update_status(self, ...): ...
       def log_event(self, event): ...
       def show_completion(self, ...): ...
       def prompt_user(self, question): ...
   ```
3. Register in factory: `tui/display_factory.py`
4. Test: `python agent.py --display-mode=my_display "test"`

## Rollback Plan

If TUI has critical issues:

### Emergency Disable
```bash
export JETBOX_TUI=plain
```

### Code Rollback
```python
# In display_factory.py, change _auto_detect():
@staticmethod
def _auto_detect(verbose: bool) -> DisplayInterface:
    # TEMPORARY: Force plain display while TUI is buggy
    return PlainDisplay(verbose=verbose)
```

### Remove TUI Entirely
```bash
# Delete TUI module
rm -rf tui/

# Restore old print() calls
git checkout HEAD~1 base_agent.py
```

## Testing

```bash
# Test plain display
JETBOX_TUI=plain python agent.py "Create calculator"

# Test TUI
JETBOX_TUI=textual python agent.py "Create calculator"

# Test auto-detect
python agent.py "Create calculator"

# Test in non-TTY (should use plain)
echo "Create calculator" | python agent.py

# Test pause/resume (TUI only)
python agent.py "Create calculator"
# Press 'p' to pause, 'r' to resume
```

## Files

- `__init__.py` - Public API exports
- `display_interface.py` - Abstract base class (contract)
- `plain_display.py` - Print-based fallback (always works)
- `textual_display.py` - Interactive TUI (main implementation)
- `display_factory.py` - Auto-detection and creation logic
- `README.md` - This file

## Dependencies

- **PlainDisplay**: None (uses stdlib only)
- **TextualDisplay**: `textual>=0.47.0`

Install with:
```bash
pip install textual
```

## Future Enhancements

- [ ] Context inspector panel (press 'c')
- [ ] File tree panel (show workspace files)
- [ ] Export logs to file (press 'e')
- [ ] Search/filter logs (press '/')
- [ ] Theme customization
- [ ] Multi-agent dashboard
- [ ] Performance charts
- [ ] Web-based remote monitoring

---

*Centralized, pluggable, easy to roll back*
