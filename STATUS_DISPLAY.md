# Agent Status Display

Enhanced visibility system for the local coding agent, providing real-time hierarchical progress tracking and performance statistics.

## Overview

The status display system provides a clean, informative view of the agent's progress through tasks, showing:

- **Hierarchical task structure**: Goal → Task → Subtask with visual indicators
- **Real-time progress**: Progress bars for tasks and subtasks
- **Performance metrics**: LLM call timing, token usage, success rates
- **Recent activity**: Latest actions and errors
- **Loop detection warnings**: Alerts when the agent enters infinite loops

## Features

### 1. Hierarchical Progress Tracking

The display shows the complete task hierarchy with clear visual indicators:

```
GOAL: Create a Python calculator package with tests

TASKS (1/3 completed):
  ► ⟳ Create calculator package structure    # Currently active

    SUBTASKS:
        ✓ Write calculator/__init__.py         # Completed
      ► ⟳ Write calculator/advanced.py         # In progress
        ○ Write pyproject.toml                 # Pending

    ○ Write comprehensive tests               # Not started
    ○ Run quality checks                      # Not started
```

**Status Icons:**
- `⟳` - Currently in progress
- `✓` - Completed successfully
- `✗` - Failed
- `⊗` - Blocked (loop detected)
- `○` - Pending (not started)

### 2. Progress Indicators

Visual progress bars show completion status:

```
PROGRESS:
  Tasks:    [█████░░░░░░░░░░░░░░░] 33%
  Subtasks: [████████░░░░░░░░░░░░] 43%
  Success:  92%
```

- **Tasks**: Overall task completion (current task / total tasks)
- **Subtasks**: Subtask completion across all tasks
- **Success**: Action success rate (successful actions / total actions)

### 3. Performance Statistics

Detailed metrics about agent performance:

```
PERFORMANCE:
  Avg LLM call:      2.15s        # Average time per LLM API call
  Avg subtask time:  1m 30s       # Average time to complete a subtask
  LLM calls:         12            # Total number of LLM calls made
  Actions executed:  25            # Total tool actions executed
  Tokens (est):      3,500         # Estimated token usage
  ⚠ Loops detected:  1             # Number of infinite loops caught
```

**Metrics Tracked:**
- **LLM timing**: Individual call times and averages
- **Subtask timing**: Time to complete each subtask
- **Throughput**: Number of LLM calls and actions
- **Token usage**: Estimated based on message count (~100 tokens/message)
- **Loop detection**: Count of detected infinite loops

### 4. Recent Activity Log

Shows the latest actions and errors:

```
RECENT ACTIVITY:
  ✓ write_file
  ✓ run_cmd
  ✗ run_cmd
    └─ Command not allowed: ['bash', '-lc', '...']

  Recent errors:
    • run_cmd rc=1: ruff: command not found
    • File 'pyproject.toml' does not exist
```

Displays:
- Last 3 actions from current subtask (with success/failure icons)
- Error messages (truncated for readability)
- Recent errors from probe state (last 2)

### 5. Compact Status Line

A one-line summary for quick updates:

```
Task 2/3 | Subtask 4/7 | ✓92% | 2m 15s
```

Shows: current task, current subtask, success rate, and runtime.

## Implementation

### Architecture

The status display system consists of three main components:

1. **`status_display.py`**: Core status rendering and statistics tracking
2. **`agent.py`**: Integration hooks in the main agent loop
3. **`context_manager.py`**: Loop detection callback support

### Key Classes

#### `StatusDisplay`

Main class that manages status rendering and statistics.

```python
from status_display import StatusDisplay

# Initialize with context manager
status = StatusDisplay(ctx)

# Record events
status.record_llm_call(duration=2.5, messages_count=10)
status.record_action(success=True)
status.record_subtask_complete(success=True)
status.record_loop()

# Render status
print(status.render(round_no=5))        # Full status display
print(status.render_compact())          # One-line summary
```

#### `PerformanceStats`

Tracks all performance metrics with helper methods.

```python
stats = PerformanceStats()
stats.avg_llm_time()          # Average LLM call time
stats.avg_subtask_time()      # Average subtask completion time
stats.success_rate()          # Overall success rate (0-1)
stats.format_duration(secs)   # Human-readable duration
```

### Integration Points

The status display is integrated into the agent at key points:

1. **Initialization**: Created after context manager setup
2. **Round start**: Full status rendered at the beginning of each round
3. **LLM calls**: Timing recorded for each chat() call
4. **Tool execution**: Success/failure tracked for each action
5. **Subtask completion**: Completion events recorded
6. **Loop detection**: Callback triggered when loops detected
7. **Completion**: Compact status shown when goal achieved

### Data Persistence

Performance statistics are saved to `.agent_context/stats.json` and persist across agent restarts. This allows:

- Resume tracking after crashes
- Historical performance analysis
- Cumulative statistics across sessions

## Usage

### Basic Usage

The status display is automatically enabled when running the agent:

```bash
python agent.py "Create a Python package with tests"
```

Output will include the status display at each round boundary.

### Running the Demo

To see the status display without running a full agent task:

```bash
python test_status_display.py
```

This demonstrates:
- Full hierarchical status display
- Progress bars at different completion levels
- Performance statistics
- Error scenarios
- Compact status line

### Configuration

Status display behavior can be configured by modifying constants in `status_display.py`:

```python
# Progress bar width (characters)
width = 20  # in _render_progress_bar()

# Number of recent actions to show
recent = subtask.actions[-3:]  # in _render_recent_activity()

# Number of recent errors to show
errors = errors[-2:]  # in _render_recent_activity()

# Description truncation length
desc = desc[:60] + "..."  # in _render_hierarchy()
```

## Benefits

### For Development

- **Debugging**: Quickly identify where the agent is stuck
- **Performance tuning**: See average LLM call times and optimize
- **Loop detection**: Immediate visibility into infinite loop issues
- **Progress tracking**: Know exactly how far along the agent is

### For Users

- **Transparency**: Clear view of what the agent is doing
- **Confidence**: Progress bars show the agent is making progress
- **Diagnostics**: Error messages help understand failures
- **Estimation**: Average task times help predict completion

## Example Output

Here's a complete example of the status display during a real agent run:

```
======================================================================
AGENT STATUS - Round 5 | Runtime: 2m 15s
======================================================================

GOAL: Create mathx package with add and multiply functions

TASKS (1/3 completed):
  ► ⟳ Implement core functions

    SUBTASKS:
        ✓ Write mathx/__init__.py with add() function
      ► ⟳ Write mathx/__init__.py with multiply() function
        ○ Write docstrings for all functions

    ✓ Write test suite
    ○ Run quality checks

PROGRESS:
  Tasks:    [██████░░░░░░░░░░░░░░] 33%
  Subtasks: [████████████░░░░░░░░] 60%
  Success:  95%

PERFORMANCE:
  Avg LLM call:      2.35s
  Avg subtask time:  45.2s
  LLM calls:         18
  Actions executed:  32
  Tokens (est):      4,200

RECENT ACTIVITY:
  ✓ write_file
  ✓ run_cmd
  ✓ read_file
======================================================================
```

## Future Enhancements

Potential improvements for the status display:

1. **Color support**: Use terminal colors for better visual hierarchy
2. **Live updates**: Refresh display without scrolling (curses/rich)
3. **Historical graphs**: Show performance trends over time
4. **Export reports**: Generate markdown/HTML summaries
5. **Task time estimates**: Predict completion time based on history
6. **Resource monitoring**: Track CPU/memory usage
7. **Interactive mode**: Allow user to pause/inspect/skip tasks

## Troubleshooting

### Status display not showing

- Check that `status_display.py` is imported in `agent.py`
- Verify `StatusDisplay` is initialized after `ContextManager`
- Ensure `status.render()` is called in the main loop

### Performance stats missing

- Check `.agent_context/stats.json` exists and is writable
- Verify `record_*()` methods are called at integration points
- Look for errors in agent log about failed saves

### Progress bars incorrect

- Verify task/subtask status fields are being updated
- Check `current_task_idx` is advancing correctly
- Ensure subtask completion is calling `mark_subtask_complete()`

### Unicode symbols not displaying

- On Windows, ensure UTF-8 encoding is enabled (see agent.py:16-18)
- Use a terminal that supports Unicode (Windows Terminal, not cmd.exe)
- Fall back to ASCII symbols if needed (modify `_get_status_icon()`)

## Files

- **`status_display.py`** (334 lines): Core status display implementation
- **`test_status_display.py`** (157 lines): Demo and testing script
- **`agent.py`**: Integration points (lines 14, 391-392, 426-427, 434, 454, 486, 495)
- **`context_manager.py`**: Loop callback support (lines 135, 236-237)
- **`.agent_context/stats.json`**: Persisted performance statistics

## Related Documentation

- **[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)**: Overall agent design
- **[CLAUDE.md](CLAUDE.md)**: Development guide and patterns
- **[context_manager.py](context_manager.py)**: Hierarchical context system
