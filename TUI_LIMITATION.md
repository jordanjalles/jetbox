# TUI In-Place Update Limitation

## The Problem

True in-place status updates (like a progress bar that stays in one place while work continues) are **fundamentally incompatible** with also printing logs between updates.

### Why It Doesn't Work

When we print:
```
📊 [██░░░] 10% | Round 5/50 | 0m30s | Running   <- Status line
[context_inspector] Saved snapshot...            <- Log 1
[task_executor] Executing tool call...           <- Log 2
[task_executor] -> write_file(...)               <- Log 3
```

And then try to update the status:
- `\r` (carriage return) only goes to start of CURRENT line (Log 3)
- `\033[F` (ANSI move up) moves up 1 line at a time
- We'd need to move up 3 lines, but we don't know how many logs printed!

### What We Tried

1. ❌ Simple `\r` overwrite - only works if nothing prints between updates
2. ❌ ANSI cursor save/restore - cursor position doesn't track through other prints
3. ❌ ANSI move up - requires counting lines, which we can't do reliably
4. ❌ TextualDisplay - requires async, causes "No screens on stack" errors

## Solutions

### Option 1: Suppress Other Logs ✅

Make status bar the ONLY output:
```python
# Disable all other logging, only show status bar
display_mode = "quiet"
```

### Option 2: Accept Append Behavior ✅ (CURRENT)

Status bar prints periodically, showing progress over time:
```
📊 [░░░░░] 2% | Round 1/50 | 0m00s | Running
[task_executor] -> write_file(...)

📊 [█░░░░] 4% | Round 2/50 | 0m10s | Running
[task_executor] -> run_bash(...)
```

This isn't "in place" but shows progress clearly.

### Option 3: Bottom Status Bar (Like htop) ⚠️

Reserve bottom line of terminal for status:
```
... logs scroll here ...
... logs scroll here ...
📊 [████░] 60% | Round 30/50 | 2m15s | Running  <- Fixed position
```

Requires:
- ANSI code to position cursor at bottom
- Terminal size detection
- Saving/restoring cursor position
- Handling terminal resize

### Option 4: Use Rich Library 🎯 (RECOMMENDED)

Replace Textual with Rich (simpler, sync-friendly):
```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("Running agent...", total=50)
    for round in range(50):
        # Work happens
        progress.update(task, advance=1)
```

Rich handles all the ANSI complexity internally and works in sync code.

## Recommendation

**Use Rich library** - it's designed for exactly this use case, works with synchronous code, and handles all the terminal complexity for us.

Alternatively, accept current behavior (status updates append as new lines) which still provides progress visibility.
