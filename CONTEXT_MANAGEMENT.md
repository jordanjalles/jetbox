# Hierarchical Context Management for Crash-Resilient Agents

## Overview

This document describes the hierarchical context management system designed for local agents that crash frequently and need to avoid infinite loops.

## Key Principles

### 1. Hierarchical Structure

The context is organized as a tree, not a flat list:

```
Goal (top-level user request)
├── Task 1 (mid-level decomposition)
│   ├── Subtask 1.1 (concrete action)
│   │   ├── Action 1.1.1 (tool call)
│   │   └── Action 1.1.2
│   └── Subtask 1.2
│       └── Action 1.2.1
├── Task 2
│   └── Subtask 2.1
│       ├── Action 2.1.1
│       └── Action 2.1.2
└── Task 3
    └── ...
```

**Benefits:**
- LLM only sees current branch (Goal → Active Task → Active Subtask → Recent Actions)
- Automatic filtering of irrelevant context
- Natural decomposition mirrors human planning
- Easy to track progress at multiple levels

### 2. Need-to-Know Context

At any moment, the LLM receives:
- **Goal**: The top-level objective
- **Current Task**: The active mid-level task
- **Active Subtask**: What's being worked on right now
- **Recent Actions**: Last 3 actions for this subtask
- **Next Subtask**: What comes after current one
- **Probe State**: Current filesystem/test status
- **Loop Warnings**: If any actions are blocked

**What's excluded:**
- Completed tasks (unless needed for context)
- Future tasks (beyond next one)
- Actions from other subtasks
- Full action history (just last 3)

This keeps context compact (typically <2000 chars vs. 10,000+ for full history).

### 3. Loop Detection

Three types of loops are automatically detected:

#### a) Simple Repetition
Same action signature attempted 3+ times:
```python
write_file(mathx/__init__.py, content="...")  # Attempt 1
write_file(mathx/__init__.py, content="...")  # Attempt 2
write_file(mathx/__init__.py, content="...")  # Attempt 3 ✓ BLOCKED
```

#### b) Alternating Pattern
Two actions alternating (A→B→A→B):
```python
run_cmd(["pytest", "-q"])     # A
write_file("test.py", "...")  # B
run_cmd(["pytest", "-q"])     # A
write_file("test.py", "...")  # B ✓ BLOCKED (pattern detected)
```

#### c) Escalating Failures
Subtask failed 2+ times → mark as blocked → escalate to task level

**When loop detected:**
1. Action is added to `blocked_actions` set
2. Future attempts return immediately with error
3. Loop logged to `.agent_context/loops.json`
4. LLM receives warning in context
5. Agent must try different approach

### 4. Crash Recovery

State is continuously persisted to `.agent_context/state.json`:

```json
{
  "goal": {
    "description": "Create mathx package",
    "tasks": [
      {
        "description": "Create package structure",
        "status": "in_progress",
        "subtasks": [
          {
            "description": "Create mathx/__init__.py",
            "status": "completed",
            "actions": [...]
          },
          {
            "description": "Add add() function",
            "status": "in_progress",
            "actions": [...]
          }
        ]
      }
    ]
  },
  "current_task_idx": 0,
  "current_subtask_idx": 1,
  "loop_counts": {"write_file::...": 2},
  "blocked_actions": ["run_cmd::{\"cmd\":[\"pytest\"]}"]
}
```

**On crash:**
1. Agent restarts
2. Loads `state.json`
3. Reconstructs full hierarchy
4. Resumes at exact subtask where it left off
5. Blocked actions remain blocked (prevents repeat loops)

## File Structure

```
.agent_context/
├── state.json         # Current hierarchical state (Goal→Task→Subtask)
├── history.jsonl      # Append-only action log (one JSON per line)
└── loops.json         # Detected loops with timestamps
```

Old files (still useful for auditing):
```
agent_ledger.log       # Old append-only log (kept for compatibility)
agent.log              # Old runtime log
status.txt             # Old flat status (replaced by state.json)
```

## API Usage

### Basic Workflow

```python
from context_manager import ContextManager, Task, Subtask

# Initialize
ctx = ContextManager()
ctx.load_or_init("Create mathx package with tests")

# Add tasks (or let LLM decompose goal)
task = Task(description="Create package structure")
task.subtasks = [
    Subtask(description="Create mathx/__init__.py"),
    Subtask(description="Add add() function"),
]
task.status = "in_progress"
task.subtasks[0].status = "in_progress"

ctx.state.goal.tasks.append(task)
ctx._save_state()

# Record actions with loop detection
allowed = ctx.record_action(
    name="write_file",
    args={"path": "mathx/__init__.py", "content": "..."},
    result="success"
)

if not allowed:
    print("Action blocked - loop detected!")

# Get compact context for LLM
context = ctx.get_compact_context(max_chars=2000)
# Send context to LLM instead of full message history

# Mark subtask complete and advance
ctx.mark_subtask_complete(success=True)
ctx.advance_to_next_subtask()
```

### Integration with Existing Agent

See `agent_integration.py` for full example. Key changes:

**OLD (agent.py):**
```python
# Flat message list
messages = [system, user, ...last 12 messages...]

# Simple dedup counter
SEEN[(name, norm)] += 1
if SEEN[(name, norm)] > 3:
    skip()

# Manual pruning
messages = _prune_history(messages, keep=12)
```

**NEW (with ContextManager):**
```python
# Hierarchical structure
ctx = ContextManager()
ctx.load_or_init(goal)

# Sophisticated loop detection
if not ctx.record_action(name, args, result):
    return {"error": "Loop detected"}

# Automatic compaction
context = ctx.get_compact_context()  # Only current branch
```

## Loop Detection Examples

### Example 1: Infinite Pytest Retries

**Without loop detection:**
```
Round 1: run pytest → fails
Round 2: run pytest → fails
Round 3: run pytest → fails
Round 4: run pytest → fails
... (continues forever)
```

**With loop detection:**
```
Round 1: run pytest → fails (attempt 1)
Round 2: run pytest → fails (attempt 2)
Round 3: run pytest → ✓ BLOCKED (loop detected)
Context includes: "⚠ Action blocked due to loops. Try different approach."
LLM sees warning and changes strategy (e.g., fix the actual test code)
```

### Example 2: Edit-Test Oscillation

**Without loop detection:**
```
Round 1: edit test.py
Round 2: run pytest → fails
Round 3: edit test.py (same edit)
Round 4: run pytest → fails
Round 5: edit test.py (same edit)
... (oscillates forever)
```

**With loop detection:**
```
Round 1: edit test.py
Round 2: run pytest → fails
Round 3: edit test.py (same edit)
Round 4: run pytest → ✓ BLOCKED (alternating pattern A-B-A-B)
LLM receives warning and must try different edit or approach
```

## Context Compaction Examples

### Full Message History (OLD)
~8000 characters:
```
[System prompt]
[User: "Create mathx"]
[Assistant: "I'll create..."]
[Tool: write_file mathx/__init__.py]
[Tool result: success]
[Assistant: "Now I'll..."]
[Tool: write_file tests/test_mathx.py]
[Tool result: success]
[Assistant: "Let me..."]
[Tool: run_cmd pytest]
[Tool result: error]
[Assistant: "I see..."]
[Tool: edit tests/test_mathx.py]
... (12 more message pairs)
```

### Hierarchical Context (NEW)
~600 characters:
```
GOAL: Create mathx package with add function and tests
Status: in_progress

CURRENT TASK: Add tests
Status: in_progress

ACTIVE SUBTASK: Fix failing pytest
Status: in_progress
Last failure: ImportError: No module named 'mathx'

Recent actions:
  - run_cmd → error
  - write_file → success
  - run_cmd → error

NEXT: Run ruff check

CURRENT STATE:
  pkg_exists: ✓
  tests_exist: ✓
  pytest_ok: ✗
  ruff_ok: ✓
```

**Reduction: 8000 → 600 chars (92% reduction!)**

## Benefits Summary

| Feature | Old Approach | New Approach |
|---------|-------------|--------------|
| **Context Size** | 8K-12K chars | 600-2K chars |
| **Loop Detection** | Simple counter (>3) | Pattern detection (3 types) |
| **Crash Recovery** | Parse ledger log | Load structured JSON |
| **Context Relevance** | All recent messages | Current branch only |
| **Progress Tracking** | Flat status string | Hierarchical tree |
| **Token Efficiency** | ~3K tokens/round | ~200-500 tokens/round |

## Testing

Run the test suite:
```bash
python test_context_manager.py
```

Tests cover:
1. Basic workflow (Goal → Task → Subtask → Action)
2. Loop detection (simple, alternating, escalating)
3. Crash recovery (save → crash → reload)
4. Hierarchical focus (filtering irrelevant tasks)
5. Probe state integration (filesystem checks)

All tests pass with clear output showing:
- ✓ Features working correctly
- Context examples at each stage
- Loop detection in action

## Migration Path

To integrate with existing `agent.py`:

1. **Phase 1**: Add parallel tracking
   - Keep existing message history
   - Add ContextManager alongside
   - Compare outputs

2. **Phase 2**: Replace loop detection
   - Use `ctx.record_action()` instead of `SEEN` dict
   - Keep old approach as fallback

3. **Phase 3**: Replace context generation
   - Use `ctx.get_compact_context()` instead of message pruning
   - Inject hierarchical context as assistant message

4. **Phase 4**: Add task decomposition
   - At goal start, decompose into tasks/subtasks
   - Can use LLM or heuristics

5. **Phase 5**: Full integration
   - Remove old message history approach
   - Use hierarchical context exclusively

See `agent_integration.py` for detailed migration example.

## Future Enhancements

Possible improvements:

1. **Automatic task decomposition**: Use LLM to break goal into tasks
2. **Dynamic re-planning**: If blocked, ask LLM to replan task hierarchy
3. **Parallel subtasks**: Execute independent subtasks concurrently
4. **Context summarization**: Compress old completed tasks into summaries
5. **Loop pattern learning**: Learn common loop patterns and avoid proactively
6. **Time-based compaction**: Auto-compact context older than N minutes
7. **Dependency tracking**: Model dependencies between subtasks

## Conclusion

The hierarchical context manager provides:
- **Crash resilience** through structured state files
- **Loop prevention** through pattern detection
- **Token efficiency** through need-to-know filtering
- **Clear progress tracking** through hierarchical decomposition

This is essential for local agents that crash frequently and operate with limited context windows.
