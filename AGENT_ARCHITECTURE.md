# Agent Architecture Documentation

## Overview

This is a local-first, crash-resilient coding agent that uses hierarchical context management to complete tasks autonomously. The agent runs with Ollama on Windows and is designed to avoid infinite loops through explicit task completion signaling.

## Core Design Principles

### 1. Hierarchical Task Structure
```
Goal (user request)
 └─ Task (high-level objective)
     └─ Subtask (concrete action)
         └─ Action (individual tool call)
```

### 2. Automatic Goal Decomposition
- LLM breaks down user goals into concrete, actionable tasks
- Each subtask must require using a tool (no abstract decision-making)
- Typically 1-3 tasks per goal

### 3. Explicit Completion Signaling
- LLM calls `mark_subtask_complete(success=True)` when done
- Agent automatically advances to next subtask/task
- Clean exit when all tasks complete

### 4. Generic State Probing
- Reports filesystem facts (files created, errors) not goal-specific checks
- No hardcoded expectations (like "pytest must pass")
- Lets LLM interpret state relative to current goal

### 5. Focused Context
- System prompt (constant instructions)
- Current goal/task/subtask (from ContextManager)
- Last 5 message exchanges (recent conversation)
- Generic filesystem state (what files exist)

## System Components

### agent.py (Main Agent Loop)

**Purpose**: Orchestrates the agent execution cycle

**Key Functions**:
- `decompose_goal(goal)` - LLM breaks goal into tasks/subtasks
- `probe_state_generic()` - Check filesystem without goal assumptions
- `build_context()` - Assemble context for LLM
- `dispatch(call)` - Execute tool calls
- `main()` - Main agent loop

**Loop Flow**:
```
1. Initialize ContextManager with goal
2. Decompose goal into tasks (if new)
3. For each round (up to MAX_ROUNDS):
   a. Update probe state (filesystem facts)
   b. Check if all tasks complete → exit
   c. Build context (system + current task + last 5 messages)
   d. Call LLM with tools
   e. If tool calls → execute and record
   f. If mark_subtask_complete → advance task
   g. If goal_complete → exit
   h. If no tool calls → final answer, exit
4. If max rounds hit → stop with status
```

### context_manager.py (State Management)

**Purpose**: Maintains hierarchical task state and detects loops

**Key Classes**:
- `Action` - Single tool call with result
- `Subtask` - Concrete task with list of actions
- `Task` - High-level objective with subtasks
- `Goal` - Top-level user request with tasks
- `ContextState` - Full state including loop detection
- `ContextManager` - Main interface for state operations

**Key Functions**:
- `load_or_init(goal)` - Load existing or create new goal
- `record_action(name, args, result)` - Track action and detect loops
- `mark_subtask_complete(success, reason)` - Mark subtask done/failed
- `advance_to_next_subtask()` - Move to next subtask
- `get_compact_context()` - Generate hierarchical summary

**Loop Detection**:
- Tracks action signatures (name + args)
- Blocks after 3 identical attempts
- Detects alternating patterns (A-B-A-B)
- Stores blocked actions to prevent retry

**State Persistence**:
- `.agent_context/state.json` - Full hierarchical state
- `.agent_context/history.jsonl` - Action log
- `.agent_context/loops.json` - Detected loops

### Tools Available

1. **list_dir(path)** - List files in directory
2. **read_file(path, max_bytes)** - Read text file
3. **write_file(path, content, create_dirs)** - Write/overwrite file
4. **run_cmd(cmd, timeout_sec)** - Run whitelisted command (python, pytest, ruff, pip)
5. **mark_subtask_complete(success, reason)** - Signal task completion

## How Task Completion Works

### Problem We Solved
**Before**: Agent would loop forever because it never knew when it was "done"
- Hardcoded completion checks (e.g., "pytest passes")
- Mismatched context (showing mathx state for hello world goal)
- No explicit completion signal

**After**: Agent explicitly signals completion
- LLM calls `mark_subtask_complete()` when subtask is done
- ContextManager advances to next subtask automatically
- Clean exit when all tasks complete

### Completion Flow

```
Round 1: LLM uses write_file to create script
         → Tool result: "Wrote 22 chars to hello.py"

Round 2: LLM sees file was created successfully
         → Calls mark_subtask_complete(success=True)
         → Agent advances to next subtask or task

Round N: All subtasks complete
         → mark_subtask_complete returns {"status": "goal_complete"}
         → Agent exits with success message
```

## Context Structure

### What LLM Sees Each Round

```
[System Prompt]
You are a local coding agent...
- Call mark_subtask_complete(success=True) when done
- Use tools to accomplish tasks
...

[Current Context]
GOAL: Write a hello world script
CURRENT TASK: Create hello_world.py script
ACTIVE SUBTASK: Write hello_world.py with print statement
FILES CREATED: (none yet)

[Recent Conversation - Last 5 exchanges]
User: (goal context)
Assistant: I'll create the file
Tool(write_file): Wrote 22 chars to hello_world.py
Assistant: File created successfully
...
```

### What's NOT in Context (By Design)

- Full message history (only last 5 exchanges)
- Hardcoded completion criteria (no "pytest must pass")
- Goal-specific state checks (no "mathx package exists")
- All previous actions (only recent in current subtask)

This keeps context:
- **Focused**: Only what's relevant to current subtask
- **Flexible**: Works for any goal without hardcoding
- **Compact**: Reduces token usage and confusion

## Task Decomposition

### Decomposition Prompt Guidelines

The LLM is instructed to create:
- **Concrete, actionable** subtasks (not abstract decisions)
- **Tool-requiring** tasks (must use write_file, run_cmd, etc.)
- **Specific filenames** and actions
- **Simple structure** (usually 1-3 tasks total)

### Good vs Bad Decomposition

**Bad** (abstract, no tool use):
```json
[
  {"description": "Choose a language", "subtasks": ["Decide Python or JavaScript"]},
  {"description": "Plan structure", "subtasks": ["Think about design"]}
]
```

**Good** (concrete, actionable):
```json
[
  {"description": "Create game script", "subtasks": ["write_file rps_game.py with game logic"]},
  {"description": "Test the game", "subtasks": ["run_cmd python rps_game.py"]}
]
```

## Generic State Probing

### Philosophy
Don't assume what the goal needs - just report facts about what happened.

### What We Probe

```python
{
  "files_written": ["hello.py", "test.py"],     # From ledger
  "files_exist": ["hello.py"],                   # Verified on disk
  "files_missing": ["test.py"],                  # Was written but now gone
  "commands_run": ["python hello.py -> rc=0"],   # Recent commands
  "recent_errors": []                            # Last 3 errors
}
```

### What We DON'T Probe
- Whether pytest passes (unless goal mentions tests)
- Whether specific package structure exists
- Code quality metrics
- Any goal-specific assumptions

The LLM interprets these facts relative to the current subtask.

## Loop Protection

### How Loops Are Detected

1. **Action Signature**: `tool_name + json.dumps(args, sort_keys=True)`
2. **Repeat Count**: Same signature attempted 3+ times → blocked
3. **Pattern Detection**: A-B-A-B alternating → blocked
4. **Recent Window**: Last 10 actions checked for patterns

### What Happens When Loop Detected

1. Action is blocked (not executed)
2. Signature added to `blocked_actions` set
3. Loop logged to `.agent_context/loops.json`
4. LLM sees reduced context (doesn't retry blocked actions)

### Why This Matters
Prevents infinite loops like:
- Rewriting same file repeatedly
- Running same failing command over and over
- Alternating between two failed approaches

## Configuration

### Environment Variables
- `OLLAMA_MODEL` - Model to use (default: "gpt-oss:20b")

### Constants (agent.py)
- `MODEL` - Ollama model tag
- `TEMP` - Temperature (0.2 for focused outputs)
- `MAX_ROUNDS` - Hard cap on iterations (24)
- `HISTORY_KEEP` - Message exchanges to retain (5)
- `SAFE_BIN` - Whitelisted commands: `{"python", "pytest", "ruff", "pip"}`

### File Locations
- `agent.log` - Runtime log (agent_v2.log currently)
- `agent_ledger.log` - Action trace (WRITE/CMD/ERROR/TRIED)
- `.agent_context/state.json` - Full state (crash recovery)
- `.agent_context/history.jsonl` - Action history
- `.agent_context/loops.json` - Detected loops

## Example Walkthrough: "Write Hello World"

### Round 0: Initialization
```
- User: "Write a hello world script"
- Agent: Decompose goal via LLM
- Result: [{"description": "Create hello_world.py", "subtasks": ["write_file hello_world.py with print statement"]}]
- State: Task 0, Subtask 0 marked as "in_progress"
```

### Round 1: Create File
```
Context sent to LLM:
  GOAL: Write a hello world script
  ACTIVE SUBTASK: write_file hello_world.py with print statement
  FILES CREATED: (none)

LLM response:
  Tool call: write_file(path="hello_world.py", content="print('Hello, World!')")

Agent executes:
  - Writes file
  - Records action in ContextManager
  - Returns: {"result": "Wrote 22 chars to hello_world.py"}
```

### Round 2: Mark Complete
```
Context sent to LLM:
  GOAL: Write a hello world script
  ACTIVE SUBTASK: write_file hello_world.py with print statement
  FILES CREATED: hello_world.py
  Recent: write_file → success

LLM response:
  Tool call: mark_subtask_complete(success=True)

Agent executes:
  - Marks subtask as "completed"
  - Tries to advance to next subtask
  - No more subtasks → marks task as "completed"
  - No more tasks → returns {"status": "goal_complete"}
  - Agent exits with "Goal achieved" message
```

Total: 2 rounds, clean exit

## Crash Recovery

### How It Works
1. All state persisted to `.agent_context/state.json` after each action
2. On restart, `ContextManager.load_or_init()` checks for existing state
3. If found and goal matches, resume from last known state
4. If not found or goal changed, start fresh

### What Gets Restored
- Full task/subtask hierarchy
- Current task/subtask indices
- All completed actions
- Blocked actions (loop detection state)
- Last probe state

### Idempotency
Designed to handle interruption at any point:
- File operations create parent dirs automatically
- Tool calls are logged before and after execution
- State saved after every tool execution
- No database dependencies (all plaintext)

## Debugging

### Check Current State
```bash
cat .agent_context/state.json
```

### Check Action History
```bash
cat .agent_context/history.jsonl | tail -20
```

### Check Detected Loops
```bash
cat .agent_context/loops.json
```

### Check Runtime Log
```bash
cat agent.log | grep "TOOL"
```

### Check Ledger Summary
```bash
cat agent_ledger.log | tail -50
```

## Common Issues & Solutions

### Issue: Agent loops forever
**Cause**: LLM not calling `mark_subtask_complete()`
**Solution**: Check system prompt includes completion instructions

### Issue: All tools return errors
**Cause**: Dispatch function error (check traceback in log)
**Solution**: Review `agent.log` for stack traces

### Issue: Task decomposition creates abstract tasks
**Cause**: Decomposition prompt not specific enough
**Solution**: Update `decompose_goal()` examples to be more concrete

### Issue: Agent exits too early
**Cause**: LLM calling `mark_subtask_complete()` prematurely
**Solution**: Make subtask descriptions more specific about what "done" means

### Issue: Context manager crashes
**Cause**: Unhashable action signature (slice objects, etc.)
**Solution**: Wrap `record_action()` in try/except (already done)

## Future Improvements

### Potential Enhancements
1. **Streaming output** - Show LLM reasoning in real-time
2. **Better error recovery** - Retry with different approach when blocked
3. **Task templates** - Pre-defined decompositions for common patterns
4. **Multi-file operations** - Batch file creation/editing
5. **Better testing** - Automatic test generation and execution
6. **Cost tracking** - Monitor token usage and LLM calls
7. **Parallel task execution** - Handle independent subtasks concurrently

### Known Limitations
1. Windows-only (due to command whitelist)
2. Local Ollama only (no cloud APIs)
3. No interactive input handling (games that need stdin)
4. Limited to 24 rounds (configurable but hardcoded)
5. No git operations beyond basic commands
6. No network operations (no curl, wget, etc.)

## Architecture Decisions

### Why Hierarchical Context?
- **Focus**: LLM only sees what's relevant to current subtask
- **Scalability**: Can handle complex multi-task goals
- **Clarity**: Easy to understand what agent is doing at any level

### Why Explicit Completion Signal?
- **Deterministic**: No guessing when agent is "done"
- **Efficient**: No extra LLM calls to check completion
- **Clear**: Both agent and user know exact state

### Why Generic Probing?
- **Flexible**: Works for any goal without hardcoding
- **Simple**: Just report facts, let LLM interpret
- **Maintainable**: No goal-specific logic to update

### Why Loop Detection?
- **Safety**: Prevents infinite resource consumption
- **Robustness**: Handles LLM mistakes gracefully
- **Visibility**: User can see what's being blocked

## Testing

### Run Simple Test
```bash
python agent.py "Write a hello world script"
```

### Run Complex Test
```bash
python agent.py "Create a rock-paper-scissors game"
```

### Check Success Criteria
1. Agent completes in < 10 rounds
2. Files are created and valid
3. Exit message is "Goal achieved"
4. No infinite loops

### Expected Behavior
- **Hello World**: ~2-3 rounds, creates hello_world.py
- **Rock-Paper-Scissors**: ~6-8 rounds, creates working game
- **Mathx Package**: ~8-12 rounds, creates package + tests, runs ruff + pytest

## Summary

This agent architecture solves the infinite loop problem through:
1. **Automatic task decomposition** - LLM creates concrete, actionable tasks
2. **Explicit completion signaling** - `mark_subtask_complete()` tool
3. **Hierarchical context management** - Focus on current branch only
4. **Generic state probing** - Facts not assumptions
5. **Loop detection** - Block repeated failed attempts

The result: A robust, local-first coding agent that completes tasks and exits cleanly without human intervention.
