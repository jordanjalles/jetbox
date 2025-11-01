# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Jetbox is a minimal local-first coding agent that runs with Ollama on Windows. The agent uses hierarchical context management to complete tasks autonomously, avoiding infinite loops through explicit task completion signaling.

**For detailed architecture documentation, see [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)**

## Development Commands

### Testing and Linting
```bash
# Run all tests
pytest -q

# Run a single test file
pytest tests/test_mathx.py -q

# Lint with ruff
ruff check .

# Auto-fix ruff issues
ruff check --fix .
```

### Running the Agent
```bash
# Run agent with default goal
python agent.py

# Run agent with custom goal
python agent.py "Your task description here"

# Check Ollama latency
python diag_speed.py

# Demo the status display
python test_status_display.py
```

### Configuration

**Agent Behavior Configuration:**
The agent's failure handling, task decomposition, and retry behavior is configured in `agent_config.yaml`. See [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md) for details.

Key settings:
- **No give-up option**: Agent always decomposes or zooms out when stuck
- **3x approach retries**: Agent gets 3 chances to reconsider approach at root before final failure
- **Configurable depth/breadth**: Adjust max_depth, max_siblings, max_rounds_per_subtask

Set the Ollama model via environment variable:
```bash
# PowerShell
$env:OLLAMA_MODEL = "gpt-oss:20b"

# Bash
export OLLAMA_MODEL="gpt-oss:20b"
```

## Architecture

### Agent Behavior System (NEW)

Jetbox uses a **composable behavior system** where all agent capabilities are provided by pluggable behaviors that can be mixed and matched via configuration files.

**What is a Behavior?**

A behavior is a self-contained module that:
- **Injects context** into LLM prompts
- **Provides tools** for the agent to use
- **Handles events** from the agent lifecycle
- **Adds instructions** to the system prompt

**Benefits**:
- **Composable**: Mix and match any behaviors without conflicts
- **Config-driven**: Change agent behavior via YAML, not code
- **Testable**: Each behavior has isolated unit tests
- **Extensible**: Create custom behaviors with clear interfaces
- **Maintainable**: Single responsibility, no hidden dependencies

**Core Principles**:
1. **Single Responsibility**: Each behavior does ONE thing
2. **Composability**: Behaviors work independently and in any combination
3. **No Hidden Dependencies**: No behavior embeds functionality from another
4. **Config-Driven**: Behaviors configured via YAML, not hardcoded
5. **Event-Driven**: Behaviors respond to lifecycle events independently
6. **Clear Interfaces**: Standardized methods across all behaviors

**Example: TaskExecutor with Behaviors**:

```python
agent = TaskExecutorAgent(
    workspace=".",
    goal="Create a calculator",
    use_behaviors=True,  # Enable behavior system
    config_file="task_executor_config.yaml"
)
```

**Config file (task_executor_config.yaml)**:

```yaml
behaviors:
  # Context management (2 behaviors compose)
  - type: SubAgentContextBehavior      # Delegated goal header
    params: {}
  - type: CompactWhenNearFullBehavior  # Context compaction
    params:
      max_tokens: 128000

  # Tool behaviors
  - type: FileToolsBehavior
    params: {}
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "pip"]
  - type: ServerToolsBehavior
    params: {}

  # Utility behaviors
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5
  - type: WorkspaceTaskNotesBehavior    # Persistent context summaries
    params: {}
```

**Why This Composition Works**:
- `SubAgentContextBehavior` injects delegated goal header
- `CompactWhenNearFullBehavior` handles context compaction separately
- `WorkspaceTaskNotesBehavior` loads persistent notes independently
- Each behavior is single-responsibility and composable
- No conflicts between behaviors

**Available Behaviors**:

**Context Management**:
- `HierarchicalContextBehavior` - Goal/Task/Subtask hierarchy
- `CompactWhenNearFullBehavior` - Append until full, then compact
- `SubAgentContextBehavior` - For delegated work
- `ArchitectContextBehavior` - For architecture design

**Tools**:
- `FileToolsBehavior` - write_file, read_file, list_dir
- `CommandToolsBehavior` - run_bash
- `ServerToolsBehavior` - Server management
- `DelegationBehavior` - Orchestrator delegation tools
- `ArchitectToolsBehavior` - Architecture artifacts

**Utilities**:
- `LoopDetectionBehavior` - Detect infinite loops
- `WorkspaceTaskNotesBehavior` - Persistent context summaries (auto-summarizes goals/tasks)
- `StatusDisplayBehavior` - Progress visualization and performance tracking

**Delegation**:
- `DelegationBehavior` - Auto-configured from agents.yaml relationships (orchestrator only)

See **[BEHAVIORS_DOCUMENTATION.md](BEHAVIORS_DOCUMENTATION.md)** for complete documentation.

### Legacy Mode (Backward Compatible)

The old context strategy system still works for backward compatibility:

```python
agent = TaskExecutorAgent(
    workspace=".",
    goal="Create a calculator",
    use_behaviors=False  # Use legacy mode (default for now)
)
```

**Deprecated (will be removed in v2.0)**:
- `context_strategies.py` - Use behaviors instead
- Hardcoded tool definitions - Use behavior tools

See **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** for migration instructions.

### Agent Design Philosophy

**Local-first and crash-resilient**: The agent expects to crash or be stopped at any time. All designs prioritize idempotency and fast rehydration from plaintext logs. No databases are used - everything is human-inspectable files.

**Verify-first, work-backward**: Before making any LLM calls, the agent probes the current state (file existence, ruff output, pytest output) using `probe_state()` in agent.py:117. From this verified state, it generates a minimal backward-chaining checklist via `plan_next()` in agent.py:172.

**Short timeboxes**: Runs are bounded by `MAX_ROUNDS` (default 24 rounds). Each round boundary triggers a probe → compact → reflect → persist cycle.

### Core Components

**agent.py** (main agent loop):
- `probe_state_generic()` - Verifies filesystem state without goal-specific assumptions
- `build_context()` - Constructs context with system prompt + current task info + recent messages
- `dispatch()` - Executes tool calls with structured error handling
- Main loop displays status, calls LLM, executes tools, tracks performance

**context_manager.py** (hierarchical context management):
- `ContextManager` - Manages Goal → Task → Subtask → Action hierarchy
- `LoopDetector` - Detects repeated action patterns (infinite loops)
- Automatic state persistence to `.agent_context/state.json`
- Loop detection with configurable thresholds

**status_display.py** (progress visibility):
- `StatusDisplay` - Renders hierarchical task display with progress bars
- `PerformanceStats` - Tracks LLM timing, tokens, success rates, loop counts
- Real-time status updates at each agent round
- Statistics persist to `.agent_context/stats.json`
- **See [STATUS_DISPLAY.md](STATUS_DISPLAY.md) for complete documentation**

**workspace_manager.py** (workspace isolation):
- `WorkspaceManager` - Creates isolated directories for each goal
- Automatic path resolution (all file operations workspace-relative)
- File tracking and workspace-scoped test/lint commands
- Prevents context distraction from root directory files
- **Workspace parameter semantics**: `workspace=None` creates new isolated workspace, `workspace=Path` reuses existing
- **See [WORKSPACE_AND_COMPLETION_FEATURES.md](WORKSPACE_AND_COMPLETION_FEATURES.md) for details**
- **See [evaluation_results/WORKSPACE_NESTING_FIX_IMPLEMENTED.md](evaluation_results/WORKSPACE_NESTING_FIX_IMPLEMENTED.md) for workspace nesting fix details**

**completion_detector.py** (completion nudging):
- Heuristic pattern matching to detect completion signals in LLM responses
- 15+ regex patterns for phrases like "task completed successfully"
- Nudges agent to call `mark_subtask_complete()` when appropriate
- Reduces false negative completion reporting
- **See [WORKSPACE_AND_COMPLETION_FEATURES.md](WORKSPACE_AND_COMPLETION_FEATURES.md) for details**

**agent_config.py** (configuration system):
- Loads behavior settings from `agent_config.yaml`
- Controls escalation strategy, retry limits, decomposition parameters
- No give-up option: Agent always decomposes or zooms out when stuck
- Approach reconsideration: 3x retries at root before final failure
- **See [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md) for complete documentation**

**jetbox_notes.py** (persistent context system):
- Auto-summarizes completed tasks and goals
- Stores summaries in `jetboxnotes.md` within workspace
- Loads notes on subsequent runs for context continuity
- Passes summaries to parent agent or console
- Prompt-engineered for factual, concise summaries (temp 0.2)
- **See [docs/jetbox_notes/](docs/jetbox_notes/) for complete documentation**

**Tool whitelist** (agent.py:32): Only `python`, `pytest`, `ruff`, and `pip` commands are allowed for Windows safety. All other commands are rejected.

**Status artifacts**:
- `.agent_context/state.json` - Hierarchical task state (Goal/Task/Subtask/Action)
- `.agent_context/history.jsonl` - Action history (append-only)
- `.agent_context/stats.json` - Performance statistics
- `.agent_workspace/{goal-slug}/` - Isolated workspace for agent work (auto-created)
- `agent_ledger.log` - Append-only trace of WRITE/CMD/ERROR actions (audit trail)
- `agent_v2.log` - Human-readable runtime log

### Context Management Strategy

**With Behavior System**:

Context management is now handled by composable behaviors:

- **HierarchicalContextBehavior**: Keeps last N message exchanges, clears on subtask transitions
- **CompactWhenNearFullBehavior**: Appends all messages until 75% full, then LLM-summarizes old messages
- **SubAgentContextBehavior**: For delegated work, appends all messages with higher token limit (128K)
- **ArchitectContextBehavior**: For architecture discussions, optimized for verbose conversations

Configure via YAML (see config files in root directory).

**Legacy Mode (Deprecated)**:

The agent aggressively compacts context to avoid repetition and stay focused:
1. **Probe real state first** - Never hallucinate file existence or test status
2. **Compress ledger** - Tail of ledger (60 lines) is deduped and summarized into categories (Files/Cmds/Errors/Tried)
3. **Deduplicate tool calls** - Same tool call with same args is skipped after 3 attempts (agent.py:313)
4. **Prune message history** - Keep only system prompt + last user message + last `HISTORY_KEEP` (12) messages
5. **Reflection at boundaries** - Each round: probe → compact → reflect → persist status

### Tool Execution

All tools in `agent.py` are tolerant of edge cases:
- `list_dir()` - Returns error strings instead of crashing on missing paths
- `read_file()` - Truncates to 200KB, uses error replacement for encoding issues
- `write_file()` - Creates parent directories automatically
- `run_cmd()` - Enforces whitelist, captures output, logs to ledger, returns structured dict

## Key Configuration

**Runtime constants (agent.py:27-39):**
- `MODEL` - Ollama model tag (default: "gpt-oss:20b", override with `OLLAMA_MODEL` env var)
- `TEMP` - Temperature for model (0.2 for focused outputs)
- `HISTORY_KEEP` - Number of recent message exchanges to retain (5)
- `SAFE_BIN` - Whitelisted commands: `{"python", "pytest", "ruff", "pip"}`

**Configurable behavior (agent_config.yaml):**
- `rounds.max_per_subtask` - Rounds before escalation (default: 6)
- `rounds.max_per_task` - Safety cap for task rounds (default: 128)
- `hierarchy.max_depth` - Maximum nesting levels (default: 5)
- `hierarchy.max_siblings` - Max subtasks per level (default: 8)
- `escalation.strategy` - `"force_decompose"` (no give-up) or `"agent_decides"`
- `escalation.zoom_out_target` - `"root"`, `"task"`, or `"parent"`
- `escalation.max_approach_retries` - Retry attempts at root (default: 3)

**See [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md) for full configuration reference.**

## Status Display

The agent includes a comprehensive status display system showing:

- **Hierarchical task view**: Goal → Task → Subtask with visual status icons
- **Progress bars**: Task completion, subtask completion, success rate
- **Performance metrics**: LLM timing, token usage, action counts, loop detection
- **Recent activity**: Last actions and errors with success/failure indicators

Example output:
```
GOAL: Create a Python calculator package

TASKS (1/3 completed):
  ► ⟳ Create calculator package structure
    SUBTASKS:
        ✓ Write calculator/__init__.py
      ► ⟳ Write calculator/advanced.py
        ○ Write pyproject.toml
    ○ Write comprehensive tests

PROGRESS:
  Tasks:    [█████░░░░░░░░░░░░░░░] 33%
  Subtasks: [████████░░░░░░░░░░░░] 43%
  Success:  92%

PERFORMANCE:
  Avg LLM call:      2.15s
  Avg subtask time:  1m 30s
  Actions executed:  25
  Tokens (est):      3,500
```

For complete documentation, see [STATUS_DISPLAY.md](STATUS_DISPLAY.md)

## Jetbox Notes System

The agent automatically captures and persists context across runs using the jetbox notes system:

**Auto-Summarization**:
- Task completion → 2-4 bullet summary appended to `jetboxnotes.md`
- Goal success → 3-6 bullet comprehensive summary appended and displayed
- Goal failure → Failure analysis with blocking factors and retry suggestions

**Persistence**:
- Notes saved to `.agent_workspace/{goal-slug}/jetboxnotes.md`
- Markdown format, human-readable
- Survives crashes and reruns

**Context Loading**:
- Existing notes loaded on agent startup
- Included in agent context (max 2000 chars)
- No selective loading - entire file included

**Quality**:
- Temperature 0.2 for factual summaries
- Prompt-engineered for concise, specific output
- Focuses on facts future tasks need to know

**Usage**:
```bash
# Run agent (summaries happen automatically)
python agent.py "Your goal here"

# Resume work in same workspace
python agent.py --workspace .agent_workspace/previous-goal "Continue work"

# View notes
cat .agent_workspace/{goal-slug}/jetboxnotes.md
```

For complete documentation, see:
- [Quick Start Guide](docs/jetbox_notes/JETBOX_NOTES_QUICKSTART.md)
- [Technical Spec](docs/jetbox_notes/JETBOX_NOTES_SPEC.md)
- [Test Report](docs/jetbox_notes/JETBOX_NOTES_TEST_REPORT.md)
- [Implementation Summary](docs/jetbox_notes/IMPLEMENTATION_COMPLETE.md)

## Ruff Configuration (pyproject.toml:1-5)

Ruff is configured with line length 88 and enforces:
- Error/warning codes: E, F, W
- Import sorting: I
- Modern Python: UP
- Naming conventions: N
- Type annotations: ANN (with some ignores for common cases)
- Complexity: C90
- Print statements: T20
- Simplification: SIM
- Unused arguments: ARG
- Pathlib usage: PTH
- Ruff-specific rules: RUF

Ignored codes allow missing annotations for `self`, `cls`, args, kwargs, and `Any`.

## Testing (pyproject.toml:7-12)

Pytest runs in quiet mode by default (`-q`). Test discovery follows standard conventions:
- Test files: `test_*.py` in `tests/` directory
- Test classes: `Test*`
- Test functions: `test_*`

## Recovery and Resumability

On startup, the agent:
1. Loads `.agent_context/state.json` if present (last known task hierarchy)
2. Probes current filesystem and tool outputs
3. Resumes from the last in-progress subtask
4. Continues from where it left off

The agent is designed to be crash-resilient:
- All state persisted to plaintext JSON files
- Action history append-only in `.agent_context/history.jsonl`
- Performance stats preserved in `.agent_context/stats.json`
- No databases - everything human-inspectable

## Development Patterns

When working in this codebase:

1. **Always probe before planning** - Run filesystem checks and tools before asking the LLM to decide what to do
2. **Prefer compact context** - Summarize and deduplicate rather than including full histories
3. **Design for interruption** - Any operation should be resumable from logs
4. **Use backward chaining** - Start from desired end state (tests pass) and work backward to current state
5. **Whitelist over blacklist** - Commands are allowed via explicit whitelist only
6. **Log everything to ledger** - WRITE/CMD/ERROR/TRIED actions go to `agent_ledger.log` for audit trail

## Package Structure

**mathx/** - Sample package demonstrating agent capabilities
- `add(a, b)` - Simple addition function (mathx/__init__.py:1)

**tests/** - Pytest test suite
- `test_mathx.py` - Three assertions testing add() with positive, negative, and zero values
- store report and summary markdowns in a subfolder that won't get added to github
- remember not to kill shell because it kills the claude process too
- swapping models in ollama takes some time