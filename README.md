# Jetbox

![Jetbox Logo](jetbox.png)

**JetBox — a fast, compact, and slightly dangerous local agent framework built for speed, autonomy, and total on-device control. Makes your fan scream like a jet at takeoff.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **Hierarchical Context Management** - Organizes work into Goal → Task → Subtask → Action
- 🤝 **Multi-Agent Orchestration** - Conversational orchestrator delegates to task executor
- ⚙️ **Fully Configurable** - YAML-based configuration for all agent behavior
- 🔄 **No Give-Up Option** - Always decomposes or zooms out (3x retry before final failure)
- 🧠 **Smart Zoom-Out** - Analyzes task tree to find root of problem (parent/task/root)
- 🔍 **Loop Detection** - Prevents infinite action loops with automatic blocking
- 💾 **Crash Recovery** - Resume from exact point of interruption
- 📊 **Real-Time Status** - Tree-based visualization with progress bars
- 🏗️ **Workspace Isolation** - Each goal gets isolated directory
- ⏱️ **Timeout Protection** - Automatic detection and recovery from LLM hangs
- 📝 **Prompt Engineering** - External YAML prompts for easy tuning

## Quick Start

### Requirements

- Python 3.10+
- Ollama running locally with a model (default: `gpt-oss:20b`)
- Packages: `ollama`, `pytest`, `ruff`, `pyyaml`, `requests`

```bash
pip install ollama pytest ruff pyyaml requests
ollama pull gpt-oss:20b  # or your preferred model
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jetbox.git
cd jetbox

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve  # if not already running
```

### Run the Agent

#### TaskExecutor Mode (Direct Execution)

```bash
python agent.py "Create a calculator module with add, subtract, multiply functions and tests"
```

The agent will:
1. Decompose the goal into tasks and subtasks
2. Execute actions in isolated workspace
3. Automatically advance through the hierarchy
4. Display real-time progress
5. Retry up to 3 times if approaches fail
6. Save all state for crash recovery

#### Orchestrator Mode (Multi-Agent Conversation)

```bash
python orchestrator_main.py
```

Interactive mode with conversational orchestrator:
- Chat naturally with the orchestrator
- Orchestrator clarifies requirements when needed
- Automatically delegates coding tasks to TaskExecutor
- Receive results and continue conversation
- Each delegation runs in isolated workspace

**Example conversation:**
```
You: make a simple HTML calculator
Orchestrator: → Delegating to TaskExecutor...
[TaskExecutor creates calculator.html]
Orchestrator: Task completed! Files created in workspace.

You: add support for keyboard input
Orchestrator: → Delegating to TaskExecutor...
[TaskExecutor adds keyboard support]
```

### Example Session

```bash
$ python agent.py "Create a simple todo list CLI app"

[info] Checking Ollama health...
[info] Ollama is responsive
[log] Starting agent with goal: Create a simple todo list CLI app
[log] Mode: ISOLATE (isolated workspace)
[log] Workspace: .agent_workspace/create-a-simple-todo-list-cli-app
[log] Decomposing goal into tasks...
[log] Decomposed into 3 tasks

======================================================================
INITIAL TASK TREE
======================================================================

Task 1/3 | Subtask 1/2 | ✓0% | 0.0s

GOAL: Create a simple todo list CLI app

TASK TREE (0/3 completed):
  ► ⟳ Create todo.py with CLI interface
    ► ⟳ Write todo.py with add/list/complete functions
      ○ Add command-line argument parsing
    ○ Create tests for todo functionality
      ○ Write test_todo.py
      ○ Run pytest
    ○ Verify code quality
      ○ Run ruff check

...
```

## Project Structure

```
jetbox/
├── agent.py                    # TaskExecutor - hierarchical execution
├── orchestrator_main.py        # Orchestrator entry point
├── orchestrator_agent.py       # Conversational orchestrator
├── agent_registry.py           # Multi-agent registry
├── base_agent.py               # Base agent class
│
├── context_manager.py          # Hierarchical state management
├── workspace_manager.py        # Workspace isolation system
├── status_display.py           # Real-time progress visualization
├── completion_detector.py      # Completion signal detection
│
├── agent_config.yaml           # ⚙️ User configuration
├── agent_config.py             # Configuration loader
├── agents.yaml                 # Multi-agent configuration
├── prompts.yaml                # 📝 All agent prompts
├── prompt_loader.py            # Prompt loading utility
│
├── tests/                      # Test infrastructure
│   ├── run_stress_tests.py    # Stress test suite
│   ├── run_l3_l7_x5.py         # L3-L7 evaluation (5 iterations)
│   ├── test_orchestrator_*.py  # Orchestrator tests
│   └── test_*.py               # Unit tests
│
├── .agent_context/             # Runtime state (auto-created)
│   ├── state.json              # Complete hierarchical state
│   ├── history.jsonl           # Action history log
│   ├── loops.json              # Detected loop patterns
│   └── stats.json              # Performance statistics
│
├── .agent_workspace/           # Isolated workspaces (auto-created)
│   └── {goal-slug}/            # One workspace per goal
│
└── docs/                       # Documentation
    └── *.md                    # Architecture and analysis docs
```

## Configuration

Customize agent behavior in `agent_config.yaml`:

```yaml
# Escalation Strategy
escalation:
  strategy: "force_decompose"    # No give-up option
  zoom_out_target: "smart"       # Intelligent zoom analysis
  max_approach_retries: 3        # Retry attempts before failure
  block_failed_paths: true       # Prevent retrying failed approaches

# Round Limits
rounds:
  max_per_subtask: 12            # Rounds before escalation
  max_per_task: 256              # Safety cap per task
  max_global: 24                 # Global round limit

# Hierarchy Limits
hierarchy:
  max_depth: 5                   # Max nesting levels
  max_siblings: 8                # Max subtasks per level

# Decomposition Behavior
decomposition:
  min_children: 2                # Min subtasks when decomposing
  max_children: 6                # Max subtasks when decomposing
  temperature: 0.2               # LLM temperature for planning
  prefer_granular: true          # Prefer more, smaller subtasks

# Loop Detection
loop_detection:
  max_action_repeats: 3          # Block after N repeats
  max_subtask_repeats: 2         # Escalate after N subtask repeats
  max_context_age: 300           # Context staleness threshold (seconds)

# Context Management
context:
  max_messages_in_memory: 12     # Message pairs in context
  max_tokens: 8000                # Token limit (0 = disabled)
  recent_actions_limit: 10        # Recent actions to show
  enable_compression: true        # Summarize old messages
  compression_threshold: 20       # Compress when > N messages
```

## Multi-Agent Architecture

Jetbox now supports two modes of operation:

### 1. TaskExecutor Mode (Single Agent)
Direct execution mode for straightforward tasks:
- Run with `python agent.py "task description"`
- Hierarchical task decomposition
- Autonomous execution with progress tracking
- Best for: Scripts, tools, testing, batch operations

### 2. Orchestrator Mode (Multi-Agent)
Conversational mode with task delegation:
- Run with `python orchestrator_main.py`
- Natural language conversation interface
- Orchestrator clarifies requirements and delegates to TaskExecutor
- TaskExecutor runs in isolated workspace per task
- Best for: Interactive development, complex projects, iterative work

**Architecture:**
```
User ↔ Orchestrator (conversation + planning)
          ↓ delegates
        TaskExecutor (autonomous execution)
```

**Key Benefits:**
- **Separation of concerns** - Orchestrator handles conversation, TaskExecutor handles work
- **Workspace isolation** - Each delegation gets clean workspace
- **Context preservation** - Orchestrator maintains conversation history
- **Failure handling** - TaskExecutor failures reported back to Orchestrator

**See:** `ORCHESTRATOR_TEST_RESULTS.md` for test results and benchmarks

## Key Features Explained

### 1. Hierarchical Context Management

The agent organizes work into a tree structure:

```
Goal: Create calculator with tests
├── Task 1: Create calculator module
│   ├── Subtask: Write calculator.py with add function
│   │   └── Action: write_file("calculator.py", ...)
│   └── Subtask: Write calculator.py with subtract function
│       └── Action: write_file("calculator.py", ...)
├── Task 2: Add tests
│   └── Subtask: Write test_calculator.py
│       └── Action: write_file("test_calculator.py", ...)
└── Task 3: Verify quality
    ├── Subtask: Run ruff linter
    └── Subtask: Run pytest
```

**Benefits:**
- **Automatic progression** - Completes subtasks and advances automatically
- **Compact context** - Only shows current branch (60-80% token reduction)
- **Crash recovery** - Resume from exact subtask after interruption
- **Need-to-know** - Each level only sees relevant parent context

### 2. Smart Zoom-Out

When stuck at max depth, the agent analyzes the task tree to determine if the problem is:
- **Localized** (zoom to parent) - Only 1-2 siblings failed
- **Systemic** (zoom to task) - Parent is struggling, need different approach
- **Fundamental** (zoom to root) - Multiple branches failing, reconsider entire strategy

### 3. Loop Detection

Detects and blocks:
- Same action repeated 3+ times
- Alternating patterns (A→B→A→B)
- Similar actions in recent history

When loop detected:
- Action is permanently blocked
- Warning shown to agent
- Forces different approach

### 4. Timeout Protection

Built-in timeout protection prevents infinite hangs:
- Monitors LLM response activity (not total time)
- Detects when Ollama stops responding (30s of inactivity)
- Generates failure report and exits gracefully
- Allows long complex tasks to complete normally

### 5. Workspace Isolation

Each goal gets isolated workspace:
```
.agent_workspace/
└── create-calculator-with-tests/
    ├── calculator.py
    ├── test_calculator.py
    └── __pycache__/
```

**Benefits:**
- No root directory pollution
- Parallel goal execution possible
- Easy cleanup
- Clear scope boundaries

### 6. Approach Reconsideration

When stuck, agent can reconsider approach at root:
- Learns from previous failures
- Extracts accomplishments (what worked)
- Identifies failed approaches (what didn't work)
- Generates completely new strategy
- 3 attempts before final failure

## Model Configuration

Set model via environment variable:

```bash
# PowerShell
$env:OLLAMA_MODEL = "qwen2.5-coder:7b"

# Bash
export OLLAMA_MODEL="qwen2.5-coder:7b"

# Or use any Ollama-compatible model
export OLLAMA_MODEL="llama3.2:3b"
```

Default model: `gpt-oss:20b`

Supported models: Any Ollama-compatible model with function calling support.

## Testing

### Run Unit Tests

```bash
pytest tests/ -q
```

### Run Stress Tests

Run specific levels (1-7):
```bash
python tests/run_stress_tests.py 3,4,5  # Run levels 3, 4, 5
```

### Run L3-L7 Evaluation Suite

5 iterations of levels 3-7 (75 tests total):
```bash
python tests/run_l3_l7_x5.py
```

### Lint Code

```bash
ruff check .
ruff check --fix .  # Auto-fix issues
```

### Check Ollama Latency

```bash
python diag_speed.py
```

## Status Display

Real-time visualization shows:

```
GOAL: Create calculator package

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
  ⚠ Loops detected:  1
```

## Documentation

- **`CLAUDE.md`** - Project instructions for AI assistants
- **`QUICK_START.md`** - Quick reference guide
- **`ORCHESTRATOR_TEST_RESULTS.md`** - Multi-agent test results
- **`ORCHESTRATOR_TEST_FINDINGS.md`** - Multi-agent issue analysis

### Architecture Documentation (`docs/architecture/`)
System design and component documentation

### Implementation Details (`docs/implementation/`)
Feature implementations, fixes, and design proposals

### Analysis & Reports (`docs/analysis/`)
Test results, failure analysis, and evaluation reports

## Crash Recovery

Agent automatically recovers from crashes:

```bash
# Session 1: Crashes mid-task
$ python agent.py "Create mathx package..."
[log] Created mathx/__init__.py
[log] Advanced to task: Add tests
^C  # Interrupted

# Session 2: Resumes exactly where it left off
$ python agent.py "Create mathx package..."
[log] Resuming existing task hierarchy
[log] Current task: Add tests (in_progress)
# ... continues from same point ...
```

State persists in:
- `.agent_context/state.json` - Full hierarchical state
- `.agent_context/history.jsonl` - Complete action history
- `.agent_context/stats.json` - Performance metrics

## Safety

**Command Whitelist:**
Only `python`, `pytest`, `ruff`, and `pip` commands are allowed.
All other commands are blocked for Windows safety.

**Workspace Isolation:**
Agent cannot modify files outside its workspace directory.

**Loop Prevention:**
Automatic detection and blocking of infinite loops.

**Timeout Protection:**
Automatic detection and recovery from LLM hangs.

**Failure Reports:**
Comprehensive markdown reports generated on all failures.

## Architecture Principles

1. **Local-first and crash-resilient** - Designed to crash/stop anytime
2. **Idempotent operations** - Safe to retry any action
3. **Plaintext state** - Human-inspectable JSON/JSONL files
4. **No databases** - Everything is files
5. **Short timeboxes** - Bounded execution with frequent persistence
6. **Probe-first** - Verify real state before planning
7. **Compact context** - Minimize LLM context size

## Troubleshooting

### Ollama Not Responding

```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
# Windows: Restart Ollama app
# Linux: systemctl restart ollama
# macOS: Restart Ollama app
```

### Agent Hanging

The agent has built-in timeout protection. If hanging persists:
1. Check Ollama health: `python diag_speed.py`
2. Try a different model: `export OLLAMA_MODEL="qwen2.5-coder:7b"`
3. Check system resources (RAM, CPU)

### Tests Failing

```bash
# Clean workspace and context
rm -rf .agent_workspace .agent_context

# Run single test in isolation
python agent.py "Write a hello world script"

# Check dependencies
pip install -r requirements.txt
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests: `pytest tests/ -q`
5. Run linter: `ruff check .`
6. Submit pull request

## License

MIT License - See LICENSE file for details

## Credits

Built with:
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Python](https://python.org/) - Programming language
- [Pytest](https://pytest.org/) - Testing framework
- [Ruff](https://github.com/astral-sh/ruff) - Fast Python linter

---

**Status:** Production-ready, actively maintained

**Version:** 2.2 (Multi-Agent Orchestration)
