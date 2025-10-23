# Jetbox

![Jetbox Logo](jetbox.png)

**JetBox — a fast, compact, and slightly dangerous local agent framework built for speed, autonomy, and total on-device control. Makes your fan scream like a jet at takeoff.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **Hierarchical Context Management** - Organizes work into Goal → Task → Subtask → Action
- ⚙️ **Fully Configurable** - YAML-based configuration for all agent behavior
- 🔄 **No Give-Up Option** - Always decomposes or zooms out (3x retry before final failure)
- 🧠 **Smart Zoom-Out** - Analyzes task tree to find root of problem (parent/task/root)
- 🔍 **Loop Detection** - Prevents infinite action loops with automatic blocking
- 💾 **Crash Recovery** - Resume from exact point of interruption
- 📊 **Real-Time Status** - Tree-based visualization with progress bars
- 🏗️ **Workspace Isolation** - Each goal gets isolated directory
- 📝 **Prompt Engineering** - External YAML prompts for easy tuning

## Quick Start

### Requirements

- Python 3.10+
- Ollama running locally with a model (default: `gpt-oss:20b`)
- Packages: `ollama`, `pytest`, `ruff`, `pyyaml`

```bash
pip install ollama pytest ruff pyyaml
ollama pull gpt-oss:20b  # or your preferred model
```

### Run the Agent

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

## Project Structure

```
jetbox/
├── agent.py                    # Main agent with hierarchical execution
├── context_manager.py          # Hierarchical state management
├── workspace_manager.py        # Workspace isolation system
├── status_display.py           # Real-time progress visualization
├── completion_detector.py      # Completion signal detection
│
├── agent_config.yaml           # ⚙️ User configuration
├── agent_config.py             # Configuration loader
├── prompts.yaml                # 📝 All agent prompts
├── prompt_loader.py            # Prompt loading utility
│
├── tests/                      # Test infrastructure
│   ├── run_stress_tests.py    # Stress test suite
│   ├── run_eval_suite.py       # Multi-iteration evaluation
│   ├── generate_eval_report.py # Report generator
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
    └── reports/                # Analysis reports
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

### 4. Workspace Isolation

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

### 5. Approach Reconsideration

When stuck, agent can reconsider approach at root:
- Learns from previous failures
- Extracts accomplishments (what worked)
- Identifies failed approaches (what didn't work)
- Generates completely new strategy
- 3 attempts before final failure

## Prompt Engineering

All prompts are externalized in `prompts.yaml` for easy tuning:

```yaml
system_prompt: |
  You are a local coding agent...

escalation_prompt: |
  ESCALATION NEEDED: You've spent {rounds_used} rounds...

decompose_subtask: |
  Break this into {min_children}-{max_children} smaller subtasks...
```

Load and format prompts:

```python
from prompt_loader import prompts

prompt = prompts.get("decompose_subtask", min_children=2, max_children=6)
```

## Testing

### Run Unit Tests

```bash
pytest tests/ -q
```

### Run Stress Tests

```bash
python tests/run_stress_tests.py 3,4,5  # Run levels 3, 4, 5
```

### Run Full Evaluation Suite

```bash
python tests/run_eval_suite.py  # 5 iterations of L3-L4-L5
```

### Generate Report

```bash
python tests/generate_eval_report.py
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

## Performance

**Evaluation Results** (L3-L4-L5 tests, 5 iterations each):
- Overall pass rate: 75.6% (34/45)
- Level 3: 53% - Advanced tasks
- Level 4: 80% - Expert tasks
- Level 5: 93% - Extreme tasks

**Interesting finding:** Agent performs better on harder tasks due to effective decomposition.

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

- **`CLAUDE.md`** - Main agent documentation
- **`AGENT_ARCHITECTURE.md`** - Detailed architecture
- **`IMPLEMENTATION_COMPLETE.md`** - Recent implementation summary
- **`EVAL_SUITE_REPORT.md`** - Latest evaluation results
- **`STATUS_DISPLAY.md`** - Status display documentation

## Development Tools

- **`agent_config.py`** - Configuration system with dataclasses
- **`prompt_loader.py`** - YAML prompt loading
- **`workspace_manager.py`** - Workspace path resolution
- **`completion_detector.py`** - Heuristic completion detection
- **`status_display.py`** - Progress visualization
- **`context_manager.py`** - Hierarchical state machine

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

## Model Configuration

Set model via environment variable:

```bash
# PowerShell
$env:OLLAMA_MODEL = "qwen2.5-coder:7b"

# Bash
export OLLAMA_MODEL="qwen2.5-coder:7b"
```

Default model: `gpt-oss:20b`

Supported models: Any Ollama-compatible model with function calling support.

## Safety

**Command Whitelist:**
Only `python`, `pytest`, `ruff`, and `pip` commands are allowed.
All other commands are blocked for Windows safety.

**Workspace Isolation:**
Agent cannot modify files outside its workspace directory.

**Loop Prevention:**
Automatic detection and blocking of infinite loops.

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

**Version:** 2.0 (Hierarchical Context Manager with Smart Zoom-Out)

**Evaluation:** 75.6% success rate on L3-L4-L5 test suite (45 tests)
