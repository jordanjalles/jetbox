# Jetbox

![Jetbox Logo](jetbox.png)

**JetBox — a local agent framework built for speed, autonomy, and total on-device control. Makes your fan scream like a jet at takeoff.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Recent Improvements

**v3.1 - Reliability & Error Recovery (Nov 2025)**
- 🛡️ **Empty Round Recovery** - Detects when LLM stops calling tools and injects recovery prompts
- ⚡ **Tool Call Error Feedback** - Immediate actionable feedback for malformed tool calls
- 📝 **Workspace Task Notes** - Renamed from jetboxnotes for clarity
- 🔒 **Configurable Whitelist** - Command safety via `jetbox_commands_whitelist` file
- 📊 **Performance Impact**: 98% faster recovery (40min → <1min), 81.5% success rate improvement

## Features

- 🎯 **Multi-Agent Orchestration** - Orchestrator → Architect → TaskExecutor workflow
- 🏗️ **Composable Enhancements** - Pluggable context management with TaskManagement + WorkspaceTaskNotes
- 📐 **Architecture Consulting** - Architect agent for complex project design
- 📋 **Task Management** - Structured task tracking with dependency resolution
- 📝 **Persistent Context** - Jetbox notes preserve context across sessions
- ⚙️ **Fully Configurable** - YAML-based configuration for all agent behavior
- 🔍 **Loop Detection** - Prevents infinite action loops with automatic recovery
- 🛡️ **Error Recovery** - Immediate feedback for malformed tool calls with actionable instructions
- 💾 **Crash Recovery** - Resume from exact point of interruption
- 🏗️ **Workspace Isolation** - Each goal gets isolated directory
- ⏱️ **Timeout Protection** - Automatic detection and recovery from LLM hangs
- 🔒 **Command Whitelist** - Configurable safety with `jetbox_commands_whitelist` file

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

#### Orchestrator Mode (Recommended - Full Workflow)

```bash
python orchestrator_main.py
```

Interactive mode with multi-agent workflow:
- Chat naturally with the orchestrator
- **Simple tasks**: Orchestrator delegates directly to TaskExecutor
- **Complex projects**: Orchestrator consults Architect first for design
- Architect creates architecture docs, module specs, and task breakdown
- Orchestrator manages task execution with dependency tracking
- Each task runs in isolated workspace

**Example conversation:**
```
You: Build a microservices system with user and product services
Orchestrator: → Consulting Architect...
Architect: Created architecture with 3 services, 8 modules, 12 tasks
Orchestrator: → Delegating Task 1: Set up user service...
[TaskExecutor implements user service]
Orchestrator: → Delegating Task 2: Set up product service...
[TaskExecutor implements product service]
Orchestrator: All 12 tasks completed! Files in workspace/.
```

#### One-Shot Mode (Quick Tasks)

```bash
python orchestrator_main.py --once "Create a JSON to CSV converter"
```

Processes request and exits (no interactive loop).

#### TaskExecutor Mode (Direct Execution)

```bash
python agent.py "Create a calculator module with tests"
```

Direct execution without orchestrator:
- Hierarchical task decomposition
- Autonomous execution with progress tracking
- Best for: Scripts, tools, testing, batch operations

### Example Session

```bash
$ python orchestrator_main.py --once "Library management system with book catalog and member services"

[orchestrator] Context strategy: append_until_full

User: Library management system with book catalog and member services

→ Consulting Architect: Library management system with book catalog...

============================================================
ARCHITECT CONSULTATION
============================================================
Project: Library management system with book catalog and member services
============================================================

[Round 1] Architect: Creating architecture overview...
[Round 2] Architect: Writing module specs...
[Round 3] Architect: Breaking down into tasks...

✅ Architecture consultation complete!

Workspace: .agent_workspace/library-management-system
Architecture documents (1):
  - architecture/system-overview.md
Module specifications (2):
  - architecture/modules/book-catalog-service.md
  - architecture/modules/member-management-service.md
Task breakdown: architecture/task-breakdown.json
  (7 tasks ready for delegation)

Tasks to delegate:
  [T1] Set up Flask application for Book Catalog Service
  [T2] Implement CRUD endpoints for books (depends on: T1)
  [T3] Set up Flask application for Member Management Service
  [T4] Implement CRUD endpoints for members (depends on: T3)
  ...

[orchestrator] Added task management enhancement (7 tasks)

→ Delegating to TaskExecutor: Set up Flask application for Book Catalog...

[TaskExecutor creates Book Catalog service]

→ Delegating to TaskExecutor: Implement CRUD endpoints for books...

[TaskExecutor implements endpoints]

...
```

## Multi-Agent Architecture

Jetbox uses a **composable multi-agent architecture** with three specialized agents:

### Architecture Overview

```
User ↔ Orchestrator (conversation + task management)
         ↓ consults (complex projects)
       Architect (architecture design)
         ↓ produces task breakdown
       Orchestrator (task delegation)
         ↓ delegates tasks
       TaskExecutor (autonomous execution)
```

### Agent Roles

**1. Orchestrator** - Conversational coordinator
- Manages user conversation
- Assesses project complexity
- Consults Architect for complex projects
- Delegates coding tasks to TaskExecutor
- Tracks task progress with dependency management
- **Enhancement**: TaskManagement (when task breakdown exists)
- **Base Strategy**: AppendUntilFull (message history management)

**2. Architect** - Design consultant
- Creates system architecture overviews
- Writes module specifications
- Breaks down projects into structured tasks
- Produces artifacts in `workspace/architecture/`
- No code execution - design only
- **Enhancement**: TaskManagement (auto-detected for existing projects)
- **Base Strategy**: AppendUntilFull (architecture discussions)

**3. TaskExecutor** - Code execution specialist
- Hierarchical task decomposition
- Autonomous code execution
- File operations, testing, linting
- Progress tracking and crash recovery
- **Enhancement**: WorkspaceTaskNotes (persistent context summaries)
- **Base Strategy**: AppendUntilFull (execution history)

### Workflow Examples

**Simple Task (No Architect)**:
```
User: "Create a JSON to CSV converter"
  ↓
Orchestrator: Direct delegation
  ↓
TaskExecutor: Creates converter.py
  ↓
User: "Task complete!"
```

**Complex Project (With Architect)**:
```
User: "Build a microservices platform"
  ↓
Orchestrator: Assess → COMPLEX
  ↓
Architect: Design architecture
  ↓ produces
  - Architecture docs
  - Module specs
  - Task breakdown (12 tasks with dependencies)
  ↓
Orchestrator: Add TaskManagement enhancement
  ↓
Orchestrator: Delegate Task 1
  ↓
TaskExecutor: Implement Task 1
  ↓
Orchestrator: Mark T1 complete, delegate Task 2
  ↓
... (continues for all tasks)
```

## Composable Enhancement System

Jetbox uses a **composable enhancement architecture** where agents combine a base strategy with pluggable enhancements:

### Base + Enhancements Pattern

**All agents use**:
- **Base**: `AppendUntilFullStrategy` - Manages message history and compaction

**Plus agent-specific enhancements**:
- **Orchestrator**: `+ TaskManagementEnhancement` (when tasks exist)
- **Architect**: `+ TaskManagementEnhancement` (auto-detected)
- **TaskExecutor**: `+ WorkspaceTaskNotesEnhancement` (always enabled)

### Enhancement Features

**TaskManagementEnhancement**:
- Injects task status into context (pending/in_progress/completed/failed)
- Shows next pending task with dependency checking
- Provides CRUD tools: read_task_breakdown, get_next_task, mark_task_status
- Displays first 5 tasks with visual status icons (○ ⟳ ✓ ✗)
- State persisted in `architecture/task-breakdown.json`

**WorkspaceTaskNotesEnhancement**:
- Loads previous work summaries from `workspace_task_notes.md`
- Auto-summarizes on task/goal completion
- Provides context continuity across agent restarts
- No additional tools needed (automatic)

### Benefits of Composition

✅ **No code duplication**: Enhancements are ~100-300 lines each
✅ **True composition**: Multiple enhancements can coexist
✅ **Clean separation**: Base handles messages, enhancements add context
✅ **Easy extension**: New enhancements follow simple interface
✅ **Backward compatible**: Old strategies wrap new enhancements

## Project Structure

```
jetbox/
├── Core Agents
│   ├── orchestrator_main.py        # Orchestrator entry point
│   ├── orchestrator_agent.py       # Conversational coordinator
│   ├── architect_agent.py          # Architecture design consultant
│   ├── base_agent.py               # Base agent class
│   ├── agent.py                    # TaskExecutor - hierarchical execution
│   └── agent_registry.py           # Multi-agent registry
│
├── Enhancements & Strategies
│   ├── context_strategies.py       # Context strategies + enhancements
│   ├── task_management_tools.py    # Task CRUD operations
│   ├── jetbox_notes.py             # Persistent context summaries
│   └── architect_tools.py          # Architecture artifact creation
│
├── Context Management
│   ├── context_manager.py          # Hierarchical state management
│   ├── workspace_manager.py        # Workspace isolation system
│   ├── status_display.py           # Real-time progress visualization
│   └── completion_detector.py      # Completion signal detection
│
├── Configuration
│   ├── agent_config.yaml           # ⚙️ User configuration
│   ├── agent_config.py             # Configuration loader
│   ├── agents.yaml                 # Multi-agent configuration
│   ├── prompts.yaml                # 📝 All agent prompts
│   └── prompt_loader.py            # Prompt loading utility
│
├── Testing & Evaluation
│   ├── tests/                      # Test infrastructure
│   │   ├── test_project_evaluation.py    # L5-L8 eval suite (15 tests)
│   │   ├── test_orchestrator_*.py        # Orchestrator tests
│   │   └── test_*.py                     # Unit tests
│   ├── run_project_evaluation.py         # Evaluation runner
│   └── evaluation_results/
│       ├── EVALUATION_SUITE_README.md    # Eval documentation
│       ├── *.md                          # Analysis reports
│       └── project_eval_results.jsonl    # Test results
│
├── Runtime State (auto-created)
│   ├── .agent_context/             # Agent state
│   │   ├── state.json              # Complete hierarchical state
│   │   ├── history.jsonl           # Action history log
│   │   └── stats.json              # Performance statistics
│   │
│   └── .agent_workspace/           # Isolated workspaces
│       └── {goal-slug}/            # One workspace per goal
│           ├── architecture/       # Architect artifacts (if used)
│           │   ├── *.md            # Architecture docs
│           │   ├── modules/*.md    # Module specs
│           │   └── task-breakdown.json   # Task breakdown
│           ├── workspace_task_notes.md      # Context summaries
│           └── ... (implementation files)
│
└── Documentation
    ├── CLAUDE.md                          # AI assistant instructions
    ├── docs/                              # Technical documentation
    │   ├── INTEGRATION_ISSUES.md          # Integration fixes
    │   └── architecture/                  # Architecture docs
    ├── evaluation_results/                # Evaluation and test results
    │   └── EVALUATION_SUITE_README.md     # Evaluation guide
    └── analysis/                          # Test analysis reports
```

## Configuration

### Agent Behavior

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

# Loop Detection
loop_detection:
  max_action_repeats: 3          # Block after N repeats
  max_subtask_repeats: 2         # Escalate after N subtask repeats

# Context Management
context:
  max_messages_in_memory: 12     # Message pairs in context
  recent_actions_limit: 10        # Recent actions to show
  enable_compression: true        # Summarize old messages
```

### Command Whitelist

Control which bash commands can be executed by editing `jetbox_commands_whitelist`:

```bash
# Jetbox Commands Whitelist
# Only commands listed here can be executed via run_bash tool

# Python tools
python
pytest
pip

# Linting
ruff
black
mypy

# Version control
git

# File operations
ls
cat
grep
find
# ... add more as needed
```

Comments (lines starting with `#`) and empty lines are ignored.

## Task Management

When Architect creates a project, tasks are tracked in `architecture/task-breakdown.json`:

```json
{
  "generated_at": "2025-10-31T...",
  "total_tasks": 7,
  "tasks": [
    {
      "id": "T1",
      "description": "Set up Flask application for Book Catalog",
      "module": "book-catalog-service",
      "dependencies": [],
      "estimated_complexity": "low",
      "priority": 1,
      "status": "completed"
    },
    {
      "id": "T2",
      "description": "Implement CRUD endpoints for books",
      "module": "book-catalog-service",
      "dependencies": ["T1"],
      "estimated_complexity": "medium",
      "priority": 2,
      "status": "in_progress"
    }
  ]
}
```

**Orchestrator context shows**:
```
======================================================================
TASK BREAKDOWN STATUS
======================================================================
Total Tasks: 7
  Pending:     3
  In Progress: 1
  Completed:   3
  Failed:      0

NEXT PENDING TASK:
  [T5] Set up Flask for Member Management Service
  Module: member-management-service
  Complexity: low
  Dependencies: (none)

TASK LIST (first 5):
  ✓ [T1] Set up Flask application for Book Catalog
  ✓ [T2] Implement CRUD endpoints for books
  ✓ [T3] Write tests for Book Catalog endpoints
  ⟳ [T4] Set up Flask for Member Management
  ○ [T5] Implement CRUD endpoints for members (deps: T4)
======================================================================
```

## Jetbox Notes System

Persistent context across runs via automatic summarization:

**Auto-Summarization**:
- Task completion → 2-4 bullet summary appended to `workspace_task_notes.md`
- Goal success → 3-6 bullet comprehensive summary
- Goal failure → Failure analysis with retry suggestions

**Persistence**:
- Notes saved in workspace (`workspace_task_notes.md`)
- Markdown format, human-readable
- Survives crashes and reruns

**Context Loading**:
- Loaded automatically on agent startup
- Included in agent context (max 2000 chars)
- Injected after system prompt

**Example**:
```markdown
# Jetbox Notes

## Task Complete - 2025-10-31 20:15:32

- Created Book Catalog Flask application with SQLAlchemy
- Implemented GET/POST/PUT/DELETE endpoints for books
- All tests passing with 100% coverage

---

## ✓ GOAL COMPLETE - 2025-10-31 20:45:12

- Implemented full library management system with 2 services
- Book Catalog service: CRUD operations with SQLite backend
- Member Management service: CRUD operations with validation
- All 7 tasks completed successfully
- Tests pass, code quality excellent (0 ruff errors)

---
```

## Evaluation Suite

Comprehensive test suite for orchestrator workflow:

```bash
# Run L5 simple utilities (3 tests)
python run_project_evaluation.py --level L5

# Run L6 multi-file modules (3 tests)
python run_project_evaluation.py --level L6

# Run L7 complete packages (3 tests)
python run_project_evaluation.py --level L7

# Run L8 full projects - with and without architect (6 tests)
python run_project_evaluation.py --level L8

# Run all 15 tests
python run_project_evaluation.py
```

**Test Categories**:
- **L5**: Simple single-file utilities (JSON converter, validator, calculator)
- **L6**: Multi-file modules (API client, pipeline, config manager)
- **L7**: Complete packages with tests (package with setup.py, library, CLI tool)
- **L8**: Full multi-component systems (microservices, web app, distributed system)

**Each test validates**:
- Code completeness (all required files created)
- Tests pass (if tests requested)
- Code quality (ruff checks pass)
- Architecture adherence (L8 only - matches architect specs)

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

### Run Evaluation Suite

```bash
# Quick validation (no LLM calls)
python test_eval_suite_quick.py

# Run one test
pytest tests/test_project_evaluation.py::test_l5_task[L5_json_csv_converter] -v

# Run full project evaluation
python run_project_evaluation.py
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

## Documentation

- **`CLAUDE.md`** - Project instructions for AI assistants
- **`docs/`** - Technical documentation
  - **`INTEGRATION_ISSUES.md`** - Integration fixes and improvements
  - **`architecture/`** - Architecture documentation
- **`evaluation_results/`** - Evaluation and test results
  - **`EVALUATION_SUITE_README.md`** - Evaluation suite guide
- **`analysis/`** - Test analysis reports

## Architecture Principles

1. **Local-first and crash-resilient** - Designed to crash/stop anytime
2. **Idempotent operations** - Safe to retry any action
3. **Plaintext state** - Human-inspectable JSON/JSONL files
4. **No databases** - Everything is files
5. **Composable enhancements** - Pluggable context management
6. **Probe-first** - Verify real state before planning
7. **Compact context** - Minimize LLM context size

## Safety

**Command Whitelist:**
Commands are controlled by the `jetbox_commands_whitelist` file in the root directory.
Only commands listed in this file are allowed for execution safety.

Default whitelist includes:
- Python tools: `python`, `pytest`, `pip`
- Linting: `ruff`, `black`, `mypy`
- Version control: `git`
- File operations: `ls`, `cat`, `grep`, `find`, `wc`, `head`, `tail`
- Package management: `npm`, `node`

To customize, edit `jetbox_commands_whitelist` (one command per line).

**Workspace Isolation:**
Agent cannot modify files outside its workspace directory.

**Loop Prevention:**
Automatic detection with recovery prompts after 3 consecutive empty rounds.

**Error Recovery:**
Malformed tool calls trigger immediate feedback with actionable instructions to help the LLM self-correct.

**Timeout Protection:**
Automatic detection and recovery from LLM hangs (no max_rounds limit, uses wall-clock timeout).

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

**Version:** 3.1 (Reliability & Error Recovery)
