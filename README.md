# Jetbox

Tiny local coding agent + sample package, built to run with Ollama on Windows. It includes:

- **Two production-ready agents** (both use gpt-oss:20b):
  - `agent_enhanced.py` - Enhanced agent with hierarchical context manager (baseline)
  - `agent_quality.py` - ⭐ **Optimized agent - 1.7x faster** (recommended)
- **Hierarchical context manager** (`context_manager.py`) - Crash-resilient task tracking with loop detection
- **Performance optimizations** - Probe caching, parallel execution, model selection (see `OPTIMIZATION_SUMMARY.md`)
- A tiny demo package `mathx` with `add(a, b)`, `multiply(a, b)` and comprehensive tests
- Dev tooling via `pytest` and `ruff` configured in `pyproject.toml`
- Comprehensive profiling tools (`profile_*.py`) and benchmarks

## Requirements

- Python 3.10+
- Ollama running locally, with a model available (default: `gpt-oss:20b`)
- Python packages: `ollama`, `pytest`, `ruff`

Install packages:

```
python -m pip install --upgrade pip
pip install ollama pytest ruff
```

## Project Structure

```
.
├─ agent.py                  # Basic agent with flat context
├─ agent_enhanced.py         # ⭐ Enhanced agent with hierarchical context (recommended)
├─ context_manager.py        # Hierarchical context manager (Goal→Task→Subtask→Action)
├─ agent_integration.py      # Integration guide and examples
├─ test_context_manager.py  # Tests for context manager
├─ diag_speed.py             # Quick timing test for Ollama responses
├─ mathx/
│  └─ __init__.py            # Demo: add(a, b), multiply(a, b)
├─ tests/
│  └─ test_mathx.py          # Pytest tests for mathx functions
├─ .agent_context/           # Context manager state (created at runtime)
│  ├─ state.json            # Hierarchical task state
│  ├─ history.jsonl         # Action history log
│  └─ loops.json            # Detected loop patterns
└─ pyproject.toml            # pytest + ruff config
```

## Usage

### Quick Start (Quality Agent - Recommended)

1) Make sure Ollama is running and you have a local model:

```bash
ollama list  # Verify model is available
```

2) Run the optimized quality agent (1.7x faster with professional code quality):

```bash
python agent_quality.py "Create mathx package with add(a,b) and multiply(a,b), add tests, run ruff and pytest."
```

The quality agent features:
- ⭐ **gpt-oss:20b** for production-quality code
- ⚡ **LLM warm-up** - eliminates 9.2s cold-start penalty (98.4% reduction!)
- ⚡ **Keep-alive thread** - maintains <200ms latency throughout workflow
- ✅ **Probe caching** - 70% cache hit rate, 250-350ms savings per round
- ✅ **Parallel execution** - ruff + pytest run concurrently
- ✅ **Hierarchical context** - automatic task progression with loop detection
- ✅ **Crash recovery** - resume from exact point of interruption

**Performance:**
- **Total time:** ~5.8s per workflow (vs 10s baseline)
- **Speedup:** 1.7x faster
- **Code quality:** ★★★★★ (type hints, docstrings, professional formatting)
- **First call:** 155ms (vs 9,376ms cold start)
- **Subsequent calls:** 146-600ms consistently

### Basic Agent (Original)

Run the basic agent with flat context:

```bash
python agent.py "Create a tiny package 'mathx' with add(a,b), add tests, then run ruff and pytest."
```

### Testing and Validation

Run tests directly:

```bash
pytest tests/ -q
```

Lint with ruff:

```bash
ruff check .
```

Test context manager:

```bash
python test_context_manager.py
```

Quick model latency check:

```bash
python diag_speed.py
```

## Hierarchical Context Manager

The enhanced agent uses a hierarchical context manager that organizes work into a tree structure:

```
Goal: "Create mathx package with add and multiply functions, add tests, run ruff and pytest"
│
├─ Task 1: Create mathx package structure
│  └─ Subtask: write_file 'mathx/__init__.py' with add() and multiply()
│     └─ Action: write_file(path="mathx/__init__.py", content="...")
│
├─ Task 2: Add tests
│  └─ Subtask: write_file 'tests/test_mathx.py' with tests
│     └─ Action: write_file(path="tests/test_mathx.py", content="...")
│
├─ Task 3: Add configuration
│  └─ Subtask: write_file 'pyproject.toml' with pytest+ruff config
│     └─ Action: write_file(path="pyproject.toml", content="...")
│
└─ Task 4: Verify quality
   ├─ Subtask: run_cmd ['ruff', 'check', '.']
   │  └─ Action: run_cmd(cmd=["ruff", "check", "."])
   └─ Subtask: run_cmd ['pytest', 'tests/', '-q']
      └─ Action: run_cmd(cmd=["pytest", "tests/", "-q"])
```

### Key Features

**1. Automatic Task Progression**
- Agent completes subtasks and automatically advances to the next one
- No manual coordination needed - the context manager handles the flow
- Example: After creating `mathx/__init__.py`, automatically moves to creating tests

**2. Loop Detection & Recovery**
- Detects when the same action is attempted multiple times
- Identifies alternating patterns (A→B→A→B)
- Automatically blocks looping actions and advances to next subtask
- Prevents the agent from getting stuck indefinitely

**3. Crash Recovery**
- Full state persists to `.agent_context/state.json`
- Agent can resume from exact subtask after interruption
- No need to restart from beginning
- Idempotent operations ensure safe retries

**4. Compact Context**
- LLM only sees the current branch of the task tree
- Reduces token usage by 60-80% compared to full message history
- Shows: Current Goal → Active Task → Active Subtask → Recent Actions
- Hides completed tasks and future tasks to maintain focus

**5. Need-to-Know Principle**
- Each level only sees relevant parent context
- Subtasks don't see sibling subtasks
- Actions don't see unrelated tasks
- Mimics how humans organize work mentally

### Context Manager State

The `.agent_context/` directory contains:

- **`state.json`** - Complete hierarchical state with all tasks, subtasks, and actions
- **`history.jsonl`** - Append-only log of all actions (JSONL format)
- **`loops.json`** - Record of detected loop patterns for analysis

### Example: Crash Recovery

```bash
# Session 1: Agent runs and crashes at task 2
$ python agent_enhanced.py "Create mathx package..."
[log] Created mathx/__init__.py
[log] Advanced to task: Add tests
# ... crash ...

# Session 2: Agent resumes exactly where it left off
$ python agent_enhanced.py "Create mathx package..."
[log] Resuming existing task hierarchy
[log] Current task: Add tests (in_progress)
[log] Subtask: write_file 'tests/test_mathx.py'
# ... continues from task 2 ...
```

### Comparison: Basic vs Enhanced Agent

| Feature | Basic Agent (`agent.py`) | Enhanced Agent (`agent_enhanced.py`) |
|---------|--------------------------|--------------------------------------|
| Context Structure | Flat message list | Hierarchical tree (Goal→Task→Subtask→Action) |
| Task Tracking | Manual checklist in messages | Automatic hierarchical state machine |
| Loop Detection | Simple deduplication (count > 3) | Pattern detection (repeats, alternating) |
| Crash Recovery | Parse ledger + status.txt | Load full state from state.json |
| Context Size | Last 12 messages (~2-3KB) | Current branch only (~500B-1KB) |
| Task Progression | Manual prompting | Automatic advancement |
| State Persistence | status.txt (plaintext summary) | state.json (complete structured state) |
| Completion Tracking | Probe-based heuristics | Explicit subtask status flags |

### Integration Guide

See `agent_integration.py` for:
- Complete integration examples
- Migration guide from basic agent
- API documentation for context manager
- Example agent loops with hierarchical context

## Performance Optimizations

The quality agent (`agent_quality.py`) includes comprehensive performance optimizations while maintaining gpt-oss:20b code quality:

### Optimization Highlights

| Optimization | Savings | Description |
|--------------|---------|-------------|
| LLM Warm-up | 9,200ms first call | Pre-warm model on startup (98.4% reduction) |
| Probe Caching | 250-350ms/round | Cache results for 3s, invalidate on file writes |
| Parallel Execution | 150ms/probe | Run ruff + pytest concurrently |
| Smart Skipping | 280ms | Skip pytest if no test directory |

### Speed Comparison

| Agent | Avg Round | Total (10 rounds) | Speedup |
|-------|-----------|-------------------|---------|
| Baseline (agent_enhanced.py) | 1000ms | 10.0s | 1.0x |
| **Optimized (agent_quality.py)** | **580ms** | **5.8s** | **1.7x faster** |

### Detailed Documentation

- `OPTIMIZATION_SUMMARY.md` - Complete optimization analysis
- `LLM_WARMUP_FINDINGS.md` - LLM warm-up deep dive (9.2s savings)
- `FINAL_PERFORMANCE_COMPARISON.md` - Full performance comparison
- `profile_*.py` - Profiling tools

## Configuration

Set the model via environment variable:

```bash
# PowerShell
$env:OLLAMA_MODEL = "gpt-oss:20b"

# Bash
export OLLAMA_MODEL="gpt-oss:20b"
```

Configuration in agent files:

- `MODEL` - Ollama model tag (default: "gpt-oss:20b")
- `TEMP` - Temperature for model (0.2 for focused outputs)
- `MAX_ROUNDS` - Hard cap on agent loop iterations (24)
- `HISTORY_KEEP` - Number of recent messages to retain in basic agent (12)
- `SAFE_BIN` - Allowed commands for safety: `{"python", "pytest", "ruff", "pip"}`

## Notes

- The agent writes a lightweight ledger to `agent_ledger.log` and logs to `agent.log`.
- The repository is intentionally minimal to keep the agent loop fast and easy to follow.

**Agent Context**
- Local‑first and crash‑resilient: the agent is expected to crash or be stopped at any time. Design for idempotency and fast rehydration from plaintext. Prefer simple files over databases so humans can inspect and edit status.
- Plaintext status first: keep a compact, human‑readable status that captures the overarching goal, current status, active subtasks, and next actions. Use this as the single source of truth for resuming work.
- Short timeboxes: run in small, bounded sessions (e.g., 30–120 seconds or a handful of rounds). At each boundary, persist status, compact context, and reflect before continuing.

**Status Storage**
- Primary artifacts:
  - `agent_ledger.log` — append‑only trace of writes and commands for audit and recovery.
  - `agent.log` — human‑readable runtime log for quick inspection.
  - Optional `status.txt` — compact snapshot to drive resumability; updated at round boundaries.
- Suggested `status.txt` format (plaintext, one concept per line):

```
Goal: <single sentence>
Status: <green/yellow/red + brief reason>
Active: <1–3 key subtasks>
Next: <up to 4 concrete next actions>
Notes: <freeform, compact, decisions/assumptions>
```

**Context Compaction**
- Keep only small, high‑signal context for the model:
  - Probe real state first (files exist, tests pass) to avoid hallucinating.
  - Summarize ledger into a 1–2 line recap (last writes/commands/errors).
  - Include current `status.txt` snapshot instead of long histories.
  - Deduplicate repeated tool calls and compress equivalent steps into a checklist.
- Bound context size by rounds and characters; prune oldest assistant/user exchanges first, keep the current plan and latest tool outputs.

**Reflection Loop**
- At each round boundary:
  1) Probe: verify end‑state quickly (ruff/pytest, key files).
  2) Compact: synthesize a tiny recap + next checklist.
  3) Reflect: compare plan vs. results; adjust next steps or stop if done.
  4) Persist: write `status.txt` and append to `agent_ledger.log`.

**Crash Recovery**
- On start, reconstruct state without the model:
  - Probe filesystem and tests.
  - Read latest `status.txt` if present; otherwise compress the tail of `agent_ledger.log` into a one‑line summary.
  - Rebuild a minimal checklist (e.g., create missing files → lint → test) and proceed.
- Aim for idempotent actions so retries are safe.

**Short Timebox Testing**
- Use brief, repeatable runs to gauge behavior and stability:
  - Set conservative constants in `agent.py` (e.g., `MAX_ROUNDS`, `HISTORY_KEEP`, `TEMP`).
  - Run targeted goals and measure: rounds taken, files touched, test outcomes, error rate.
  - Example: `python agent.py "Create mathx.add, add tests, run ruff and pytest."`
  - Validate after each run: `ruff check .` then `pytest -q`.
  - Inspect `agent_ledger.log` for repeated or low‑value steps; tighten compaction rules if needed.

**Pseudocode (Context Manager)**
```
def step():
    state = probe_state()  # files, ruff, pytest
    if state.pytest_ok:
        persist_status(goal, "green", active=[], next=[], notes="tests pass")
        return DONE

    recap = ledger_compact_tail(max_lines=60)  # tiny, deduped
    status = read_status_txt_or_default(goal)

    plan = plan_next(state)  # concrete, small checklist
    messages = compact_messages(system, status, recap, plan)

    msg = llm(messages, tools=TOOL_SPECS)
    for call in msg.tool_calls:
        out = dispatch(call)
        append_ledger(call, out)

    reflect = compare(plan, observed=outcomes_from_ledger_tail())
    persist_status(goal, reflect.status, reflect.active, reflect.next, reflect.notes)
    return CONTINUE
```

This approach keeps the model context tiny, leans on real system probes, and ensures the agent can recover quickly after interruptions while making steady, test‑driven progress.
