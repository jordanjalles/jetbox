# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Jetbox is a minimal local-first coding agent that runs with Ollama on Windows. The agent operates in short timeboxes, is designed to be crash-resilient, and follows a verify-first, work-backward approach to completing tasks.

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
```

### Configuration
Set the Ollama model via environment variable:
```bash
# PowerShell
$env:OLLAMA_MODEL = "gpt-oss:20b"

# Bash
export OLLAMA_MODEL="gpt-oss:20b"
```

## Architecture

### Agent Design Philosophy

**Local-first and crash-resilient**: The agent expects to crash or be stopped at any time. All designs prioritize idempotency and fast rehydration from plaintext logs. No databases are used - everything is human-inspectable files.

**Verify-first, work-backward**: Before making any LLM calls, the agent probes the current state (file existence, ruff output, pytest output) using `probe_state()` in agent.py:117. From this verified state, it generates a minimal backward-chaining checklist via `plan_next()` in agent.py:172.

**Short timeboxes**: Runs are bounded by `MAX_ROUNDS` (default 24 rounds). Each round boundary triggers a probe → compact → reflect → persist cycle.

### Core Components

**agent.py** (main agent loop):
- `probe_state()` - Verifies filesystem state, runs ruff and pytest without asking the LLM
- `plan_next()` - Generates concrete backward-chaining checklist from current state
- `dispatch()` - Executes tool calls with deduplication (skips after 3 identical calls)
- `_prune_history()` - Keeps message context compact (last `HISTORY_KEEP` messages + system + last user)
- `_ledger_summary()` - Compresses `agent_ledger.log` into a 1-2 line recap
- `_status_lines()` - Formats status snapshot as plaintext (Goal/Status/Active/Next/Notes)

**Tool whitelist** (agent.py:21): Only `python`, `pytest`, `ruff`, and `pip` commands are allowed for Windows safety. All other commands are rejected.

**Status artifacts**:
- `agent_ledger.log` - Append-only trace of WRITE/CMD/ERROR/TRIED actions (audit trail)
- `agent.log` - Human-readable runtime log
- `status.txt` - Compact snapshot (Goal/Status/Active/Next/Notes format) for crash recovery

### Context Management Strategy

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

## Key Constants (agent.py:12-22)

- `MODEL` - Ollama model tag (default: "gpt-oss:20b", override with `OLLAMA_MODEL` env var)
- `TEMP` - Temperature for model (0.2 for focused outputs)
- `MAX_ROUNDS` - Hard cap on agent loop iterations (24)
- `HISTORY_KEEP` - Number of recent messages to retain in context (12)
- `SAFE_BIN` - Whitelisted commands: `{"python", "pytest", "ruff", "pip"}`

## Target Files (agent.py:120-124)

The agent probes these specific files to determine task completion:
- `mathx/__init__.py` - Package entry point
- `tests/test_mathx.py` - Test file
- `pyproject.toml` - Build and tool configuration

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
1. Reads `status.txt` if present (last known state)
2. Probes current filesystem and tool outputs
3. Reconstructs minimal checklist from probe results
4. Continues from where it left off

If status.txt is missing, the agent falls back to compressing the tail of `agent_ledger.log`.

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
