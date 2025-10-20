# Jetbox

Tiny local coding agent + sample package, built to run with Ollama on Windows. It includes:

- A minimal agent (`agent.py`) that plans simple steps, runs safe local commands, and iterates via the Ollama Python API.
- A tiny demo package `mathx` with `add(a, b)` and tests in `tests/`.
- Dev tooling via `pytest` and `ruff` configured in `pyproject.toml`.
- A quick latency check script (`diag_speed.py`) for your Ollama model.

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
├─ agent.py            # Tiny local coding agent for Ollama
├─ diag_speed.py       # Quick timing test for Ollama responses
├─ mathx/
│  └─ __init__.py      # Demo: add(a, b)
├─ tests/
│  └─ test_mathx.py    # Pytest tests for mathx.add
└─ pyproject.toml      # pytest + ruff config
```

## Usage

1) Make sure Ollama is running and you have a local model (default env var `OLLAMA_MODEL` or the default in code `gpt-oss:20b`).

2) Run the agent with a goal:

```
python agent.py "Create a tiny package 'mathx' with add(a,b), add tests, then run ruff and pytest."
```

3) Run tests directly:

```
pytest -q
```

4) Lint with ruff:

```
ruff check .
```

5) Quick model latency check:

```
python diag_speed.py
```

## Configuration

- Set the model via environment variable before running:

```
$env:OLLAMA_MODEL = "gpt-oss:20b"   # PowerShell
# or
export OLLAMA_MODEL="gpt-oss:20b"   # bash
```

Within `agent.py`, tweak:

- `MODEL`, `TEMP`, `MAX_ROUNDS`, `HISTORY_KEEP`
- Allowed binaries for safety on Windows: `SAFE_BIN = {"python", "pytest", "ruff", "pip"}`

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
