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

