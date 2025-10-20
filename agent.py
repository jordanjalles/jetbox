# agent.py — tiny local coding agent for Ollama (gpt-oss) on Windows
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ollama import chat

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------
# Config
# ----------------------------
MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")  # set to your tag
TEMP = 0.2
MAX_ROUNDS = 24                 # hard cap so we never run forever
HISTORY_KEEP = 12               # keep last N messages (plus system + last user)
LOGFILE = "agent.log"
LEDGER = Path("agent_ledger.log")
SAFE_BIN = {"python", "pytest", "ruff", "pip"}  # windows-safe commands only
STATUS_FILE = Path("status.txt")


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError:
        return ""


def _write_status_snapshot(snapshot: str) -> None:
    try:
        text = snapshot if snapshot.endswith("\n") else f"{snapshot}\n"
        STATUS_FILE.write_text(text, encoding="utf-8")
        _ledger_append("WRITE", str(STATUS_FILE))
    except Exception as exc:
        _ledger_append("ERROR", f"status_write_failed: {exc}")


def _status_lines(
    goal: str, state: dict[str, Any], checklist: list[str], summary: str
) -> str:
    def classify_status() -> str:
        if state.get("pytest_ok") is True:
            return "green: tests pass"
        missing = [name for name, exists in (
            ("pkg", state.get("pkg_exists")),
            ("tests", state.get("tests_exist")),
            ("pyproject", state.get("pyproject_exists")),
        ) if not exists]
        if missing:
            return f"red: missing {', '.join(missing)}"
        if state.get("pytest_ok") is False:
            return "red: pytest failing"
        if state.get("ruff_ok") is False:
            return "yellow: fix lint"
        return "yellow: in progress"

    def choose_active(items: list[str]) -> str:
        active = [s for s in items if s.lower().startswith("write_file")] or items[:2]
        return " | ".join(active[:3]) or "None"

    def choose_next(items: list[str]) -> str:
        return " | ".join(items[:4]) or "None"

    notes_parts: list[str] = []
    if state.get("ruff_err"):
        notes_parts.append(f"ruff: {state['ruff_err'][:80]}")
    if state.get("pytest_err"):
        notes_parts.append(f"pytest: {state['pytest_err'][:80]}")
    if summary:
        notes_parts.append(summary[:160])
    notes = "; ".join(notes_parts) or ""

    lines = [
        f"Goal: {goal}",
        f"Status: {classify_status()}",
        f"Active: {choose_active(checklist)}",
        f"Next: {choose_next(checklist)}",
        f"Notes: {notes or 'None'}",
    ]
    return "\n".join(lines)

# ----------------------------
# Logging / Ledger
# ----------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[log] {msg}")

def _ledger_append(kind: str, detail: str) -> None:
    # kind: WRITE / CMD / ERROR / TRIED
    line = f"{kind}\t{detail.replace(chr(10),' ')[:400]}\n"
    if LEDGER.exists():
        LEDGER.write_text(LEDGER.read_text(encoding="utf-8") + line, encoding="utf-8")
    else:
        LEDGER.write_text(line, encoding="utf-8")

def _ledger_summary(max_lines: int = 80) -> str:
    if not LEDGER.exists():
        return "No prior steps."
    lines = LEDGER.read_text(encoding="utf-8").splitlines()[-max_lines:]
    created = [line.split("\t", 1)[1] for line in lines if line.startswith("WRITE")]
    cmds = [line.split("\t", 1)[1] for line in lines if line.startswith("CMD")]
    errs = [line.split("\t", 1)[1] for line in lines if line.startswith("ERROR")]
    tried = [line.split("\t", 1)[1] for line in lines if line.startswith("TRIED")]
    parts = []
    if created:
        parts.append("Files: " + ", ".join(dict.fromkeys(created))[:600])
    if cmds:
        parts.append("Cmds: " + ", ".join(dict.fromkeys(cmds))[:600])
    if errs:
        parts.append("Errors: " + " | ".join(errs[-5:])[:600])
    if tried:
        parts.append("Tried: " + ", ".join(dict.fromkeys(tried))[:600])
    return "; ".join(parts) or "Steps exist but nothing notable."

# ---- Goal-state probe + plan (verify-first, work-backward) ----

TARGET_FILES = {
    "pkg": Path("mathx/__init__.py"),
    "tests": Path("tests/test_mathx.py"),
    "pyproject": Path("pyproject.toml"),
}

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def probe_state() -> dict[str, Any]:
    """Check current end-state and key artifacts without asking the model."""
    state: dict[str, Any] = {
        "pkg_exists": _exists(TARGET_FILES["pkg"]),
        "tests_exist": _exists(TARGET_FILES["tests"]),
        "pyproject_exists": _exists(TARGET_FILES["pyproject"]),
        "ruff_ok": None,
        "pytest_ok": None,
        "ruff_err": "",
        "pytest_err": "",
    }

    # Try ruff
    rc = run_cmd(["ruff", "check", "."], timeout_sec=60)
    if "returncode" in rc:
        state["ruff_ok"] = (rc["returncode"] == 0)
        if rc["returncode"] != 0:
            state["ruff_err"] = (rc.get("stderr") or rc.get("stdout") or "")[:300]
    else:
        state["ruff_ok"] = False
        state["ruff_err"] = (rc.get("error") or "")[:300]

    # Try pytest (quiet)
    rc = run_cmd(["pytest", "-q"], timeout_sec=90)
    if "returncode" in rc:
        state["pytest_ok"] = (rc["returncode"] == 0)
        if rc["returncode"] != 0:
            state["pytest_err"] = (rc.get("stderr") or rc.get("stdout") or "")[:300]
    else:
        state["pytest_ok"] = False
        state["pytest_err"] = (rc.get("error") or "")[:300]

    # Log a one-line summary for resilience across crashes
    _ledger_append("TRIED", f"probe ruff={state['ruff_ok']} pytest={state['pytest_ok']} files="
                  f"{'pkg' if state['pkg_exists'] else ''}/"
                  f"{'tests' if state['tests_exist'] else ''}/"
                  f"{'pyproject' if state['pyproject_exists'] else ''}")

    return state

def plan_next(state: dict[str, Any]) -> list[str]:
    """Return a minimal checklist based on current state (backward from the goal)."""
    # If tests pass, we are done.
    if state.get("pytest_ok") is True:
        return ["DONE (pytest passes)"]

    steps: list[str] = []
    # Ensure artifacts exist before re-running tools
    if not state.get("pkg_exists"):
        steps.append("write_file 'mathx/__init__.py' with add(a,b)")
    if not state.get("tests_exist"):
        steps.append("write_file 'tests/test_mathx.py' with 3 tests for add(a,b)")
    if not state.get("pyproject_exists"):
        steps.append("write_file 'pyproject.toml' with pytest+ruff config")

    # Then attempt ruff, then pytest
    steps.append("run_cmd ['ruff','check','.']")
    steps.append("run_cmd ['pytest','-q']")
    return steps


# ----------------------------
# Tools (tolerant + safe)
# ----------------------------
def list_dir(path: str | None = ".", **kwargs) -> list[str]:
    """List files (non-recursive). Ignores extra args; treats '' or None as '.' on Windows."""
    p = path or "."
    try:
        return sorted(os.listdir(p))
    except FileNotFoundError as e:
        return [f"__error__: {e}"]

def read_file(path: str, max_bytes: int = 200_000) -> str:
    """Read a text file (truncated)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read(max_bytes)

def write_file(path: str, content: str, create_dirs: bool = True) -> str:
    """Write/overwrite a text file."""
    if create_dirs:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    _ledger_append("WRITE", path)
    return f"Wrote {len(content)} chars to {path}"

def run_cmd(cmd: list[str], timeout_sec: int = 60) -> dict[str, Any]:
    """Run a whitelisted command: first token must be in SAFE_BIN (Windows-friendly)."""
    if not cmd or cmd[0] not in SAFE_BIN:
        err = f"Command not allowed: {cmd!r}. Use only {sorted(SAFE_BIN)}."
        _ledger_append("ERROR", err)
        return {"error": err}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        out = {
            "returncode": p.returncode,
            "stdout": p.stdout[-50_000:],
            "stderr": p.stderr[-50_000:],
        }
        _ledger_append("CMD", f"{cmd} -> rc={p.returncode}")
        if p.returncode != 0:
            _ledger_append("ERROR", f"run_cmd rc={p.returncode}: {p.stderr[:200]}")
        return out
    except subprocess.TimeoutExpired:
        err = f"timeout after {timeout_sec}s"
        _ledger_append("ERROR", f"run_cmd timeout: {cmd}")
        return {"error": err}
    except Exception as e:
        _ledger_append("ERROR", f"run_cmd exception: {e}")
        return {"error": str(e)}

TOOLS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "run_cmd": run_cmd,
}

def tool_specs() -> list[dict[str, Any]]:
    def spec(fn: str, desc: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": fn,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": [k for k, v in params.items() if v.get("required")],
                },
            },
        }
    return [
        spec("list_dir", "List files (non-recursive).", {"path": {"type": "string"}}),
        spec("read_file", "Read a text file (truncated).", {
            "path": {"type": "string", "required": True},
            "max_bytes": {"type": "number"},
        }),
        spec("write_file", "Write/overwrite a text file.", {
            "path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
            "create_dirs": {"type": "boolean"},
        }),
        spec("run_cmd", "Run a safe command (python/pytest/ruff/pip).", {
            "cmd": {"type": "array", "items": {"type": "string"}, "required": True},
            "timeout_sec": {"type": "number"},
        }),
    ]

# ----------------------------
# Anti-loop: dedupe repeated tool calls (name + normalized args)
# ----------------------------
SEEN: defaultdict[tuple[str, str], int] = defaultdict(int)

def _norm_args(args: Any) -> str:
    try:
        return json.dumps(args if isinstance(args, dict) else json.loads(args or "{}"), sort_keys=True)
    except Exception:
        return "{}"

def dispatch(call: dict[str, Any]) -> dict[str, Any]:
    name = call["function"]["name"]
    args = call["function"].get("arguments", "{}")
    norm = _norm_args(args)
    SEEN[(name, norm)] += 1

    # record attempt
    if name == "write_file":
        try:
            path = (json.loads(args) if isinstance(args, str) else args or {}).get("path")
            _ledger_append("TRIED", f"write_file {path}")
        except Exception:
            _ledger_append("TRIED", "write_file ?")
    elif name == "run_cmd":
        try:
            cmd = (json.loads(args) if isinstance(args, str) else args or {}).get("cmd")
            _ledger_append("TRIED", f"run_cmd {cmd}")
        except Exception:
            _ledger_append("TRIED", "run_cmd ?")

    # Deduplicate pathological repeats
    if SEEN[(name, norm)] > 3:
        log(f"TOOL⚠ dedupe: {name} x{SEEN[(name, norm)]}")
        return {"note": "dedup-skipped", "tool": name}

    # Execute
    arg_str = args if isinstance(args, str) else json.dumps(args)
    log(f"TOOL→ {name} args={arg_str[:200].replace(chr(10),' ')}")
    fn = TOOLS.get(name)
    if not fn:
        _ledger_append("ERROR", f"unknown tool {name}")
        log(f"TOOL✖ unknown: {name}")
        return {"error": f"unknown tool {name}"}
    try:
        data = json.loads(args) if isinstance(args, str) else (args or {})
        out = fn(**data) if data else fn()  # type: ignore
        preview = f"{type(out).__name__}"
        if isinstance(out, list):
            preview = f"list(len={len(out)})"
        elif isinstance(out, dict):
            preview = f"dict(keys={list(out)[:6]})"
        log(f"TOOL✓ {name} → {preview}")
        return {"result": out}
    except Exception as e:
        _ledger_append("ERROR", f"{name} failed: {e}")
        log(f"TOOL✖ {name} error={e}")
        return {"error": str(e)}

# ----------------------------
# Context management (compact)
# ----------------------------
SYSTEM_PROMPT = (
    "You are a fully local coding agent on Windows.\n"
    "- Plan briefly, then use tools.\n"
    "- Read only what’s necessary.\n"
    "- Prefer run_cmd with python/pytest/ruff/pip.\n"
    "- If a command fails repeatedly, try a different approach.\n"
    "- Aim to finish, not to rewrite the same file endlessly."
)

def _prune_history(messages: list[dict[str, Any]], keep: int = HISTORY_KEEP) -> list[dict[str, Any]]:
    if not messages:
        return []
    # Keep first system + last user + last `keep`
    out: list[dict[str, Any]] = []
    if messages[0].get("role") == "system":
        out.append(messages[0])

    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is not None:
        out.append(messages[last_user_idx])

    tail = messages[-keep:]
    pruned_tail: list[dict[str, Any]] = []
    for m in tail:
        if m.get("role") == "tool":
            pruned_tail.append({
                "role": "tool",
                "content": '{"result":"(omitted)"}',
                **({"tool_call_id": m.get("tool_call_id")} if m.get("tool_call_id") else {})
            })
        else:
            pruned_tail.append(m)

    seen = set()
    final: list[dict[str, Any]] = []
    for m in out + pruned_tail:
        key = (m.get("role"), m.get("content"), m.get("tool_call_id"))
        if key in seen:
            continue
        seen.add(key)
        final.append(m)
    return final

# ----------------------------
# Main loop
# ----------------------------
def main() -> None:
    goal = "Create a tiny package 'mathx' with add(a,b), add tests, then run ruff and pytest."
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    prior_status = _safe_read(STATUS_FILE).strip()
    if prior_status:
        messages.append({"role": "assistant", "content": f"(status)\n{prior_status}"})

    for round_no in range(1, MAX_ROUNDS + 1):
        # 0) Verify end-state first (fast, local, no LLM)
        state = probe_state()
        summary = _ledger_summary(max_lines=60)
        if state.get("pytest_ok") is True:
            print("\n=== Agent Reply ===\nDONE — tests pass. Nothing to do.")
            return

        checklist = plan_next(state)
        checklist_text = " | ".join(checklist)

        # 1) Compact state + plan to keep context tiny and focused
        summary = _ledger_summary(max_lines=60)
        summary_msg = {
            "role": "assistant",
            "content": (
                f"(state) ruff_ok={state['ruff_ok']} pytest_ok={state['pytest_ok']} "
                f"files: pkg={state['pkg_exists']} tests={state['tests_exist']} pyproject={state['pyproject_exists']}"
            ),
        }
        plan_msg = {"role": "assistant", "content": f"(next) {checklist_text}"}

        if messages and messages[0].get("role") == "system":
            messages = [messages[0], summary_msg, plan_msg] + messages[1:]
        else:
            messages = [summary_msg, plan_msg] + messages

        # 2) Keep history bounded
        messages = _prune_history(messages, keep=HISTORY_KEEP)
        log(f"ROUND {round_no}: sending {len(messages)} msgs")

        # 3) Ask the model what to do next (with tools)
        t0 = time.time()
        resp = chat(
            model=MODEL,
            messages=messages,
            tools=tool_specs(),
            options={"temperature": TEMP},
            stream=False,
        )
        log(f"ROUND {round_no}: chat() {time.time() - t0:.2f}s")

        msg = resp["message"]
        calls = msg.get("tool_calls") or []
        if calls:
            names = ", ".join(c["function"]["name"] for c in calls)
            log(f"ROUND {round_no}: tool_calls → {names} (n={len(calls)})")

            for c in calls:
                # Execute tool
                try:
                    tool_result = dispatch(c)
                except Exception as e:
                    tool_result = {"error": f"dispatch-failed: {e}"}

                # Return tool result to model (preserve linkage)
                messages.append({
                    "role": "tool",
                    "tool_call_id": c.get("id"),
                    "content": json.dumps(tool_result),
                })

                # Gentle correction for Unix shells on Windows
                if c["function"]["name"] == "run_cmd":
                    try:
                        args = c["function"]["arguments"]
                        arg_obj = json.loads(args) if isinstance(args, str) else (args or {})
                        cmd0 = (arg_obj.get("cmd") or [None])[0]
                    except Exception:
                        cmd0 = None
                    if cmd0 and cmd0 not in SAFE_BIN:
                        messages.append({
                            "role": "assistant",
                            "content": (
                                "Environment: Windows. Use only run_cmd with "
                                f"{sorted(SAFE_BIN)}. For Python: ['python','-c','print(1)']."
                            ),
                        })

                # If a repeated call was skipped, push it forward
                if isinstance(tool_result, dict) and tool_result.get("note") == "dedup-skipped":
                    messages.append({
                        "role": "assistant",
                        "content": (
                            "Avoid repeating the same tool call. Next steps:\n"
                            "1) write_file 'tests/test_mathx.py' with 3 tests for add(a,b)\n"
                            "2) write_file 'pyproject.toml' with pytest+ruff config\n"
                            "3) run_cmd ['ruff','check','.']\n"
                            "4) run_cmd ['pytest','-q']\n"
                            "Return DONE when finished."
                        ),
                    })

            continue  # let the model observe the tool outputs and continue

        # No tool calls → final answer
        print("\n=== Agent Reply ===\n" + msg["content"])
        return

    # Hard stop
    print("\n[stopped] hit MAX_ROUNDS without a final answer.\n")


if __name__ == "__main__":
    main()
