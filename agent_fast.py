#!/usr/bin/env python3
# agent_fast.py — optimized agent with 5x performance improvements
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from ollama import chat
from context_manager import ContextManager, Subtask, Task

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------
# Config (OPTIMIZED FOR SPEED)
# ----------------------------
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")  # 18x faster than gpt-oss:20b!
TEMP = 0.2
MAX_ROUNDS = 24
LOGFILE = "agent_fast.log"
SAFE_BIN = {"python", "pytest", "ruff", "pip"}

# Performance optimizations
PROBE_CACHE_TTL = 3.0  # Cache probe results for 3 seconds
PARALLEL_PROBES = True  # Run ruff + pytest in parallel
SKIP_PROBE_IF_NO_WRITES = True  # Don't re-probe if no files written

# ----------------------------
# Logging
# ----------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")

# ----------------------------
# Probe Cache (Optimization #2)
# ----------------------------
class ProbeCache:
    """Cache probe results to avoid redundant ruff/pytest runs."""

    def __init__(self):
        self.last_probe_result: dict[str, Any] | None = None
        self.last_probe_time: float = 0
        self.last_write_time: float = 0
        self.files_written_since_probe: set[str] = set()

    def record_write(self, path: str) -> None:
        """Record that a file was written."""
        self.last_write_time = time.time()
        self.files_written_since_probe.add(path)

    def should_probe(self) -> bool:
        """Check if we should re-probe or use cache."""
        if self.last_probe_result is None:
            return True  # Never probed

        if SKIP_PROBE_IF_NO_WRITES and not self.files_written_since_probe:
            # No writes since last probe, use cache
            time_since_probe = time.time() - self.last_probe_time
            if time_since_probe < PROBE_CACHE_TTL:
                log(f"  ↻ Using cached probe (age: {time_since_probe:.1f}s)")
                return False

        return True

    def update(self, result: dict[str, Any]) -> None:
        """Update cache with new probe result."""
        self.last_probe_result = result
        self.last_probe_time = time.time()
        self.files_written_since_probe.clear()

    def get(self) -> dict[str, Any]:
        """Get cached probe result."""
        return self.last_probe_result or {}

# Global cache
probe_cache = ProbeCache()

# ----------------------------
# Probe state (OPTIMIZED)
# ----------------------------
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
    """Check current state (with caching and parallel execution)."""
    # Check cache first
    if not probe_cache.should_probe():
        return probe_cache.get()

    t0 = time.perf_counter()

    state: dict[str, Any] = {
        "pkg_exists": _exists(TARGET_FILES["pkg"]),
        "tests_exist": _exists(TARGET_FILES["tests"]),
        "pyproject_exists": _exists(TARGET_FILES["pyproject"]),
        "ruff_ok": None,
        "pytest_ok": None,
        "ruff_err": "",
        "pytest_err": "",
    }

    # Optimization: Run ruff and pytest in parallel
    def run_ruff():
        return run_cmd(["ruff", "check", ".", "--select", "E,F"], timeout_sec=10)

    def run_pytest():
        # Skip pytest if tests don't exist yet
        if not state["tests_exist"]:
            return {"returncode": 5, "stderr": "tests/ not found (skipped)", "stdout": ""}
        return run_cmd(["pytest", "tests/", "-q", "-x"], timeout_sec=15)  # Fail fast

    if PARALLEL_PROBES:
        with ThreadPoolExecutor(max_workers=2) as executor:
            ruff_future = executor.submit(run_ruff)
            pytest_future = executor.submit(run_pytest)

            # Get results
            ruff_rc = ruff_future.result()
            pytest_rc = pytest_future.result()
    else:
        # Sequential fallback
        ruff_rc = run_ruff()
        pytest_rc = run_pytest()

    # Process ruff result
    if "returncode" in ruff_rc:
        state["ruff_ok"] = (ruff_rc["returncode"] == 0)
        if ruff_rc["returncode"] != 0:
            state["ruff_err"] = (ruff_rc.get("stderr") or ruff_rc.get("stdout") or "")[:200]
    else:
        state["ruff_ok"] = False
        state["ruff_err"] = (ruff_rc.get("error") or "")[:200]

    # Process pytest result
    if "returncode" in pytest_rc:
        state["pytest_ok"] = (pytest_rc["returncode"] == 0)
        if pytest_rc["returncode"] != 0:
            state["pytest_err"] = (pytest_rc.get("stderr") or pytest_rc.get("stdout") or "")[:200]
    else:
        state["pytest_ok"] = False
        state["pytest_err"] = (pytest_rc.get("error") or "")[:200]

    elapsed = (time.perf_counter() - t0) * 1000
    log(f"  ⚡ probe_state: {elapsed:.1f}ms")

    # Update cache
    probe_cache.update(state)

    return state

# ----------------------------
# Tools (with probe cache integration)
# ----------------------------
def list_dir(path: str | None = ".", **kwargs) -> list[str]:
    """List files (non-recursive)."""
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
    # Record write for probe cache
    probe_cache.record_write(path)
    return f"Wrote {len(content)} chars to {path}"

def run_cmd(cmd: list[str], timeout_sec: int = 60) -> dict[str, Any]:
    """Run a whitelisted command."""
    if not cmd or cmd[0] not in SAFE_BIN:
        err = f"Command not allowed: {cmd!r}. Use only {sorted(SAFE_BIN)}."
        return {"error": err}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        return {
            "returncode": p.returncode,
            "stdout": p.stdout[-10_000:],  # Truncate output
            "stderr": p.stderr[-10_000:],
        }
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_sec}s"}
    except Exception as e:
        return {"error": str(e)}

TOOLS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "run_cmd": run_cmd,
}

def tool_specs() -> list[dict[str, Any]]:
    """Minimal tool specs for speed."""
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
        spec("write_file", "Write/create a file", {
            "path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
        }),
        spec("run_cmd", "Run command (python/pytest/ruff/pip)", {
            "cmd": {"type": "array", "items": {"type": "string"}, "required": True},
        }),
        spec("read_file", "Read file", {
            "path": {"type": "string", "required": True},
        }),
        spec("list_dir", "List directory", {
            "path": {"type": "string"},
        }),
    ]

# ----------------------------
# Enhanced Agent (same as agent_enhanced.py but optimized)
# ----------------------------
class FastAgent:
    """Fast agent with hierarchical context + performance optimizations."""

    def __init__(self, goal: str) -> None:
        self.ctx = ContextManager()
        self.ctx.load_or_init(goal)
        self._initialize_tasks_from_goal(goal)

    def _initialize_tasks_from_goal(self, goal: str) -> None:
        """Create initial task hierarchy if not resuming."""
        if self.ctx.state.goal and self.ctx.state.goal.tasks:
            log("↻ Resuming existing task hierarchy")
            return

        log("✦ Creating new task hierarchy")
        if "mathx" in goal.lower():
            tasks = [
                Task(
                    description="Create mathx package structure",
                    subtasks=[
                        Subtask(description="write_file 'mathx/__init__.py' with add(a,b) and multiply(a,b)"),
                    ],
                ),
                Task(
                    description="Add tests",
                    subtasks=[
                        Subtask(description="write_file 'tests/test_mathx.py' with tests for add and multiply"),
                    ],
                ),
                Task(
                    description="Add pyproject.toml configuration",
                    subtasks=[
                        Subtask(description="write_file 'pyproject.toml' with pytest+ruff config"),
                    ],
                ),
                Task(
                    description="Verify quality",
                    subtasks=[
                        Subtask(description="run_cmd ['ruff', 'check', '.']"),
                        Subtask(description="run_cmd ['pytest', 'tests/', '-q']"),
                    ],
                ),
            ]

            for task in tasks:
                task.parent_goal = goal
                self.ctx.state.goal.tasks.append(task)

            if tasks:
                tasks[0].status = "in_progress"
                tasks[0].subtasks[0].status = "in_progress"

            self.ctx._save_state()

    def get_current_subtask(self) -> Subtask | None:
        task = self.ctx._get_current_task()
        if not task:
            return None
        return task.active_subtask()

    def advance_workflow(self, probe: dict[str, Any]) -> bool:
        if probe.get("pytest_ok") and probe.get("ruff_ok"):
            if self.ctx.state.goal:
                self.ctx.state.goal.status = "completed"
                self.ctx._save_state()
            return False

        task = self.ctx._get_current_task()
        if not task:
            return False

        subtask = task.active_subtask()
        if not subtask:
            if self.ctx.advance_to_next_subtask():
                return True
            else:
                task.status = "completed"
                self.ctx.state.current_task_idx += 1
                self.ctx._save_state()

                next_task = self.ctx._get_current_task()
                if next_task:
                    next_task.status = "in_progress"
                    if next_task.subtasks:
                        next_task.subtasks[0].status = "in_progress"
                    self.ctx._save_state()
                    return True
                else:
                    return False

        return True

    def execute_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self.ctx.record_action(name, args, result="pending"):
            return {
                "error": "Action blocked: loop detected",
                "suggestion": "Try a different approach.",
            }

        try:
            tool_fn = TOOLS.get(name)
            if not tool_fn:
                result = {"error": f"Unknown tool: {name}"}
                self.ctx.action_history[-1].result = "error"
                self.ctx.action_history[-1].error_msg = result["error"]
            else:
                output = tool_fn(**args)
                result = {"result": output}
                self.ctx.action_history[-1].result = "success"

        except Exception as e:
            result = {"error": str(e)}
            self.ctx.action_history[-1].result = "error"
            self.ctx.action_history[-1].error_msg = str(e)

        self.ctx._save_state()
        return result

    def should_mark_subtask_complete(self, probe: dict[str, Any]) -> tuple[bool, str]:
        subtask = self.get_current_subtask()
        if not subtask:
            return False, ""

        desc = subtask.description.lower()

        if "mathx/__init__.py" in desc:
            if probe.get("pkg_exists"):
                return True, "mathx/__init__.py created"
        elif "test_mathx.py" in desc:
            if probe.get("tests_exist"):
                return True, "tests/test_mathx.py created"
        elif "pyproject.toml" in desc:
            if probe.get("pyproject_exists"):
                return True, "pyproject.toml created"
        elif "ruff" in desc:
            if probe.get("ruff_ok"):
                return True, "ruff check passed"
        elif "pytest" in desc:
            if probe.get("pytest_ok"):
                return True, "pytest passed"

        return False, ""

    def get_llm_prompt(self, probe: dict[str, Any]) -> str:
        """Compact prompt (optimized for speed)."""
        subtask = self.get_current_subtask()
        if not subtask:
            return "All subtasks complete."

        # Minimal context
        lines = [f"Task: {subtask.description}"]

        # Add state only if relevant
        if not probe.get("pkg_exists"):
            lines.append("Note: mathx/__init__.py does not exist")
        if not probe.get("tests_exist"):
            lines.append("Note: tests/test_mathx.py does not exist")

        return "\n".join(lines)

# ----------------------------
# Main loop (OPTIMIZED)
# ----------------------------
SYSTEM_PROMPT = "You are a coding agent. Complete the current task using tools. Be direct and efficient."

def main() -> None:
    goal = "Create mathx package with add(a,b) and multiply(a,b), add tests, run ruff and pytest."
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])

    log(f"🚀 Starting FAST agent (model: {MODEL})")
    log(f"Goal: {goal}")

    agent = FastAgent(goal)
    total_start = time.time()

    for round_no in range(1, MAX_ROUNDS + 1):
        round_start = time.perf_counter()

        # 1. Probe current state (with caching)
        probe = probe_state()
        agent.ctx.update_probe_state(probe)

        # 2. Check completion and advance
        should_complete, reason = agent.should_mark_subtask_complete(probe)
        if should_complete:
            log(f"✓ Subtask complete: {reason}")
            agent.ctx.mark_subtask_complete(success=True)
            if not agent.advance_workflow(probe):
                elapsed = time.time() - total_start
                log(f"🎉 DONE in {elapsed:.1f}s ({round_no} rounds)")
                print(f"\n✅ Goal achieved in {elapsed:.1f}s ({round_no} rounds)")
                return

        if not agent.advance_workflow(probe):
            elapsed = time.time() - total_start
            log(f"🎉 DONE in {elapsed:.1f}s ({round_no} rounds)")
            print(f"\n✅ Goal achieved in {elapsed:.1f}s ({round_no} rounds)")
            return

        subtask = agent.get_current_subtask()
        if not subtask:
            continue

        # 3. Build compact prompt
        llm_prompt = agent.get_llm_prompt(probe)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": llm_prompt},
        ]

        # 4. LLM call
        llm_start = time.perf_counter()
        resp = chat(
            model=MODEL,
            messages=messages,
            tools=tool_specs(),
            options={"temperature": TEMP},
            stream=False,
        )
        llm_time = (time.perf_counter() - llm_start) * 1000

        calls = resp["message"].get("tool_calls") or []

        if calls:
            for call in calls:
                name = call["function"]["name"]
                args_str = call["function"].get("arguments", "{}")
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})

                result = agent.execute_tool_call(name, args)

                if "error" in result and "loop detected" in result.get("error", ""):
                    agent.ctx.mark_subtask_complete(success=False, reason="Loop detected")
                    if not agent.advance_workflow(probe):
                        print("\n⚠ Blocked by loops")
                        return

        round_time = (time.perf_counter() - round_start) * 1000
        log(f"Round {round_no}: {round_time:.0f}ms (LLM: {llm_time:.0f}ms)")

    elapsed = time.time() - total_start
    log(f"⏱ Stopped at MAX_ROUNDS after {elapsed:.1f}s")
    print(f"\n⏱ Hit MAX_ROUNDS after {elapsed:.1f}s")

if __name__ == "__main__":
    main()
