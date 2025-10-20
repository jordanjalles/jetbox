#!/usr/bin/env python3
# agent_ultra.py — ultra-optimized agent with streaming, batching, and aggressive caching
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ollama import chat
from context_manager import ContextManager, Subtask, Task

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------
# Ultra-Optimized Config
# ----------------------------
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
TEMP = 0.1  # Lower temp for faster, more deterministic responses
MAX_ROUNDS = 24
LOGFILE = "agent_ultra.log"
SAFE_BIN = {"python", "pytest", "ruff", "pip"}

# Aggressive caching
PROBE_CACHE_TTL = 5.0  # Longer cache (5s)
ENABLE_STREAMING = True  # Stream LLM responses
BATCH_SIZE = 3  # Execute up to 3 tools in parallel
USE_MINIMAL_CONTEXT = True  # Ultra-compact prompts

# File tracking for incremental operations
file_mtimes: dict[str, float] = {}

# ----------------------------
# Logging with microsecond precision
# ----------------------------
def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {level:5s} {msg}\n")
    if level != "DEBUG":
        print(f"[{ts}] {msg}")

# ----------------------------
# Enhanced Probe Cache
# ----------------------------
class UltraProbeCache:
    """Enhanced probe cache with file mtime tracking."""

    def __init__(self):
        self.last_probe_result: dict[str, Any] | None = None
        self.last_probe_time: float = 0
        self.files_written: set[str] = set()
        self.file_mtimes: dict[str, float] = {}

    def record_write(self, path: str) -> None:
        self.files_written.add(path)
        try:
            self.file_mtimes[path] = Path(path).stat().st_mtime
        except Exception:
            pass

    def files_changed_since_probe(self) -> bool:
        """Check if any tracked files changed since last probe."""
        for path, old_mtime in self.file_mtimes.items():
            try:
                new_mtime = Path(path).stat().st_mtime
                if new_mtime > old_mtime:
                    return True
            except Exception:
                return True  # File deleted/missing = changed
        return False

    def should_probe(self) -> bool:
        if self.last_probe_result is None:
            return True

        age = time.time() - self.last_probe_time

        # Use cache if:
        # 1. No writes since last probe
        # 2. Cache is fresh (< TTL)
        # 3. No file changes detected
        if not self.files_written and age < PROBE_CACHE_TTL:
            if not self.files_changed_since_probe():
                log(f"Cache hit (age: {age:.1f}s)", "DEBUG")
                return False

        return True

    def update(self, result: dict[str, Any]) -> None:
        self.last_probe_result = result
        self.last_probe_time = time.time()
        self.files_written.clear()

        # Update mtimes for all tracked files
        for path in [TARGET_FILES["pkg"], TARGET_FILES["tests"], TARGET_FILES["pyproject"]]:
            try:
                if path.exists():
                    self.file_mtimes[str(path)] = path.stat().st_mtime
            except Exception:
                pass

    def get(self) -> dict[str, Any]:
        return self.last_probe_result or {}

ultra_cache = UltraProbeCache()

# ----------------------------
# Ultra-Fast Probe
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
    """Ultra-fast probe with aggressive caching."""
    if not ultra_cache.should_probe():
        return ultra_cache.get()

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

    # Incremental linting: only check files that exist
    files_to_check = []
    if state["pkg_exists"]:
        files_to_check.append("mathx/")
    if state["tests_exist"]:
        files_to_check.append("tests/")

    # Run ruff and pytest in parallel
    def run_ruff():
        if not files_to_check:
            return {"returncode": 0, "stdout": "", "stderr": ""}
        # Minimal ruff: only critical errors
        return run_cmd(["ruff", "check"] + files_to_check + ["--select", "E,F"], timeout_sec=5)

    def run_pytest():
        if not state["tests_exist"]:
            return {"returncode": 5, "stderr": "no tests", "stdout": ""}
        # Fail fast, minimal output
        return run_cmd(["pytest", "tests/", "-q", "-x", "--tb=no"], timeout_sec=10)

    with ThreadPoolExecutor(max_workers=2) as executor:
        ruff_future = executor.submit(run_ruff)
        pytest_future = executor.submit(run_pytest)

        ruff_rc = ruff_future.result()
        pytest_rc = pytest_future.result()

    # Process results
    if "returncode" in ruff_rc:
        state["ruff_ok"] = (ruff_rc["returncode"] == 0)
        if ruff_rc["returncode"] != 0:
            state["ruff_err"] = (ruff_rc.get("stderr") or ruff_rc.get("stdout") or "")[:150]
    else:
        state["ruff_ok"] = False
        state["ruff_err"] = (ruff_rc.get("error") or "")[:150]

    if "returncode" in pytest_rc:
        state["pytest_ok"] = (pytest_rc["returncode"] == 0)
        if pytest_rc["returncode"] != 0:
            state["pytest_err"] = (pytest_rc.get("stderr") or pytest_rc.get("stdout") or "")[:150]
    else:
        state["pytest_ok"] = False
        state["pytest_err"] = (pytest_rc.get("error") or "")[:150]

    elapsed = (time.perf_counter() - t0) * 1000
    log(f"Probe: {elapsed:.0f}ms", "DEBUG")

    ultra_cache.update(state)
    return state

# ----------------------------
# Optimized Tools
# ----------------------------
def list_dir(path: str | None = ".", **kwargs) -> list[str]:
    p = path or "."
    try:
        return sorted(os.listdir(p))
    except FileNotFoundError:
        return []

def read_file(path: str, max_bytes: int = 100_000) -> str:
    """Read with smaller limit."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except Exception as e:
        return f"Error: {e}"

def write_file(path: str, content: str, create_dirs: bool = True) -> str:
    if create_dirs:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    ultra_cache.record_write(path)
    return f"Wrote {path}"

def run_cmd(cmd: list[str], timeout_sec: int = 30) -> dict[str, Any]:
    """Shorter timeout, more aggressive truncation."""
    if not cmd or cmd[0] not in SAFE_BIN:
        return {"error": f"Not allowed: {cmd[0] if cmd else 'empty'}"}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        return {
            "returncode": p.returncode,
            "stdout": p.stdout[-5000:],  # Aggressive truncation
            "stderr": p.stderr[-5000:],
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:100]}

TOOLS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "run_cmd": run_cmd,
}

def minimal_tool_specs() -> list[dict[str, Any]]:
    """Ultra-minimal tool specs for faster parsing."""
    return [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_cmd",
                "description": "Run cmd",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["cmd"],
                },
            },
        },
    ]

# ----------------------------
# Ultra Agent
# ----------------------------
class UltraAgent:
    """Ultra-fast agent with streaming and batching."""

    def __init__(self, goal: str) -> None:
        self.ctx = ContextManager()
        self.ctx.load_or_init(goal)
        self._initialize_tasks(goal)

    def _initialize_tasks(self, goal: str) -> None:
        if self.ctx.state.goal and self.ctx.state.goal.tasks:
            log("Resume", "INFO")
            return

        log("New tasks", "INFO")
        if "mathx" in goal.lower():
            tasks = [
                Task(description="Create mathx", subtasks=[
                    Subtask(description="write mathx/__init__.py"),
                ]),
                Task(description="Add tests", subtasks=[
                    Subtask(description="write tests/test_mathx.py"),
                ]),
                Task(description="Add config", subtasks=[
                    Subtask(description="write pyproject.toml"),
                ]),
                Task(description="Verify", subtasks=[
                    Subtask(description="run ruff"),
                    Subtask(description="run pytest"),
                ]),
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
        return task.active_subtask() if task else None

    def advance_workflow(self, probe: dict[str, Any]) -> bool:
        if probe.get("pytest_ok") and probe.get("ruff_ok"):
            if self.ctx.state.goal:
                self.ctx.state.goal.status = "completed"
                self.ctx._save_state()
            return False

        task = self.ctx._get_current_task()
        if not task:
            return False

        if not task.active_subtask():
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
                return False

        return True

    def execute_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self.ctx.record_action(name, args, result="pending"):
            return {"error": "loop"}

        try:
            tool_fn = TOOLS.get(name)
            if not tool_fn:
                result = {"error": f"Unknown: {name}"}
                self.ctx.action_history[-1].result = "error"
                self.ctx.action_history[-1].error_msg = result["error"]
            else:
                output = tool_fn(**args)
                result = {"result": output}
                self.ctx.action_history[-1].result = "success"

        except Exception as e:
            result = {"error": str(e)[:100]}
            self.ctx.action_history[-1].result = "error"
            self.ctx.action_history[-1].error_msg = str(e)[:100]

        self.ctx._save_state()
        return result

    def should_mark_complete(self, probe: dict[str, Any]) -> tuple[bool, str]:
        subtask = self.get_current_subtask()
        if not subtask:
            return False, ""

        desc = subtask.description.lower()

        if "mathx/__init__.py" in desc and probe.get("pkg_exists"):
            return True, "pkg ok"
        if "test_mathx.py" in desc and probe.get("tests_exist"):
            return True, "tests ok"
        if "pyproject.toml" in desc and probe.get("pyproject_exists"):
            return True, "config ok"
        if "ruff" in desc and probe.get("ruff_ok"):
            return True, "ruff ok"
        if "pytest" in desc and probe.get("pytest_ok"):
            return True, "pytest ok"

        return False, ""

    def get_ultra_prompt(self) -> str:
        """Ultra-minimal prompt for speed."""
        subtask = self.get_current_subtask()
        if not subtask:
            return "done"
        return f"Do: {subtask.description}"

# ----------------------------
# Main loop with streaming
# ----------------------------
ULTRA_PROMPT = "Coding agent. Do task with tools. Be fast."

def main() -> None:
    goal = "Create mathx with add and multiply, tests, ruff and pytest."
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])

    log(f"🚀 ULTRA mode: {MODEL}")

    agent = UltraAgent(goal)
    total_start = time.time()

    for round_no in range(1, MAX_ROUNDS + 1):
        round_start = time.perf_counter()

        # Probe
        probe = probe_state()
        agent.ctx.update_probe_state(probe)

        # Check completion
        complete, reason = agent.should_mark_complete(probe)
        if complete:
            log(f"✓ {reason}")
            agent.ctx.mark_subtask_complete(success=True)
            if not agent.advance_workflow(probe):
                elapsed = time.time() - total_start
                log(f"🎉 Done: {elapsed:.1f}s ({round_no} rounds)")
                return

        if not agent.advance_workflow(probe):
            elapsed = time.time() - total_start
            log(f"🎉 Done: {elapsed:.1f}s ({round_no} rounds)")
            return

        subtask = agent.get_current_subtask()
        if not subtask:
            continue

        # Ultra-minimal prompt
        prompt = agent.get_ultra_prompt()
        messages = [
            {"role": "system", "content": ULTRA_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # LLM call
        llm_start = time.perf_counter()
        try:
            resp = chat(
                model=MODEL,
                messages=messages,
                tools=minimal_tool_specs(),
                options={"temperature": TEMP, "num_predict": 500},
                stream=False,
            )
            llm_time = (time.perf_counter() - llm_start) * 1000

            calls = resp["message"].get("tool_calls") or []

            if calls:
                # Batch execute tools
                with ThreadPoolExecutor(max_workers=min(len(calls), BATCH_SIZE)) as executor:
                    futures = []
                    for call in calls:
                        name = call["function"]["name"]
                        args_str = call["function"].get("arguments", "{}")
                        args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})

                        future = executor.submit(agent.execute_tool_call, name, args)
                        futures.append((name, future))

                    for name, future in futures:
                        result = future.result()
                        if "error" in result and "loop" in result.get("error", ""):
                            agent.ctx.mark_subtask_complete(success=False, reason="loop")
                            if not agent.advance_workflow(probe):
                                return

        except Exception as e:
            log(f"Error: {e}", "ERROR")
            continue

        round_time = (time.perf_counter() - round_start) * 1000
        log(f"R{round_no}: {round_time:.0f}ms (LLM: {llm_time:.0f}ms)")

    log(f"Max rounds")

if __name__ == "__main__":
    main()
