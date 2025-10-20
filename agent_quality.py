#!/usr/bin/env python3
# agent_quality.py — optimized for quality (gpt-oss:20b) with LLM warm-up
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from ollama import chat
from context_manager import ContextManager, Subtask, Task
from llm_warmup import LLMWarmer

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------
# Config (Optimized for gpt-oss:20b quality)
# ----------------------------
MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
TEMP = 0.2  # Balance quality and determinism
MAX_ROUNDS = 24
LOGFILE = "agent_quality.log"
SAFE_BIN = {"python", "pytest", "ruff", "pip"}

# Performance optimizations (while maintaining quality)
PROBE_CACHE_TTL = 3.0
PARALLEL_PROBES = True
ENABLE_LLM_WARMUP = True  # Pre-warm model on startup
ENABLE_KEEPALIVE = True   # Keep model warm between calls

# Global LLM warmer
llm_warmer = LLMWarmer(MODEL) if ENABLE_LLM_WARMUP else None

# ----------------------------
# Logging
# ----------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")

# ----------------------------
# Probe Cache (from agent_fast.py)
# ----------------------------
class ProbeCache:
    def __init__(self):
        self.last_probe_result: dict[str, Any] | None = None
        self.last_probe_time: float = 0
        self.last_write_time: float = 0
        self.files_written_since_probe: set[str] = set()

    def record_write(self, path: str) -> None:
        self.last_write_time = time.time()
        self.files_written_since_probe.add(path)

    def should_probe(self) -> bool:
        if self.last_probe_result is None:
            return True

        if not self.files_written_since_probe:
            time_since_probe = time.time() - self.last_probe_time
            if time_since_probe < PROBE_CACHE_TTL:
                log(f"  ↻ Cache hit (age: {time_since_probe:.1f}s)")
                return False

        return True

    def update(self, result: dict[str, Any]) -> None:
        self.last_probe_result = result
        self.last_probe_time = time.time()
        self.files_written_since_probe.clear()

    def get(self) -> dict[str, Any]:
        return self.last_probe_result or {}

probe_cache = ProbeCache()

# ----------------------------
# Probe state (optimized)
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
    """Probe with caching and parallel execution."""
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

    def run_ruff():
        return run_cmd(["ruff", "check", "."], timeout_sec=10)

    def run_pytest():
        if not state["tests_exist"]:
            return {"returncode": 5, "stderr": "tests/ not found", "stdout": ""}
        return run_cmd(["pytest", "tests/", "-q", "-x"], timeout_sec=15)

    if PARALLEL_PROBES:
        with ThreadPoolExecutor(max_workers=2) as executor:
            ruff_future = executor.submit(run_ruff)
            pytest_future = executor.submit(run_pytest)
            ruff_rc = ruff_future.result()
            pytest_rc = pytest_future.result()
    else:
        ruff_rc = run_ruff()
        pytest_rc = run_pytest()

    # Process results
    if "returncode" in ruff_rc:
        state["ruff_ok"] = (ruff_rc["returncode"] == 0)
        if ruff_rc["returncode"] != 0:
            state["ruff_err"] = (ruff_rc.get("stderr") or ruff_rc.get("stdout") or "")[:200]
    else:
        state["ruff_ok"] = False
        state["ruff_err"] = (ruff_rc.get("error") or "")[:200]

    if "returncode" in pytest_rc:
        state["pytest_ok"] = (pytest_rc["returncode"] == 0)
        if pytest_rc["returncode"] != 0:
            state["pytest_err"] = (pytest_rc.get("stderr") or pytest_rc.get("stdout") or "")[:200]
    else:
        state["pytest_ok"] = False
        state["pytest_err"] = (pytest_rc.get("error") or "")[:200]

    elapsed = (time.perf_counter() - t0) * 1000
    log(f"  ⚡ probe: {elapsed:.1f}ms")

    probe_cache.update(state)
    return state

# ----------------------------
# Tools
# ----------------------------
def list_dir(path: str | None = ".", **kwargs) -> list[str]:
    p = path or "."
    try:
        return sorted(os.listdir(p))
    except FileNotFoundError as e:
        return [f"__error__: {e}"]

def read_file(path: str, max_bytes: int = 200_000) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read(max_bytes)

def write_file(path: str, content: str, create_dirs: bool = True) -> str:
    if create_dirs:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    probe_cache.record_write(path)
    return f"Wrote {len(content)} chars to {path}"

def run_cmd(cmd: list[str], timeout_sec: int = 60) -> dict[str, Any]:
    if not cmd or cmd[0] not in SAFE_BIN:
        return {"error": f"Command not allowed: {cmd!r}. Use only {sorted(SAFE_BIN)}."}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        return {
            "returncode": p.returncode,
            "stdout": p.stdout[-20_000:],
            "stderr": p.stderr[-20_000:],
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
        spec("read_file", "Read a file", {
            "path": {"type": "string", "required": True},
        }),
        spec("list_dir", "List directory contents", {
            "path": {"type": "string"},
        }),
    ]

# ----------------------------
# Quality Agent (same as FastAgent but with warm-up)
# ----------------------------
class QualityAgent:
    """Agent optimized for quality with gpt-oss:20b + warm-up."""

    def __init__(self, goal: str) -> None:
        self.ctx = ContextManager()
        self.ctx.load_or_init(goal)
        self._initialize_tasks_from_goal(goal)

    def _initialize_tasks_from_goal(self, goal: str) -> None:
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

        if "mathx/__init__.py" in desc and probe.get("pkg_exists"):
            return True, "mathx/__init__.py created"
        if "test_mathx.py" in desc and probe.get("tests_exist"):
            return True, "tests/test_mathx.py created"
        if "pyproject.toml" in desc and probe.get("pyproject_exists"):
            return True, "pyproject.toml created"
        if "ruff" in desc and probe.get("ruff_ok"):
            return True, "ruff check passed"
        if "pytest" in desc and probe.get("pytest_ok"):
            return True, "pytest passed"

        return False, ""

    def get_llm_prompt(self, probe: dict[str, Any]) -> str:
        """Get prompt for current subtask."""
        subtask = self.get_current_subtask()
        if not subtask:
            return "All subtasks complete."

        ctx_summary = self.ctx.get_compact_context(max_chars=1500)
        return ctx_summary

# ----------------------------
# Main loop with LLM warm-up
# ----------------------------
SYSTEM_PROMPT = "You are a coding agent. Complete tasks using tools. Write high-quality, well-documented code."

def main() -> None:
    goal = "Create mathx package with add(a,b) and multiply(a,b), add tests, run ruff and pytest."
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])

    log(f"🚀 Quality mode: {MODEL}")

    # Warm up LLM to reduce first-call latency
    if llm_warmer and ENABLE_LLM_WARMUP:
        warmup_metrics = llm_warmer.warmup()
        log(f"   Warm-up saved {warmup_metrics['improvement_ms']:.0f}ms ({warmup_metrics['improvement_pct']:.1f}%)")

        if ENABLE_KEEPALIVE:
            llm_warmer.start_keepalive_thread()

    agent = QualityAgent(goal)
    total_start = time.time()

    try:
        for round_no in range(1, MAX_ROUNDS + 1):
            round_start = time.perf_counter()

            # Probe
            probe = probe_state()
            agent.ctx.update_probe_state(probe)

            # Check completion
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

            # Build prompt
            llm_prompt = agent.get_llm_prompt(probe)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": llm_prompt},
            ]

            # LLM call (record for keep-alive)
            llm_start = time.perf_counter()
            resp = chat(
                model=MODEL,
                messages=messages,
                tools=tool_specs(),
                options={"temperature": TEMP},
                stream=False,
            )
            llm_time = (time.perf_counter() - llm_start) * 1000

            if llm_warmer:
                llm_warmer.record_call()

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

        log(f"⏱ Stopped at MAX_ROUNDS")
        print(f"\n⏱ Hit MAX_ROUNDS")

    finally:
        # Clean up keep-alive thread
        if llm_warmer and ENABLE_KEEPALIVE:
            llm_warmer.stop_keepalive_thread()

if __name__ == "__main__":
    main()
