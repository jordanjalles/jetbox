#!/usr/bin/env python3
# agent_enhanced.py — agent with hierarchical context manager integration
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ollama import chat
from context_manager import ContextManager, Subtask, Task

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------
# Config
# ----------------------------
MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
TEMP = 0.2
MAX_ROUNDS = 24
HISTORY_KEEP = 12
LOGFILE = "agent_enhanced.log"
SAFE_BIN = {"python", "pytest", "ruff", "pip"}

# ----------------------------
# Logging
# ----------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[log] {msg}")

# ----------------------------
# Probe state (unchanged from original)
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

    # Try pytest (quiet, only tests/)
    rc = run_cmd(["pytest", "tests/", "-q"], timeout_sec=90)
    if "returncode" in rc:
        state["pytest_ok"] = (rc["returncode"] == 0)
        if rc["returncode"] != 0:
            state["pytest_err"] = (rc.get("stderr") or rc.get("stdout") or "")[:300]
    else:
        state["pytest_ok"] = False
        state["pytest_err"] = (rc.get("error") or "")[:300]

    return state

# ----------------------------
# Tools
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
            "stdout": p.stdout[-50_000:],
            "stderr": p.stderr[-50_000:],
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
# Enhanced Agent with Context Manager
# ----------------------------
class EnhancedAgent:
    """Agent with hierarchical context management."""

    def __init__(self, goal: str) -> None:
        self.ctx = ContextManager()
        self.ctx.load_or_init(goal)
        self._initialize_tasks_from_goal(goal)

    def _initialize_tasks_from_goal(self, goal: str) -> None:
        """Create initial task hierarchy if not resuming."""
        if self.ctx.state.goal and self.ctx.state.goal.tasks:
            log("Resuming existing task hierarchy")
            return

        log("Creating new task hierarchy for goal")
        # Decompose goal into tasks
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

            # Mark first task/subtask as active
            if tasks:
                tasks[0].status = "in_progress"
                tasks[0].subtasks[0].status = "in_progress"

            self.ctx._save_state()
            log(f"Created {len(tasks)} tasks")

    def get_current_subtask(self) -> Subtask | None:
        """Get the current active subtask."""
        task = self.ctx._get_current_task()
        if not task:
            return None
        return task.active_subtask()

    def advance_workflow(self, probe: dict[str, Any]) -> bool:
        """
        Advance to next subtask/task based on probe state.
        Returns True if there's more work, False if done.
        """
        # Check if goal is complete
        if probe.get("pytest_ok") and probe.get("ruff_ok"):
            if self.ctx.state.goal:
                self.ctx.state.goal.status = "completed"
                self.ctx._save_state()
            return False

        # Get current task
        task = self.ctx._get_current_task()
        if not task:
            return False

        # Check current subtask
        subtask = task.active_subtask()
        if not subtask:
            # Try to advance to next subtask
            if self.ctx.advance_to_next_subtask():
                log(f"Advanced to next subtask in task: {task.description}")
                return True
            else:
                # Task complete, move to next task
                task.status = "completed"
                self.ctx.state.current_task_idx += 1
                self.ctx._save_state()

                # Start next task
                next_task = self.ctx._get_current_task()
                if next_task:
                    next_task.status = "in_progress"
                    if next_task.subtasks:
                        next_task.subtasks[0].status = "in_progress"
                    self.ctx._save_state()
                    log(f"Advanced to next task: {next_task.description}")
                    return True
                else:
                    log("All tasks completed!")
                    return False

        return True

    def execute_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute tool call with loop detection."""
        # Check if action is allowed (not in loop)
        if not self.ctx.record_action(name, args, result="pending"):
            log(f"Action blocked (loop): {name}")
            return {
                "error": "Action blocked: loop detected",
                "suggestion": "This action has been tried multiple times. Try a different approach.",
            }

        # Execute the tool
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
                log(f"Tool {name} succeeded")

        except Exception as e:
            result = {"error": str(e)}
            self.ctx.action_history[-1].result = "error"
            self.ctx.action_history[-1].error_msg = str(e)
            log(f"Tool {name} failed: {e}")

        self.ctx._save_state()
        return result

    def should_mark_subtask_complete(self, probe: dict[str, Any]) -> tuple[bool, str]:
        """
        Check if current subtask should be marked complete based on probe state.
        Returns (should_complete, reason).
        """
        subtask = self.get_current_subtask()
        if not subtask:
            return False, ""

        desc = subtask.description.lower()

        # Check if subtask objective is met
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
            elif probe.get("ruff_ok") is False:
                return False, f"ruff failed: {probe.get('ruff_err', '')[:100]}"
        elif "pytest" in desc:
            if probe.get("pytest_ok"):
                return True, "pytest passed"
            elif probe.get("pytest_ok") is False:
                return False, f"pytest failed: {probe.get('pytest_err', '')[:100]}"

        return False, ""

    def get_llm_prompt(self, probe: dict[str, Any]) -> str:
        """Generate prompt for LLM with hierarchical context."""
        # Get compact hierarchical context
        ctx_summary = self.ctx.get_compact_context(max_chars=1500)

        # Add current subtask instruction
        subtask = self.get_current_subtask()
        if subtask:
            instruction = f"\nCURRENT SUBTASK: {subtask.description}\n"
            instruction += "Complete this subtask using the available tools, then I will check the result.\n"
        else:
            instruction = "\nAll subtasks complete. Waiting for verification.\n"

        # Add guidance if loops detected
        guidance = ""
        if self.ctx.state.blocked_actions:
            guidance = f"\n⚠ WARNING: {len(self.ctx.state.blocked_actions)} actions are blocked due to loops.\n"
            guidance += "Try a different approach if your action is blocked.\n"

        prompt = f"{ctx_summary}\n{instruction}{guidance}"
        return prompt

# ----------------------------
# Main loop
# ----------------------------
SYSTEM_PROMPT = (
    "You are a coding agent that completes tasks step by step.\n"
    "- Focus on the CURRENT SUBTASK shown in the context\n"
    "- Use tools to complete the subtask\n"
    "- Don't repeat actions that have already succeeded\n"
    "- If an action is blocked, try a different approach\n"
    "- Work methodically through each subtask\n"
)

def main() -> None:
    goal = "Create a tiny package 'mathx' with add(a,b) and multiply(a,b), add tests, then run ruff and pytest."
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])

    log(f"Starting with goal: {goal}")
    agent = EnhancedAgent(goal)

    for round_no in range(1, MAX_ROUNDS + 1):
        # 1) Probe current state
        probe = probe_state()
        agent.ctx.update_probe_state(probe)
        log(f"ROUND {round_no}: probe pkg={probe['pkg_exists']} tests={probe['tests_exist']} pytest={probe['pytest_ok']} ruff={probe['ruff_ok']}")

        # 2) Check if subtask is complete and advance if needed
        should_complete, reason = agent.should_mark_subtask_complete(probe)
        if should_complete:
            log(f"Subtask complete: {reason}")
            agent.ctx.mark_subtask_complete(success=True)
            if not agent.advance_workflow(probe):
                print("\n=== Agent Reply ===\n✓ DONE — All tasks completed successfully!")
                return

        # 3) Check if we can still make progress
        if not agent.advance_workflow(probe):
            print("\n=== Agent Reply ===\n✓ DONE — Goal achieved!")
            return

        # 4) Get current subtask and prepare prompt
        subtask = agent.get_current_subtask()
        if not subtask:
            log("No active subtask, but work remains")
            continue

        # 5) Build messages for LLM
        llm_prompt = agent.get_llm_prompt(probe)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": llm_prompt},
        ]

        log(f"ROUND {round_no}: sending prompt for subtask '{subtask.description[:50]}'")

        # 6) Ask the model
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
            log(f"ROUND {round_no}: {len(calls)} tool call(s)")

            for call in calls:
                name = call["function"]["name"]
                args_str = call["function"].get("arguments", "{}")
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})

                log(f"  → {name}({list(args.keys())})")

                # Execute with loop detection
                result = agent.execute_tool_call(name, args)

                if "error" in result:
                    log(f"  ✗ {result['error']}")

                    # If loop detected, mark subtask as failed and try to recover
                    if "loop detected" in result.get("error", ""):
                        agent.ctx.mark_subtask_complete(success=False, reason="Loop detected")
                        # Try to advance to next subtask
                        if not agent.advance_workflow(probe):
                            print("\n=== Agent Reply ===\nBlocked by loops, cannot continue.")
                            return
                else:
                    log(f"  ✓ success")

        else:
            # No tool calls - model gave a text response
            log(f"ROUND {round_no}: no tool calls, model response: {msg.get('content', '')[:100]}")

    print("\n[stopped] hit MAX_ROUNDS")
    print(f"\nFinal status:\n{agent.ctx.get_compact_context()}")

if __name__ == "__main__":
    main()
