# agent_v2.py — improved local coding agent with proper context management
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ollama import chat
from context_manager import ContextManager
from status_display import StatusDisplay
from workspace_manager import WorkspaceManager
from completion_detector import analyze_llm_response

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
HISTORY_KEEP = 5  # Keep last 5 message exchanges
LOGFILE = "agent_v2.log"
LEDGER = Path("agent_ledger.log")
SAFE_BIN = {"python", "pytest", "ruff", "pip"}

# ----------------------------
# Logging / Ledger
# ----------------------------
def log(msg: str) -> None:
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[log] {msg}")

def _ledger_append(kind: str, detail: str) -> None:
    line = f"{kind}\t{detail.replace(chr(10),' ')[:400]}\n"
    if LEDGER.exists():
        LEDGER.write_text(LEDGER.read_text(encoding="utf-8") + line, encoding="utf-8")
    else:
        LEDGER.write_text(line, encoding="utf-8")

# ----------------------------
# Generic state probe
# ----------------------------
def probe_state_generic() -> dict[str, Any]:
    """Generic filesystem probe - no goal-specific assumptions."""

    state = {
        "files_written": [],      # Files we wrote (from ledger)
        "files_exist": [],        # Which of those actually exist now
        "files_missing": [],      # Which we wrote but are now gone
        "commands_run": [],       # Commands we executed
        "recent_errors": [],      # Last few errors
    }

    if not LEDGER.exists():
        return state

    lines = LEDGER.read_text(encoding="utf-8").splitlines()[-30:]
    writes = []

    for line in lines:
        if line.startswith("WRITE\t"):
            filepath = line.split("\t", 1)[1]
            writes.append(filepath)
        elif line.startswith("CMD\t"):
            state["commands_run"].append(line.split("\t", 1)[1])
        elif line.startswith("ERROR\t"):
            state["recent_errors"].append(line.split("\t", 1)[1])

    # Dedupe writes (keep order)
    seen = set()
    for w in writes:
        if w not in seen:
            state["files_written"].append(w)
            seen.add(w)

    # Check which files actually exist
    for filepath in state["files_written"]:
        if Path(filepath).exists():
            state["files_exist"].append(filepath)
        else:
            state["files_missing"].append(filepath)

    # Keep only recent items
    state["commands_run"] = list(dict.fromkeys(state["commands_run"]))[-5:]
    state["recent_errors"] = state["recent_errors"][-3:]

    return state

# ----------------------------
# Tools
# ----------------------------
def list_dir(path: str | None = ".", **kwargs) -> list[str]:
    """List files (non-recursive, workspace-aware)."""
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path or ".")
    else:
        resolved_path = Path(path or ".")

    try:
        return sorted(os.listdir(resolved_path))
    except FileNotFoundError as e:
        return [f"__error__: {e}"]

def read_file(path: str, max_bytes: int = 200_000) -> str:
    """Read a text file (truncated, workspace-aware)."""
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path)
    else:
        resolved_path = Path(path)

    with open(resolved_path, encoding="utf-8", errors="replace") as f:
        return f.read(max_bytes)

def write_file(path: str, content: str, create_dirs: bool = True) -> str:
    """Write/overwrite a text file (workspace-aware)."""
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path)
        _workspace.track_file(path)  # Track file creation
        display_path = _workspace.relative_path(resolved_path)
    else:
        resolved_path = Path(path)
        display_path = path

    if create_dirs:
        os.makedirs(os.path.dirname(resolved_path) or ".", exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(content)
    _ledger_append("WRITE", str(resolved_path))
    return f"Wrote {len(content)} chars to {display_path}"

def run_cmd(cmd: list[str], timeout_sec: int = 60) -> dict[str, Any]:
    """Run a whitelisted command (workspace-aware): first token must be in SAFE_BIN."""
    global _workspace

    if not cmd or cmd[0] not in SAFE_BIN:
        err = f"Command not allowed: {cmd!r}. Use only {sorted(SAFE_BIN)}."
        _ledger_append("ERROR", err)
        return {"error": err}

    # Determine working directory
    cwd = str(_workspace.workspace_dir) if _workspace else None

    # Set up environment with PYTHONPATH for workspace
    env = os.environ.copy()
    if _workspace and cwd:
        # Add workspace directory to PYTHONPATH for pytest imports
        if "pytest" in cmd or "python" in cmd:
            env["PYTHONPATH"] = cwd

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, cwd=cwd, env=env)
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

# Global context manager and workspace manager references
_ctx: ContextManager | None = None
_workspace: WorkspaceManager | None = None

def mark_subtask_complete(success: bool = True, reason: str = "") -> dict[str, Any]:
    """
    Mark current subtask as complete or failed.
    Call this when you've finished the current subtask.

    Args:
        success: True if subtask completed successfully, False if failed
        reason: Optional explanation (required if success=False)
    """
    global _ctx
    if not _ctx:
        return {"error": "No active context"}

    _ctx.mark_subtask_complete(success, reason)
    _ledger_append("SUBTASK", f"marked {'complete' if success else 'failed'}: {reason}")

    if success:
        # Try to advance to next subtask
        has_next = _ctx.advance_to_next_subtask()
        if not has_next:
            # Check if there are more tasks
            _ctx.state.current_task_idx += 1
            if _ctx.state.goal and _ctx.state.current_task_idx >= len(_ctx.state.goal.tasks):
                # All tasks complete
                return {"status": "goal_complete", "message": "All tasks finished!"}
            else:
                # Move to next task
                task = _ctx._get_current_task()
                if task and task.subtasks:
                    task.subtasks[0].status = "in_progress"
                    _ctx._save_state()
                    return {
                        "status": "task_advanced",
                        "next_task": task.description,
                        "next_subtask": task.subtasks[0].description
                    }
        else:
            next_subtask = _ctx._get_current_task().active_subtask()
            return {
                "status": "subtask_advanced",
                "next_subtask": next_subtask.description if next_subtask else "unknown"
            }
    else:
        return {"status": "subtask_failed", "reason": reason}

TOOLS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "run_cmd": run_cmd,
    "mark_subtask_complete": mark_subtask_complete,
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
        spec("mark_subtask_complete", "Mark current subtask as complete. Call this when you finish a subtask.", {
            "success": {"type": "boolean", "description": "True if completed successfully, False if failed"},
            "reason": {"type": "string", "description": "Optional explanation (required if failed)"},
        }),
    ]

# ----------------------------
# Task decomposition
# ----------------------------
def decompose_goal(goal: str) -> list[dict[str, Any]]:
    """Ask LLM to decompose goal into tasks and subtasks."""
    prompt = f"""Break down this goal into CONCRETE, ACTIONABLE tasks and subtasks.

Goal: {goal}

Rules:
- Each subtask must require using a tool (write_file, run_cmd, read_file, list_dir)
- No abstract decision-making tasks like "choose" or "decide"
- Be specific about filenames and actions
- Keep it simple - usually 1-3 tasks total

Return a JSON array of tasks. Each task should have:
- "description": brief task description
- "subtasks": array of subtask descriptions (strings)

Example for "Write a hello world script":
[
  {{"description": "Create hello_world.py script", "subtasks": ["Write hello_world.py with print statement"]}}
]

Example for "Create a math package":
[
  {{"description": "Create package files", "subtasks": ["Write mathx/__init__.py with add function", "Write tests/test_mathx.py with tests"]}},
  {{"description": "Verify quality", "subtasks": ["Run ruff check", "Run pytest"]}}
]

Return ONLY the JSON array, no other text.
"""

    log("Decomposing goal into tasks...")
    resp = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )

    content = resp["message"]["content"].strip()
    # Extract JSON from markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        tasks_data = json.loads(content)
        log(f"Decomposed into {len(tasks_data)} tasks")
        return tasks_data
    except json.JSONDecodeError as e:
        log(f"Failed to parse task decomposition: {e}")
        # Fallback: create a single generic task
        return [{"description": goal, "subtasks": ["Complete the goal"]}]

# ----------------------------
# Context building
# ----------------------------
SYSTEM_PROMPT = """You are a local coding agent running on Windows.

Your workflow:
1. Work on your current subtask using the available tools
2. When you complete a subtask, call mark_subtask_complete(success=True)
3. If you cannot complete it, call mark_subtask_complete(success=False, reason="...")
4. The system will automatically advance you to the next subtask

Guidelines:
- Use tools to read, write files, and run commands
- Only python, pytest, ruff, and pip commands are allowed
- When you finish your current subtask, ALWAYS call mark_subtask_complete
- Be concise and focused on the current subtask
"""

def build_context(ctx: ContextManager, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build context with: system prompt + current task info + last N messages."""

    # Start with system prompt
    context = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add current goal/task/subtask context
    if ctx.state.goal:
        task = ctx._get_current_task()
        subtask = task.active_subtask() if task else None

        context_info = [
            f"GOAL: {ctx.state.goal.description}",
        ]

        if task:
            context_info.append(f"CURRENT TASK: {task.description}")

        if subtask:
            context_info.append(f"ACTIVE SUBTASK: {subtask.description}")
        else:
            context_info.append("ACTIVE SUBTASK: (none - call mark_subtask_complete to advance)")

        # Add generic filesystem state
        probe = probe_state_generic()
        if probe["files_exist"]:
            context_info.append(f"FILES CREATED: {', '.join(probe['files_exist'])}")
        if probe["recent_errors"]:
            context_info.append(f"RECENT ERRORS: {probe['recent_errors'][-1][:100]}")

        context.append({"role": "user", "content": "\n".join(context_info)})

    # Add last N message exchanges (keep it simple)
    recent = messages[-HISTORY_KEEP * 2:] if len(messages) > HISTORY_KEEP * 2 else messages
    context.extend(recent)

    return context

# ----------------------------
# Dispatch tool calls
# ----------------------------
def dispatch(call: dict[str, Any]) -> dict[str, Any]:
    import traceback
    name = call["function"]["name"]
    args = call["function"].get("arguments", "{}")

    log(f"TOOL→ {name} args={args[:200] if isinstance(args, str) else str(args)[:200]}")
    fn = TOOLS.get(name)
    if not fn:
        log(f"TOOL✖ unknown: {name}")
        return {"error": f"unknown tool {name}"}
    try:
        data = json.loads(args) if isinstance(args, str) else (args or {})
        out = fn(**data) if data else fn()
        log(f"TOOL✓ {name} → {type(out).__name__}")
        return {"result": out}
    except Exception as e:
        log(f"TOOL✖ {name} error={e}")
        log(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

# ----------------------------
# Main loop
# ----------------------------
def main() -> None:
    global _ctx, _workspace

    goal = "Write a hello world script"
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])

    log(f"Starting agent with goal: {goal}")

    # Initialize workspace manager (isolated directory for this goal)
    _workspace = WorkspaceManager(goal)
    log(f"Workspace: {_workspace.workspace_dir}")

    # Initialize context manager
    _ctx = ContextManager()
    _ctx.load_or_init(goal)

    # Initialize status display
    status = StatusDisplay(_ctx)
    _ctx.loop_callback = status.record_loop  # Wire up loop detection callback

    # If new goal, decompose into tasks
    if not _ctx.state.goal or not _ctx.state.goal.tasks:
        tasks_data = decompose_goal(goal)

        # Build task hierarchy
        from context_manager import Task, Subtask
        for task_data in tasks_data:
            task = Task(
                description=task_data["description"],
                subtasks=[Subtask(description=st) for st in task_data["subtasks"]],
                status="pending",
                parent_goal=goal,
            )
            if _ctx.state.goal:
                _ctx.state.goal.tasks.append(task)

        # Mark first task/subtask as active
        if _ctx.state.goal and _ctx.state.goal.tasks:
            _ctx.state.goal.tasks[0].status = "in_progress"
            if _ctx.state.goal.tasks[0].subtasks:
                _ctx.state.goal.tasks[0].subtasks[0].status = "in_progress"

        _ctx._save_state()

    # Message history (just the conversation, not context info)
    messages: list[dict[str, Any]] = []

    for round_no in range(1, MAX_ROUNDS + 1):
        # Update probe state in context manager
        probe = probe_state_generic()
        _ctx.update_probe_state(probe)

        # Display status at start of round
        print("\n" + status.render(round_no))
        print()  # Add spacing

        # Check if goal is complete
        if (_ctx.state.goal and
            _ctx.state.current_task_idx >= len(_ctx.state.goal.tasks)):
            print("\n=== Agent Complete ===")
            print(f"Goal achieved: {goal}")
            print(status.render_compact())
            if probe["files_exist"]:
                print(f"Files created: {', '.join(probe['files_exist'])}")
            return

        # Build context for this round
        context = build_context(_ctx, messages)

        log(f"ROUND {round_no}: sending {len(context)} messages")

        # Call LLM
        t0 = time.time()
        resp = chat(
            model=MODEL,
            messages=context,
            tools=tool_specs(),
            options={"temperature": TEMP},
            stream=False,
        )
        llm_duration = time.time() - t0
        status.record_llm_call(llm_duration, len(context))
        log(f"ROUND {round_no}: chat() {llm_duration:.2f}s")

        msg = resp["message"]
        calls = msg.get("tool_calls") or []

        if calls:
            names = ", ".join(c["function"]["name"] for c in calls)
            log(f"ROUND {round_no}: tool_calls → {names} (n={len(calls)})")

            # Add assistant message with tool calls
            messages.append(msg)

            # Analyze LLM response for completion signals ONCE per response
            current_subtask = _ctx._get_current_task().active_subtask()
            subtask_desc = current_subtask.description if current_subtask else None
            analysis = analyze_llm_response(msg.get("content", ""), calls, subtask_desc)

            for c in calls:
                tool_name = c["function"]["name"]

                # Execute tool
                try:
                    tool_result = dispatch(c)
                except Exception as e:
                    tool_result = {"error": f"dispatch-failed: {e}"}
                    log(f"Dispatch error: {e}")

                # Record action in context manager (only for non-completion tools)
                if tool_name != "mark_subtask_complete":
                    try:
                        args_dict = json.loads(c["function"]["arguments"]) if isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]
                        result_status = "success" if "result" in tool_result else "error"
                        error_msg = str(tool_result.get("error", ""))
                        # Don't let record_action fail - just log if it does
                        try:
                            _ctx.record_action(tool_name, args_dict, result_status, error_msg)
                            status.record_action(result_status == "success")
                        except Exception as record_err:
                            log(f"Failed to record action: {record_err}")
                    except Exception as parse_err:
                        log(f"Failed to parse args for recording: {parse_err}")
                else:
                    # Record subtask completion
                    if isinstance(tool_result, dict):
                        success = tool_result.get("status") in ["subtask_advanced", "task_advanced", "goal_complete"]
                        status.record_subtask_complete(success)

                # Add tool result to messages (with nudge if needed on LAST tool call)
                tool_result_str = json.dumps(tool_result)
                is_last_call = (c == calls[-1])

                if is_last_call and analysis["should_nudge"]:
                    # Append nudge message to last tool result
                    tool_result_with_nudge = tool_result.copy() if isinstance(tool_result, dict) else {}
                    tool_result_with_nudge["_nudge"] = analysis["nudge_message"]
                    tool_result_str = json.dumps(tool_result_with_nudge)
                    log(f"NUDGE: {analysis['reason']}")

                messages.append({
                    "role": "tool",
                    "content": tool_result_str,
                })

                # Check if goal completed via mark_subtask_complete
                if isinstance(tool_result, dict) and tool_result.get("status") == "goal_complete":
                    print("\n=== Agent Complete ===")
                    print(f"Goal achieved: {goal}")
                    if probe["files_exist"]:
                        print(f"Files created: {', '.join(probe['files_exist'])}")
                    return

            continue

        # No tool calls - check for completion signals before giving final answer
        messages.append(msg)

        # Analyze for completion signals
        current_subtask = _ctx._get_current_task().active_subtask()
        subtask_desc = current_subtask.description if current_subtask else None
        analysis = analyze_llm_response(msg.get("content", ""), calls, subtask_desc)

        if analysis["should_nudge"]:
            # Agent mentioned completion but didn't call mark_subtask_complete
            # Add a strong system message to prompt the tool call
            log(f"NUDGE: {analysis['reason']}")
            nudge_msg = (
                f"{analysis['nudge_message']}\n\n"
                f"IMPORTANT: You must call mark_subtask_complete(success=True) to advance. "
                f"Do not give a final answer until the goal is marked complete."
            )
            messages.append({
                "role": "user",
                "content": nudge_msg
            })
            # Continue to next round to let agent respond with mark_subtask_complete
            continue

        # No completion signal - this is a real final answer
        print("\n=== Agent Reply ===")
        print(msg["content"])
        return

    # Hit max rounds
    print(f"\n[stopped] Hit MAX_ROUNDS ({MAX_ROUNDS}) without completion.")
    print(f"Current task: {_ctx._get_current_task().description if _ctx._get_current_task() else 'none'}")

if __name__ == "__main__":
    main()
