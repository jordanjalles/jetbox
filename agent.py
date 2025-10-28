# agent_v2.py — improved local coding agent with proper context management
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from threading import Thread

from ollama import Client
from ollama._types import ResponseError
from context_manager import ContextManager
from status_display import StatusDisplay
from workspace_manager import WorkspaceManager
from completion_detector import analyze_llm_response
from agent_config import config
import tools  # Shared tool implementations
from llm_utils import chat_with_inactivity_timeout, check_ollama_health  # LLM utilities
from context_strategies import build_hierarchical_context  # Context building

# Initialize Ollama client with proper host configuration
OLLAMA_CLIENT = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------
# Config (loaded from agent_config.yaml)
# ----------------------------
MODEL = config.llm.model
TEMP = config.llm.temperature
HISTORY_KEEP = config.context.history_keep


# ----------------------------
# Timeout wrapper for LLM calls - NOW IN llm_utils.py
# ----------------------------
# chat_with_inactivity_timeout moved to llm_utils.py
# (imported at top of file - see: from llm_utils import chat_with_inactivity_timeout)
LOGFILE = "agent_v2.log"
LEDGER = Path("agent_ledger.log")
# SAFE_BIN moved to tools.py

# Load from config file
MAX_ROUNDS = config.rounds.max_global
MAX_ROUNDS_PER_SUBTASK = config.rounds.max_per_subtask
MAX_SUBTASK_DEPTH = config.hierarchy.max_depth
MAX_SUBTASK_SIBLINGS = config.hierarchy.max_siblings

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

def send_message_to_orchestrator(message: str, severity: str = "info") -> None:
    """
    Send a message from TaskExecutor to Orchestrator.

    Messages are written to .agent_context/messages_to_orchestrator.jsonl
    and can be read by the orchestrator to understand issues.

    Args:
        message: Message content
        severity: "info", "warning", or "error"
    """
    import time
    from datetime import datetime

    msg_file = Path(".agent_context/messages_to_orchestrator.jsonl")
    msg_file.parent.mkdir(exist_ok=True)

    msg_entry = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity,
        "message": message,
    }

    # Append to file
    with open(msg_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg_entry) + "\n")

    log(f"MESSAGE_TO_ORCHESTRATOR [{severity}]: {message}")

# ----------------------------
# Generic state probe
# ----------------------------
def probe_state_generic() -> dict[str, Any]:
    """Generic filesystem probe - no goal-specific assumptions.

    Only includes files/errors from the current session (after last SESSION_START marker).
    """

    state = {
        "files_written": [],      # Files we wrote (from ledger)
        "files_exist": [],        # Which of those actually exist now
        "files_missing": [],      # Which we wrote but are now gone
        "commands_run": [],       # Commands we executed
        "recent_errors": [],      # Last few errors
        "workspace_empty": False, # Whether workspace is empty
        "warning": None,          # Workspace mismatch warning if applicable
    }

    if not LEDGER.exists():
        # Check for empty workspace with read/modify task
        if _workspace:
            ws_files = [f for f in _workspace.workspace_dir.glob("*") if f.is_file() and not f.name.startswith(".")]
            if not ws_files and _ctx:
                task = _ctx._get_current_task()
                current_subtask = task.active_subtask() if task else None
                if current_subtask:
                    desc_lower = current_subtask.description.lower()
                    read_keywords = ["read", "modify", "update", "edit", "load", "open", "inspect"]
                    if any(keyword in desc_lower for keyword in read_keywords):
                        state["workspace_empty"] = True
                        state["warning"] = (
                            "⚠️ WORKSPACE MISMATCH: Workspace is empty but task requires reading/modifying files. "
                            "This likely indicates the wrong workspace was used. Consider escalating or checking "
                            "if files should exist in a different workspace."
                        )
                        # Send message to orchestrator
                        send_message_to_orchestrator(
                            f"Workspace mismatch detected: Empty workspace but task '{current_subtask.description}' "
                            f"requires reading/modifying files. Current workspace: {_workspace.workspace_dir}",
                            severity="warning"
                        )
        return state

    # Read ledger and find last SESSION_START marker
    all_lines = LEDGER.read_text(encoding="utf-8").splitlines()

    # Find last session marker
    session_start_idx = 0
    for i in range(len(all_lines) - 1, -1, -1):
        if all_lines[i].startswith("SESSION_START\t"):
            session_start_idx = i
            break

    # Only process lines after session start
    lines = all_lines[session_start_idx:]

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

    # Check for empty workspace with read/modify task (even with ledger)
    if _workspace:
        ws_files = [f for f in _workspace.workspace_dir.glob("*") if f.is_file() and not f.name.startswith(".")]
        if not ws_files and _ctx:
            task = _ctx._get_current_task()
            current_subtask = task.active_subtask() if task else None
            if current_subtask:
                desc_lower = current_subtask.description.lower()
                read_keywords = ["read", "modify", "update", "edit", "load", "open", "inspect"]
                if any(keyword in desc_lower for keyword in read_keywords):
                    state["workspace_empty"] = True
                    state["warning"] = (
                        "⚠️ WORKSPACE MISMATCH: Workspace is empty but task requires reading/modifying files. "
                        "This likely indicates the wrong workspace was used. Consider escalating or checking "
                        "if files should exist in a different workspace."
                    )

    return state

# ----------------------------
# Tools - NOW IN tools.py
# All tool implementations moved to tools.py for code reuse
# ----------------------------
# (Tool functions removed - import from tools module instead)

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

def read_file(path: str, max_bytes: int = 200_000, offset: int = 0) -> str:
    """Read a text file with optional offset for pagination (workspace-aware)."""
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path)
    else:
        resolved_path = Path(path)

    with open(resolved_path, encoding="utf-8", errors="replace") as f:
        if offset > 0:
            f.seek(offset)
        content = f.read(max_bytes)

        # Add helpful metadata if file was truncated or offset was used
        file_size = resolved_path.stat().st_size
        if offset > 0 or len(content) == max_bytes:
            end_pos = offset + len(content)
            metadata = f"\n[FILE READ: bytes {offset}-{end_pos} of {file_size} total]"
            if end_pos < file_size:
                metadata += f"\n[NOTE: {file_size - end_pos} bytes remaining. Use offset={end_pos} to continue reading]"
            return content + metadata
        return content

def grep_file(path: str, pattern: str, context_lines: int = 3, max_matches: int = 50) -> str:
    """
    Search for pattern in file and return matching lines with context.
    Uses Python regex. Returns line numbers and surrounding context.
    """
    import re
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path)
    else:
        resolved_path = Path(path)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[ERROR] Invalid regex pattern: {e}"

    matches = []
    try:
        with open(resolved_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Find all matching lines
        matching_line_nums = []
        for i, line in enumerate(lines):
            if regex.search(line):
                matching_line_nums.append(i)
                if len(matching_line_nums) >= max_matches:
                    break

        if not matching_line_nums:
            return f"[NO MATCHES] Pattern '{pattern}' not found in {path}"

        # Build output with context
        result_lines = []
        result_lines.append(f"[GREP] Found {len(matching_line_nums)} match(es) for '{pattern}' in {path}")
        result_lines.append("")

        # For each match, include context lines
        included_lines = set()
        for line_num in matching_line_nums:
            start = max(0, line_num - context_lines)
            end = min(len(lines), line_num + context_lines + 1)

            # Add separator if there's a gap from previous match
            if included_lines and start > max(included_lines) + 1:
                result_lines.append("...")
                result_lines.append("")

            for i in range(start, end):
                if i not in included_lines:
                    prefix = ">>> " if i == line_num else "    "
                    result_lines.append(f"{prefix}{i+1:4d} | {lines[i].rstrip()}")
                    included_lines.add(i)

            result_lines.append("")

        if len(matching_line_nums) >= max_matches:
            result_lines.append(f"[NOTE] Showing first {max_matches} matches. File may contain more.")

        return "\n".join(result_lines)

    except Exception as e:
        return f"[ERROR] Failed to search file: {e}"

def write_file(path: str, content: str, create_dirs: bool = True) -> str:
    """Write/overwrite a text file (workspace-aware)."""
    global _workspace

    # Resolve path through workspace if available
    if _workspace:
        resolved_path = _workspace.resolve_path(path)

        # Safety check in edit mode: prevent modifying agent code
        if _workspace.is_edit_mode:
            forbidden_files = {'agent.py', 'context_manager.py', 'workspace_manager.py',
                              'status_display.py', 'completion_detector.py', 'agent_config.py'}
            if resolved_path.name in forbidden_files:
                error_msg = f"[SAFETY] Cannot modify agent code in edit mode: {resolved_path.name}"
                log(error_msg)
                return error_msg

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

# ----------------------------
# Server management tools
# ----------------------------

def start_server(cmd: list[str], name: str = None) -> dict[str, Any]:
    """
    Request orchestrator to start a server in the background.

    Args:
        cmd: Command to run (e.g., ['python', '-m', 'http.server', '8000'])
        name: Optional server name (auto-generated if not provided)

    Returns:
        Server info or error
    """
    global _workspace

    # Validate command
    if not cmd or cmd[0] not in SAFE_BIN:
        return {"error": f"Command not allowed: {cmd!r}. Use only {sorted(SAFE_BIN)}."}

    # Generate server ID
    server_id = name or f"server_{int(time.time())}"

    # Set up paths
    cwd = str(_workspace.workspace_dir) if _workspace else os.getcwd()
    log_file = os.path.join(cwd, f"{server_id}.log")

    # Write request
    request_file = Path(".agent_context/server_requests.jsonl")
    request_file.parent.mkdir(parents=True, exist_ok=True)

    request = {
        "action": "start",
        "server_id": server_id,
        "cmd": cmd,
        "cwd": cwd,
        "log_file": log_file,
    }

    with open(request_file, 'a') as f:
        f.write(json.dumps(request) + '\n')

    # Poll for response (wait up to 5 seconds)
    response = _wait_for_server_response(timeout=5.0)

    if response:
        _ledger_append("SERVER", f"start {server_id} -> {response.get('success', False)}")
        return response
    else:
        return {"error": "Timeout waiting for orchestrator to start server"}


def stop_server(server_id: str) -> dict[str, Any]:
    """Request orchestrator to stop a server."""
    request = {"action": "stop", "server_id": server_id}

    with open(".agent_context/server_requests.jsonl", 'a') as f:
        f.write(json.dumps(request) + '\n')

    response = _wait_for_server_response(timeout=5.0)

    if response:
        _ledger_append("SERVER", f"stop {server_id} -> {response.get('success', False)}")

    return response or {"error": "Timeout waiting for response"}


def check_server(server_id: str, tail_lines: int = 20) -> dict[str, Any]:
    """Check server status and get recent logs."""
    request = {"action": "check", "server_id": server_id, "tail_lines": tail_lines}

    with open(".agent_context/server_requests.jsonl", 'a') as f:
        f.write(json.dumps(request) + '\n')

    response = _wait_for_server_response(timeout=5.0)
    return response or {"error": "Timeout waiting for response"}


def list_servers() -> dict[str, Any]:
    """List all running servers."""
    request = {"action": "list"}

    with open(".agent_context/server_requests.jsonl", 'a') as f:
        f.write(json.dumps(request) + '\n')

    response = _wait_for_server_response(timeout=5.0)
    return response or {"error": "Timeout waiting for response"}


def _wait_for_server_response(timeout: float = 5.0) -> dict[str, Any] | None:
    """
    Wait for orchestrator response.

    Polls response file for new line matching our request.
    """
    response_file = Path(".agent_context/server_responses.jsonl")

    # Count existing lines to know where to start reading
    existing_lines = 0
    if response_file.exists():
        with open(response_file, 'r') as f:
            existing_lines = len(f.readlines())

    start_time = time.time()

    while time.time() - start_time < timeout:
        if response_file.exists():
            with open(response_file, 'r') as f:
                lines = f.readlines()

            # Check for new lines
            if len(lines) > existing_lines:
                # Return the newest response
                response_line = lines[-1].strip()
                if response_line:
                    return json.loads(response_line)

        time.sleep(0.1)

    return None

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

    # Log visible progress
    if success:
        current_task = _ctx._get_current_task()
        if current_task:
            completed = sum(1 for st in current_task.subtasks if st.status == "completed")
            total = len(current_task.subtasks)
            log(f"✓ Progress: {completed}/{total} subtasks complete ({completed/total*100:.0f}%)")
            print(f"\n{'='*70}")
            print(f"✓ SUBTASK COMPLETE: {reason if reason else 'success'}")
            print(f"Progress: {completed}/{total} subtasks ({completed/total*100:.0f}%)")
            print(f"{'='*70}\n")

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

# ----------------------------
# Phase 2: Agent-Driven Escalation (NO GIVE-UP)
# ----------------------------
def agent_decide_escalation(
    subtask: Any,
    reason: str
) -> str:
    """
    Decide escalation strategy based on config and subtask depth.

    Strategy:
    - If can decompose AND strategy is "force_decompose": Always decompose
    - If can decompose AND strategy is "agent_decides": Ask agent
    - If at max depth: Force zoom out to reconsider approach
    - Never give up - always either decompose or zoom out

    Returns: "decompose" or "zoom_out" (never "give_up")
    """
    # Check if decomposition is possible
    can_decompose = subtask.can_add_child()

    # Force decompose strategy from config
    if config.escalation.strategy == "force_decompose":
        if can_decompose:
            log(f"[escalation] Forcing decomposition (depth {subtask.depth}/{MAX_SUBTASK_DEPTH})")
            return "decompose"
        else:
            log(f"[escalation] At max depth/width, zooming out to {config.escalation.zoom_out_target}")
            return "zoom_out"

    # Agent decides strategy (but no give-up option)
    if config.escalation.strategy == "agent_decides":
        if not can_decompose:
            # No choice - must zoom out
            log(f"[escalation] At max depth/width, forced zoom out")
            return "zoom_out"

        # Build escalation prompt (NO GIVE-UP OPTION)
        escalation_prompt = f"""ESCALATION NEEDED: You've spent {subtask.rounds_used} rounds on this subtask without completing it.

Current subtask: {subtask.description}
Reason: {reason}
Depth in hierarchy: {subtask.depth}/{MAX_SUBTASK_DEPTH}
Recent actions: {[a.name for a in subtask.actions[-5:]]}

You have TWO options:

A) DECOMPOSE - Break this subtask into {config.decomposition.min_children}-{config.decomposition.max_children} smaller, more specific subtasks
   - Use this when: The subtask has multiple distinct steps that can be done independently
   - Benefit: Each smaller subtask is easier to complete and verify
   - You have {MAX_SUBTASK_DEPTH - subtask.depth} levels remaining

B) ZOOM OUT - Save progress and try a completely different strategy
   - Use this when: Current approach is fundamentally flawed and you need to reconsider
   - What happens: System will zoom to {config.escalation.zoom_out_target} level and reconsider approach
   - Benefit: Fresh perspective on the problem

IMPORTANT: Giving up is NOT an option. You must either decompose or zoom out.

What should you do? Respond with ONLY:
- "DECOMPOSE: <brief reason>" to break into smaller subtasks
- "ZOOM_OUT: <brief reason>" to reconsider approach

Your decision:"""

        # Ask the LLM to decide
        try:
            response = OLLAMA_CLIENT.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": escalation_prompt}
                ],
                options={"temperature": 0.3}  # Slightly higher temp for decision-making
            )

            decision_text = response["message"]["content"].strip().upper()

            # Parse decision (only decompose or zoom_out allowed)
            if "DECOMPOSE" in decision_text:
                log(f"Agent chose DECOMPOSE: {decision_text}")
                return "decompose"
            else:
                # Default to zoom_out if unclear
                log(f"Agent chose ZOOM_OUT: {decision_text}")
                return "zoom_out"
        except Exception as e:
            log(f"Escalation decision failed: {e}, defaulting to zoom_out")
            return "zoom_out"

    # Fallback: decompose if possible, otherwise zoom out
    return "decompose" if can_decompose else "zoom_out"


def find_smart_zoom_target(current_subtask: Any, ctx: ContextManager) -> str:
    """
    Analyze the subtask tree to find the best zoom-out target.

    Strategy:
    1. If current subtask has only 1-2 failed siblings, zoom to parent (localized issue)
    2. If parent has mostly failed children, zoom to task (systemic issue in parent's approach)
    3. If task has multiple failed branches, zoom to root (fundamental approach problem)
    4. Otherwise, zoom to parent (conservative default)

    Returns: "parent", "task", or "root"
    """
    # Get parent subtask if exists
    parent = current_subtask.parent_subtask if hasattr(current_subtask, 'parent_subtask') else None

    # If at top level (no parent), must go to task
    if not parent:
        log("[smart_zoom] No parent subtask, zooming to task")
        return "task"

    # Analyze siblings (other children of parent)
    siblings = parent.child_subtasks if hasattr(parent, 'child_subtasks') else []
    if siblings:
        failed_siblings = [s for s in siblings if s.status in ("failed", "blocked")]
        total_siblings = len(siblings)

        # If less than half of siblings failed, problem is localized
        if len(failed_siblings) < total_siblings / 2:
            log(f"[smart_zoom] Only {len(failed_siblings)}/{total_siblings} siblings failed, zooming to parent")
            return "parent"

    # Analyze parent's status - if parent itself is struggling, go higher
    if parent.rounds_used > config.rounds.max_per_subtask * 0.7:
        log(f"[smart_zoom] Parent has {parent.rounds_used} rounds, zooming to task")
        return "task"

    # Check if multiple branches at task level are failing
    current_task = ctx.state.goal.tasks[ctx.state.current_task_idx]
    task_subtasks = current_task.subtasks
    if task_subtasks:
        failed_task_subtasks = [s for s in task_subtasks if s.status in ("failed", "blocked")]

        # If more than 2/3 of top-level subtasks failed, fundamental issue
        if len(failed_task_subtasks) > len(task_subtasks) * 2 / 3:
            log(f"[smart_zoom] {len(failed_task_subtasks)}/{len(task_subtasks)} task subtasks failed, zooming to root")
            return "root"

    # Default: zoom to parent (conservative)
    log("[smart_zoom] No clear pattern, zooming to parent")
    return "parent"


def generate_failure_report(goal: str, ctx: ContextManager, reason: str) -> str:
    """
    Generate a comprehensive failure report for debugging and analysis.

    Returns the path to the generated report.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/failure_report_{timestamp}.md"

    # Ensure reports directory exists
    Path("reports").mkdir(exist_ok=True)

    lines = []
    lines.append(f"# Agent Failure Report")
    lines.append(f"")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Goal:** {goal}")
    lines.append(f"**Failure Reason:** {reason}")
    lines.append(f"")

    if not ctx.state.goal:
        lines.append("No goal state available.")
        Path(report_path).write_text("\n".join(lines), encoding="utf-8")
        return report_path

    # Overall summary
    lines.append("## Summary")
    lines.append(f"")
    total_tasks = len(ctx.state.goal.tasks)
    completed_tasks = sum(1 for t in ctx.state.goal.tasks if t.status == "completed")
    lines.append(f"- **Tasks Completed:** {completed_tasks}/{total_tasks}")
    lines.append(f"- **Current Task Index:** {ctx.state.current_task_idx}")
    lines.append(f"")

    # Task breakdown
    lines.append("## Task Breakdown")
    lines.append("")

    for i, task in enumerate(ctx.state.goal.tasks):
        is_current = i == ctx.state.current_task_idx
        marker = "**[CURRENT]**" if is_current else ""

        lines.append(f"### Task {i+1}: {task.description} {marker}")
        lines.append(f"")
        lines.append(f"- **Status:** {task.status}")
        if hasattr(task, 'approach_attempts') and task.approach_attempts > 0:
            lines.append(f"- **Approach Attempts:** {task.approach_attempts}")
        if hasattr(task, 'failed_approaches') and task.failed_approaches:
            lines.append(f"- **Failed Approaches:**")
            for fa in task.failed_approaches:
                lines.append(f"  - {fa}")
        lines.append(f"")

        # Subtask breakdown
        if task.subtasks:
            total_subtasks = len(task.subtasks)
            completed_subtasks = sum(1 for st in task.subtasks if st.status == "completed")
            lines.append(f"**Subtasks:** {completed_subtasks}/{total_subtasks} completed")
            lines.append(f"")

            for j, subtask in enumerate(task.subtasks):
                _write_subtask_report(lines, subtask, indent=0)

        lines.append(f"")

    # Blockers and issues
    lines.append("## Identified Blockers")
    lines.append("")

    blockers = []
    for task in ctx.state.goal.tasks:
        for subtask in task.subtasks:
            if subtask.status in ["blocked", "failed"]:
                blocker = f"- **{subtask.description}**"
                if subtask.failure_reason:
                    blocker += f": {subtask.failure_reason}"
                if hasattr(subtask, 'tried_approaches') and subtask.tried_approaches:
                    blocker += f"\n  - Tried: {', '.join(subtask.tried_approaches[:3])}"
                blockers.append(blocker)

    if blockers:
        lines.extend(blockers)
    else:
        lines.append("No specific blockers identified.")

    lines.append(f"")

    # Progress achieved
    lines.append("## Progress Achieved")
    lines.append("")

    accomplishments = []
    for task in ctx.state.goal.tasks:
        for subtask in task.subtasks:
            if hasattr(subtask, 'accomplishments') and subtask.accomplishments:
                for acc in subtask.accomplishments:
                    accomplishments.append(f"- {acc}")

    if accomplishments:
        lines.extend(accomplishments)
    else:
        lines.append("No measurable progress recorded.")

    lines.append(f"")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("Based on this failure analysis:")
    lines.append("1. Review the blockers listed above")
    lines.append("2. Check if the goal is achievable with current tools")
    lines.append("3. Consider breaking down blocked subtasks further")
    lines.append("4. Review failed approaches to avoid repeating them")
    lines.append("")

    # Write report
    report_content = "\n".join(lines)
    Path(report_path).write_text(report_content, encoding="utf-8")

    log(f"[report] Generated failure report: {report_path}")
    return report_path


def _write_subtask_report(lines: list[str], subtask: Any, indent: int) -> None:
    """Helper to recursively write subtask info to report."""
    indent_str = "  " * indent

    # Subtask header
    status_emoji = {"completed": "✓", "failed": "✗", "blocked": "⊘", "in_progress": "⟳", "pending": "○"}
    emoji = status_emoji.get(subtask.status, "?")

    lines.append(f"{indent_str}- {emoji} **{subtask.description}**")
    lines.append(f"{indent_str}  - Status: {subtask.status}")

    if hasattr(subtask, 'depth'):
        lines.append(f"{indent_str}  - Depth: Level {subtask.depth}")
    if hasattr(subtask, 'rounds_used') and subtask.rounds_used > 0:
        lines.append(f"{indent_str}  - Rounds Used: {subtask.rounds_used}")

    # Accomplishments
    if hasattr(subtask, 'accomplishments') and subtask.accomplishments:
        lines.append(f"{indent_str}  - **Accomplishments:**")
        for acc in subtask.accomplishments:
            lines.append(f"{indent_str}    - {acc}")

    # Tried approaches
    if hasattr(subtask, 'tried_approaches') and subtask.tried_approaches:
        lines.append(f"{indent_str}  - **Tried (failed):**")
        for tried in subtask.tried_approaches[:5]:  # Limit to 5
            lines.append(f"{indent_str}    - {tried}")

    # Context notes
    if hasattr(subtask, 'context_notes') and subtask.context_notes:
        lines.append(f"{indent_str}  - **Notes:** {subtask.context_notes}")

    # Failure reason
    if subtask.status in ["failed", "blocked"] and subtask.failure_reason:
        lines.append(f"{indent_str}  - **Failure Reason:** {subtask.failure_reason}")

    lines.append("")

    # Recurse for child subtasks
    if hasattr(subtask, 'child_subtasks') and subtask.child_subtasks:
        for child in subtask.child_subtasks:
            _write_subtask_report(lines, child, indent + 1)


def reconsider_approach_at_root(task: Any, ctx: ContextManager) -> bool:
    """
    Zoom out to root and reconsider the entire approach.

    Returns True if approach was reconsidered and we should retry.
    Returns False if max retries exhausted and task should be marked failed.
    """
    if not config.approach_retry.enabled:
        log("[approach] Approach retry disabled in config")
        return False

    # Check if we've exceeded max retries
    if task.approach_attempts >= config.escalation.max_approach_retries:
        log(f"[approach] Max approach retries ({config.escalation.max_approach_retries}) exceeded")
        return False

    # Increment attempt counter
    task.approach_attempts += 1

    # Collect what we've learned
    failed_subtasks = [st for st in task.subtasks if st.status in ["failed", "blocked"]]
    failed_approaches = [st.description for st in failed_subtasks]

    # Save failed approach summary
    approach_summary = f"Attempt {task.approach_attempts}: Failed subtasks: {', '.join(failed_approaches[:3])}"
    task.failed_approaches.append(approach_summary)

    log(f"[approach] Reconsidering approach (attempt {task.approach_attempts}/{config.escalation.max_approach_retries})")
    print(f"\n{'='*70}")
    print(f"🔄 RECONSIDERING APPROACH (Attempt {task.approach_attempts}/{config.escalation.max_approach_retries})")
    print(f"Task: {task.description}")
    print(f"\nPrevious failed approaches:")
    for i, summary in enumerate(task.failed_approaches, 1):
        print(f"  {i}. {summary}")
    print(f"{'='*70}\n")

    # Reset or preserve based on config
    if config.approach_retry.reset_subtasks_on_retry:
        if config.approach_retry.preserve_completed:
            # Keep completed subtasks, reset others
            task.subtasks = [st for st in task.subtasks if st.status == "completed"]
            log(f"[approach] Preserved {len(task.subtasks)} completed subtasks")
        else:
            # Complete reset
            task.subtasks = []
            log("[approach] Complete reset - all subtasks cleared")

    # Collect detailed context from failed subtasks
    failed_context = []
    for st in failed_subtasks:
        context = f"{st.description}"
        if hasattr(st, 'accomplishments') and st.accomplishments:
            context += f"\n  Accomplished: {', '.join(st.accomplishments[:3])}"
        if hasattr(st, 'tried_approaches') and st.tried_approaches:
            context += f"\n  Tried: {', '.join(st.tried_approaches[:3])}"
        if hasattr(st, 'context_notes') and st.context_notes:
            context += f"\n  Notes: {st.context_notes}"
        failed_context.append(context)

    # Ask LLM to reconsider approach
    if config.approach_retry.retry_style == "learn_from_failures":
        reconsider_prompt = f"""APPROACH RECONSIDERATION NEEDED

Task: {task.description}
Attempt: {task.approach_attempts}/{config.escalation.max_approach_retries}

Previous approaches that FAILED:
{chr(10).join(f"- {fa}" for fa in task.failed_approaches)}

DETAILED FAILURE CONTEXT:
{chr(10).join(failed_context) if failed_context else "(no details available)"}

What worked (completed subtasks):
{chr(10).join(f"✓ {st.description}" for st in task.subtasks if st.status == "completed") or "(none)"}

ANALYSIS - What we learned from failures:
- Some actions succeeded (see accomplishments above)
- Some approaches failed (see tried approaches above)
- We need a DIFFERENT strategy that avoids the same blockers

You need to propose a COMPLETELY DIFFERENT approach. Consider:
1. What assumptions were wrong in previous attempts?
2. Is there a simpler, more direct path to the goal?
3. Are we solving the right problem?
4. What alternative strategies haven't been tried?
5. What succeeded that we should keep doing?

Return a JSON array of NEW subtasks for a FRESH approach:
["New subtask 1", "New subtask 2", ...]

Be creative and avoid repeating previous failed patterns.
Your new approach (JSON only):"""
    else:  # fresh_start
        reconsider_prompt = f"""Task: {task.description}

Previous attempts failed. Start fresh with a new approach.

Return a JSON array of subtasks:
["Subtask 1", "Subtask 2", ...]"""

    try:
        response = OLLAMA_CLIENT.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a strategic problem solver who learns from failures."},
                {"role": "user", "content": reconsider_prompt}
            ],
            options={"temperature": 0.4}  # Higher creativity for new approaches
        )

        content = response["message"]["content"].strip()

        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        from context_manager import Subtask
        new_subtasks_data = json.loads(content)

        # Create new subtasks
        new_subtasks = []
        for desc in new_subtasks_data[:config.decomposition.max_children]:
            new_st = Subtask(
                description=desc,
                status="pending",
                depth=1,
                parent_subtask=""
            )
            new_subtasks.append(new_st)

        # Append new subtasks to task
        task.subtasks.extend(new_subtasks)

        # Mark first new subtask as in_progress
        if new_subtasks:
            new_subtasks[0].status = "in_progress"

        # Reset task status
        task.status = "in_progress"

        ctx._save_state()

        log(f"[approach] Created {len(new_subtasks)} new subtasks for fresh approach")
        print(f"✓ New approach with {len(new_subtasks)} subtasks:")
        for i, st in enumerate(new_subtasks, 1):
            print(f"  {i}. {st.description}")
        print()

        return True

    except Exception as e:
        log(f"[approach] Failed to reconsider: {e}")
        return False


def agent_decompose_subtask(
    parent_subtask: Any
) -> list[Any]:
    """
    Agent decomposes current subtask into smaller children.
    Agent decides WHAT the subtasks should be.
    """
    granularity_hint = "VERY GRANULAR - prefer MORE subtasks over fewer" if config.decomposition.prefer_granular else "balanced"

    decompose_prompt = f"""You need to break down this subtask into smaller, VERY SPECIFIC pieces:

Current subtask: {parent_subtask.description}
Actions taken so far: {len(parent_subtask.actions)}
What worked: {[a.name for a in parent_subtask.actions if a.result == "success"]}
What failed: {[a.name for a in parent_subtask.actions if a.result == "error"]}

IMPORTANT: Break this into {config.decomposition.min_children}-{config.decomposition.max_children} TINY, CONCRETE subtasks ({granularity_hint}). Each subtask should:
- Be completable in 2-4 actions MAX
- Produce visible, verifiable output (a file, test passing, etc.)
- Have ZERO ambiguity about what "done" means
- Be as granular as possible - prefer MORE subtasks over fewer

Return a JSON array of subtask descriptions:
["Subtask 1 description", "Subtask 2 description", ...]

GOOD Example (GRANULAR) for "Create calculator with tests":
["Write calculator.py with ONLY add function",
 "Write test_calculator.py with ONLY test_add",
 "Run pytest on test_add and verify it passes",
 "Add subtract function to calculator.py",
 "Add test_subtract to test file",
 "Run full pytest suite"]

BAD Example (TOO BROAD):
["Create calculator.py with all functions",
 "Write comprehensive tests",
 "Make sure everything works"]

Your decomposition (JSON only - be VERY GRANULAR):"""

    try:
        response = OLLAMA_CLIENT.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a task decomposition expert."},
                {"role": "user", "content": decompose_prompt}
            ],
            options={"temperature": config.decomposition.temperature}
        )

        content = response["message"]["content"].strip()

        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            from context_manager import Subtask
            subtask_descriptions = json.loads(content)

            # Create child subtasks
            children = []
            for desc in subtask_descriptions[:MAX_SUBTASK_SIBLINGS]:  # Limit siblings
                child = Subtask(
                    description=desc,
                    status="pending",
                    depth=parent_subtask.depth + 1,
                    parent_subtask=parent_subtask.description
                )
                children.append(child)

            parent_subtask.child_subtasks = children
            log(f"Decomposed into {len(children)} child subtasks")
            return children

        except json.JSONDecodeError:
            log(f"Failed to parse decomposition: {content}")
            return []
    except Exception as e:
        log(f"Decomposition failed: {e}")
        return []


# Tool dispatch - uses tools module
# Wrapper functions inject global context (_ctx, _workspace) into tool calls
def _mark_subtask_complete_wrapper(success: bool = True, reason: str = "") -> dict[str, Any]:
    """Wrapper to inject context manager into tools.mark_subtask_complete."""
    global _ctx
    return tools.mark_subtask_complete(success, reason, context_manager=_ctx)

TOOLS = {
    "list_dir": tools.list_dir,
    "read_file": tools.read_file,
    "grep_file": tools.grep_file,
    "write_file": tools.write_file,
    "run_cmd": tools.run_cmd,
    "start_server": tools.start_server,
    "stop_server": tools.stop_server,
    "check_server": tools.check_server,
    "list_servers": tools.list_servers,
    "mark_subtask_complete": _mark_subtask_complete_wrapper,
}

def tool_specs() -> list[dict[str, Any]]:
    """Return tool specifications for LLM function calling."""
    # Use tool definitions from tools.py module
    return tools.get_tool_definitions()

def tool_specs_legacy() -> list[dict[str, Any]]:
    """Legacy tool specs function - kept for reference, not used."""
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
    return_legacy = [
        spec("list_dir", "List files (non-recursive).", {"path": {"type": "string"}}),
        spec("read_file", "Read a text file with optional pagination. Use offset to read large files in chunks.", {
            "path": {"type": "string", "required": True},
            "max_bytes": {"type": "number", "description": "Maximum bytes to read (default: 200000). Files are truncated if larger."},
            "offset": {"type": "number", "description": "Byte offset to start reading from (default: 0). Use for pagination of large files."},
        }),
        spec("grep_file", "Search for regex pattern in a file and return matching lines with context. More efficient than read_file for finding specific content.", {
            "path": {"type": "string", "required": True},
            "pattern": {"type": "string", "required": True, "description": "Python regex pattern to search for"},
            "context_lines": {"type": "number", "description": "Number of context lines before/after each match (default: 3)"},
            "max_matches": {"type": "number", "description": "Maximum number of matches to return (default: 50)"},
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
        spec("start_server", "Start a web server or long-running process in background (managed by orchestrator). "
             "Server persists across task executions. Use for HTTP servers, dev servers, etc. "
             "Returns immediately - server runs in background.", {
            "cmd": {"type": "array", "items": {"type": "string"}, "required": True,
                    "description": "Command to run, e.g. ['python', '-m', 'http.server', '8000']"},
            "name": {"type": "string", "description": "Optional server name (e.g. 'webapp'). Auto-generated if not provided."},
        }),
        spec("stop_server", "Stop a running background server.", {
            "server_id": {"type": "string", "required": True, "description": "Server ID returned by start_server"},
        }),
        spec("check_server", "Check server status and get recent log output.", {
            "server_id": {"type": "string", "required": True, "description": "Server ID to check"},
            "tail_lines": {"type": "number", "description": "Number of recent log lines to return (default: 20)"},
        }),
        spec("list_servers", "List all running background servers.", {}),
        spec("mark_subtask_complete", "Mark current subtask as complete. Call this when you finish a subtask.", {
            "success": {"type": "boolean", "description": "True if completed successfully, False if failed"},
            "reason": {"type": "string", "description": "Optional explanation (required if failed)"},
        }),
    ]

# ----------------------------
# Task decomposition
# ----------------------------
def decompose_goal(goal: str, is_edit_mode: bool = False, additional_context: str = "") -> list[dict[str, Any]]:
    """Ask LLM to decompose goal into tasks and subtasks."""

    # If in edit mode, add exploration context to the prompt
    edit_mode_guidance = ""
    if is_edit_mode:
        edit_mode_guidance = """
IMPORTANT: This is an EDIT MODE task - you are modifying an existing project.
- The first task MUST be to explore and understand the current project structure
- Use list_dir and read_file to understand what exists before making changes
- Do NOT assume the project structure - you must discover it first
"""

    # Add additional context if provided
    context_section = ""
    if additional_context:
        context_section = f"""
Additional Context:
{additional_context}
"""

    prompt = f"""Break down this goal into CONCRETE, ACTIONABLE tasks and subtasks.

Goal: {goal}
{edit_mode_guidance}{context_section}
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

Example for "Add feature to existing project" (EDIT MODE):
[
  {{"description": "Understand current project structure", "subtasks": ["List files in current directory", "Read main application files", "Identify where feature should be added"]}},
  {{"description": "Implement new feature", "subtasks": ["Modify relevant files with new feature code"]}},
  {{"description": "Test changes", "subtasks": ["Run tests to verify feature works"]}}
]

Return ONLY the JSON array, no other text.
"""

    log("Decomposing goal into tasks...")
    try:
        resp = chat_with_inactivity_timeout(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
            inactivity_timeout=30,  # 30s of no activity = Ollama is hung
        )
    except TimeoutError as e:
        log(f"Ollama stopped responding during decomposition: {e}")
        # Fallback: create a basic task structure
        if is_edit_mode:
            return [
                {"description": "Understand current project", "subtasks": ["List files in workspace", "Read key files to understand structure"]},
                {"description": "Implement changes", "subtasks": ["Make necessary code changes"]},
                {"description": "Verify and test", "subtasks": ["Run tests", "Check code quality"]}
            ]
        else:
            return [
                {"description": "Understand and plan", "subtasks": ["Read relevant files", "Identify what needs to be done"]},
                {"description": "Implement changes", "subtasks": ["Make necessary code changes"]},
                {"description": "Verify and test", "subtasks": ["Run tests", "Check code quality"]}
            ]

    content = resp["message"]["content"].strip()
    # Extract JSON from markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        tasks_data = json.loads(content)
        log(f"Decomposed into {len(tasks_data)} tasks")

        # If in edit mode and LLM forgot to add exploration task, inject it
        if is_edit_mode and tasks_data:
            first_task = tasks_data[0]
            # Check if first task involves exploration
            has_exploration = any(
                keyword in first_task["description"].lower()
                for keyword in ["understand", "explore", "analyze", "examine", "inspect", "review"]
            )
            if not has_exploration:
                # Inject exploration task at the beginning
                log("Injecting exploration task for edit mode")
                exploration_task = {
                    "description": "Understand current project structure",
                    "subtasks": [
                        "List all files in workspace directory",
                        "Read key project files to understand architecture",
                        "Document current structure for subsequent tasks"
                    ]
                }
                tasks_data.insert(0, exploration_task)

        return tasks_data
    except json.JSONDecodeError as e:
        log(f"Failed to parse task decomposition: {e}")
        # Fallback: create a single generic task
        if is_edit_mode:
            return [
                {"description": "Understand project structure", "subtasks": ["List workspace files", "Read main files"]},
                {"description": goal, "subtasks": ["Complete the goal"]}
            ]
        else:
            return [{"description": goal, "subtasks": ["Complete the goal"]}]

# ----------------------------
# Context building
# ----------------------------
def build_context(ctx: ContextManager, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build context with: system prompt + current task info + last N messages."""
    # Use shared hierarchical context strategy
    return build_hierarchical_context(
        context_manager=ctx,
        messages=messages,
        system_prompt=config.llm.system_prompt,
        config=config,
        probe_state_func=probe_state_generic,  # Pass probe function for filesystem state
    )

# ----------------------------
# Dispatch tool calls
# ----------------------------
def dispatch(call: dict[str, Any]) -> dict[str, Any]:
    import traceback
    name = call["function"]["name"]
    args = call["function"].get("arguments", "{}")

    log(f"TOOL→ {name} args={args[:200] if isinstance(args, str) else str(args)[:200]}")

    # Check if action is blocked (loop detected)
    data = json.loads(args) if isinstance(args, str) else (args or {})
    action_sig = f"{name}::{json.dumps(data, sort_keys=True)}"

    if _ctx and action_sig in _ctx.state.blocked_actions:
        # Track repeated blocked attempts
        attempt_count = _ctx.add_blocked_attempt(action_sig)

        # If too many blocked attempts, force escalation
        if attempt_count >= 3:
            log(f"TOOL✖ {name} blocked and repeatedly attempted ({attempt_count}x) - FORCING ESCALATION")
            return {
                "error": f"Action blocked due to repetition (attempted {attempt_count} times after blocking). "
                         f"This indicates a fundamental issue. Please try a COMPLETELY different approach or escalate.",
                "force_escalate": True,
            }

        log(f"TOOL✖ {name} blocked (loop detected, attempt {attempt_count})")
        return {
            "error": f"Action blocked due to repetition (tried {_ctx.state.loop_counts.get(action_sig, 0)} times). "
                     f"Try a different approach.",
        }

    fn = TOOLS.get(name)
    if not fn:
        log(f"TOOL✖ unknown: {name}")
        return {"error": f"unknown tool {name}"}
    try:
        out = fn(**data) if data else fn()
        log(f"TOOL✓ {name} → {type(out).__name__}")
        return {"result": out}
    except Exception as e:
        log(f"TOOL✖ {name} error={e}")
        log(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

# ----------------------------
# Ollama Health Check - NOW IN llm_utils.py
# ----------------------------
# check_ollama_health moved to llm_utils.py
# (imported at top of file)


def wait_for_ollama(max_wait: int = 30, check_interval: int = 5) -> bool:
    """Wait for Ollama to become responsive, with retry logic."""
    import time

    print("[info] Checking Ollama health...")

    if check_ollama_health():
        print("[info] Ollama is responsive")
        return True

    print(f"[warning] Ollama not responding. Waiting up to {max_wait}s for recovery...")

    elapsed = 0
    while elapsed < max_wait:
        time.sleep(check_interval)
        elapsed += check_interval

        if check_ollama_health():
            print(f"[info] Ollama recovered after {elapsed}s")
            return True

        print(f"[warning] Still waiting... ({elapsed}s/{max_wait}s)")

    return False


# ----------------------------
# Main loop
# ----------------------------
def main() -> None:
    global _ctx, _workspace

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Jetbox coding agent")
    parser.add_argument("goal", nargs="*", help="Goal/task description")
    parser.add_argument("--workspace", "-w", type=str, default=None,
                       help="Edit mode: work in this existing directory (default: create isolated workspace)")
    parser.add_argument("--context", "-c", type=str, default=None,
                       help="Additional context about the task (e.g., specific requirements, constraints)")
    args = parser.parse_args()

    goal = " ".join(args.goal) if args.goal else "Write a hello world script"
    workspace_path = args.workspace
    additional_context = args.context

    # Check Ollama health BEFORE doing anything else
    if not wait_for_ollama(max_wait=30, check_interval=5):
        print("[error] Ollama is not responding after 30s. Cannot start agent.")
        print("[error] Please ensure Ollama is running: ollama serve")
        sys.exit(1)

    log(f"Starting agent with goal: {goal}")

    # Add session marker to ledger
    from datetime import datetime
    _ledger_append("SESSION_START", f"{datetime.now().isoformat()} | {goal[:80]}")

    # Initialize workspace manager
    if workspace_path:
        # Edit mode: work in specified existing directory
        _workspace = WorkspaceManager(goal, workspace_path=workspace_path)
        log(f"Mode: EDIT (working in existing directory)")
        log(f"Workspace: {_workspace.workspace_dir}")
    else:
        # Isolate mode: create isolated directory for this goal
        _workspace = WorkspaceManager(goal)
        log(f"Mode: ISOLATE (isolated workspace)")
        log(f"Workspace: {_workspace.workspace_dir}")

    # Configure tools module with workspace and ledger
    tools.set_workspace(_workspace)
    tools.set_ledger(LEDGER)

    # Initialize context manager
    _ctx = ContextManager()
    _ctx.load_or_init(goal)

    # Initialize status display (reset stats if new goal)
    status = StatusDisplay(_ctx, reset_stats=_ctx.is_new_goal)
    _ctx.loop_callback = status.record_loop  # Wire up loop detection callback

    # If new goal, decompose into tasks
    if not _ctx.state.goal or not _ctx.state.goal.tasks:
        tasks_data = decompose_goal(
            goal,
            is_edit_mode=_workspace.is_edit_mode,
            additional_context=additional_context or ""
        )

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

        # Show initial task tree immediately after decomposition
        print("\n" + "=" * 70)
        print("INITIAL TASK TREE")
        print("=" * 70)
        print()
        print(status.render_compact())
        print()

    # Message history (just the conversation, not context info)
    messages: list[dict[str, Any]] = []

    # Phase 2: Track rounds per subtask
    # Turn counter increments by 1 per turn, resets to 0 when moving to next subtask
    current_subtask_rounds = 0
    last_subtask_sig = None
    task_total_rounds = 0  # Safety cap per task

    for round_no in range(1, MAX_ROUNDS + 1):
        # Update probe state in context manager
        probe = probe_state_generic()
        _ctx.update_probe_state(probe)

        # Phase 2: Get current subtask and track rounds
        # IMPORTANT: Capture subtask at START of round for accurate counter tracking
        round_start_subtask = None
        round_start_sig = None
        current_task = _ctx._get_current_task()
        if current_task:
            current_subtask = current_task.active_subtask()
            if current_subtask:
                current_sig = current_subtask.signature()
                round_start_subtask = current_subtask
                round_start_sig = current_sig

                # Reset counter if subtask changed
                # Turn counter always resets to 0 when moving to next task/subtask
                if current_sig != last_subtask_sig:
                    current_subtask_rounds = 0
                    last_subtask_sig = current_sig

                # Check per-subtask round limit
                if current_subtask_rounds >= MAX_ROUNDS_PER_SUBTASK:
                    log(f"Subtask '{current_subtask.description}' hit {MAX_ROUNDS_PER_SUBTASK} rounds")

                    # Agent decides: Decompose or zoom out
                    escalation_decision = agent_decide_escalation(
                        current_subtask,
                        reason="max_rounds_per_subtask"
                    )

                    if escalation_decision == "decompose":
                        # Agent will create child subtasks
                        children = agent_decompose_subtask(current_subtask)
                        if children:
                            # Mark current subtask as having children, move to first child
                            current_subtask.status = "decomposed"
                            children[0].status = "in_progress"
                            current_subtask_rounds = 0  # Reset for decomposition
                            last_subtask_sig = None
                            _ctx._save_state()
                            log(f"Decomposed into {len(children)} subtasks, starting with: {children[0].description}")

                            # Show decomposition to user
                            print(f"\n{'='*70}")
                            print(f"🔀 DECOMPOSING: {current_subtask.description}")
                            print(f"Created {len(children)} granular subtasks:")
                            for i, child in enumerate(children, 1):
                                print(f"  {i}. {child.description}")
                            print(f"Starting with: {children[0].description}")
                            print(f"{'='*70}\n")
                            continue
                        else:
                            log("Decomposition failed, falling back to zoom_out")
                            escalation_decision = "zoom_out"

                    if escalation_decision == "zoom_out":
                        # Mark current subtask as blocked
                        current_subtask.status = "blocked"
                        current_subtask.failure_reason = f"Hit {MAX_ROUNDS_PER_SUBTASK} rounds, zooming out"
                        _ctx._save_state()

                        # Determine zoom target based on config
                        zoom_target_config = config.escalation.zoom_out_target

                        # Smart zoom: analyze tree to find best target
                        if zoom_target_config == "smart":
                            zoom_target = find_smart_zoom_target(current_subtask, _ctx)
                            log(f"[smart_zoom] Determined target: {zoom_target}")
                        else:
                            zoom_target = zoom_target_config

                        if zoom_target == "root":
                            # Zoom all the way to root and reconsider approach
                            log(f"[zoom] Zooming to root to reconsider approach")
                            retry_success = reconsider_approach_at_root(current_task, _ctx)

                            if retry_success:
                                # Approach reconsidered, reset round counters and continue
                                current_subtask_rounds = 0
                                last_subtask_sig = None
                                task_total_rounds = 0
                                log("[zoom] Continuing with new approach")
                                continue
                            else:
                                # Max retries exhausted - generate failure report
                                log("[zoom] Max approach retries exhausted, task failed")
                                current_task.status = "failed"
                                _ctx._save_state()

                                # Generate comprehensive failure report
                                report_path = generate_failure_report(
                                    goal,
                                    _ctx,
                                    f"Max approach retries ({config.escalation.max_approach_retries}) exhausted on task: {current_task.description}"
                                )
                                print(f"\n{'='*70}")
                                print(f"❌ TASK FAILED AFTER {config.escalation.max_approach_retries} RETRY ATTEMPTS")
                                print(f"Task: {current_task.description}")
                                print(f"\n📊 Detailed failure report generated: {report_path}")
                                print(f"{'='*70}\n")
                                break

                        elif zoom_target == "task":
                            # Try next subtask in current task
                            has_next = _ctx.advance_to_next_subtask()
                            if not has_next:
                                # No more subtasks, reconsider task approach
                                log("[zoom] No more subtasks, reconsidering task approach")
                                retry_success = reconsider_approach_at_root(current_task, _ctx)
                                if not retry_success:
                                    current_task.status = "failed"
                                    _ctx._save_state()
                                    report_path = generate_failure_report(
                                        goal, _ctx,
                                        f"No more subtasks and max retries exhausted: {current_task.description}"
                                    )
                                    print(f"\n📊 Failure report: {report_path}\n")
                                    break
                                task_total_rounds = 0
                            current_subtask_rounds = 0
                            last_subtask_sig = None
                            continue

                        else:  # "parent"
                            # Move to parent or next sibling
                            has_next = _ctx.advance_to_next_subtask()
                            if not has_next:
                                # At end of task, reconsider
                                retry_success = reconsider_approach_at_root(current_task, _ctx)
                                if not retry_success:
                                    current_task.status = "failed"
                                    _ctx._save_state()
                                    report_path = generate_failure_report(
                                        goal, _ctx,
                                        f"Exhausted all retry attempts: {current_task.description}"
                                    )
                                    print(f"\n📊 Failure report: {report_path}\n")
                                    break
                                task_total_rounds = 0
                            current_subtask_rounds = 0
                            last_subtask_sig = None
                            continue

                # Per-task limit removed - using MAX_ROUNDS (global) and MAX_ROUNDS_PER_SUBTASK instead

        # Set activity: Thinking
        status.set_activity("thinking: planning next actions")

        # Calculate accurate context stats for visualization
        # Build the actual context to measure it
        full_context = build_context(_ctx, messages)

        # Calculate token counts (approximate: 4 chars per token)
        system_tokens = len(full_context[0]["content"]) // 4 if full_context else 0

        # Task/goal description is in second message (user role with context_info)
        task_tokens = len(full_context[1]["content"]) // 4 if len(full_context) > 1 else 0

        # Agent output: all assistant messages (content + tool_calls)
        agent_tokens = 0
        for m in full_context[2:]:
            if m.get("role") == "assistant":
                # Count content if present
                content = str(m.get("content", ""))
                agent_tokens += len(content) // 4
                # Count tool_calls if present (they take up significant tokens)
                if "tool_calls" in m:
                    # Each tool call has name + arguments
                    for tc in m["tool_calls"]:
                        agent_tokens += len(tc["function"]["name"]) // 4
                        args = tc["function"].get("arguments", "")
                        agent_tokens += len(str(args)) // 4

        # System feedback: all tool results (file reads, cmd outputs, errors, etc)
        system_feedback_tokens = sum(len(str(m.get("content", ""))) // 4
                                     for m in full_context[2:] if m.get("role") == "tool")

        context_stats = {
            "system_prompt": system_tokens,
            "task_desc": task_tokens,
            "agent_output": agent_tokens,
            "system_feedback": system_feedback_tokens,  # Renamed from system_interaction
        }

        # Display status at start of round
        # Use in-place update after first round to avoid clutter
        use_in_place = (round_no > 1)
        if use_in_place:
            # For in-place: print with end='' to avoid extra newline, then flush
            print(status.render(round_no, context_stats, in_place=True,
                              subtask_rounds=current_subtask_rounds, max_rounds=MAX_ROUNDS_PER_SUBTASK),
                  end='', flush=True)
            print()  # Single newline for spacing
        else:
            # For first round: add newline before
            print("\n" + status.render(round_no, context_stats, in_place=False,
                                      subtask_rounds=current_subtask_rounds, max_rounds=MAX_ROUNDS_PER_SUBTASK))
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

        # Set activity: Calling LLM
        status.set_activity("calling LLM: waiting for response")

        # Call LLM with inactivity timeout protection
        t0 = time.time()
        try:
            resp = chat_with_inactivity_timeout(
                model=MODEL,
                messages=context,
                options={"temperature": TEMP},
                tools=tool_specs(),
                inactivity_timeout=30,  # Fail if Ollama stops responding for 30s
            )
        except TimeoutError as e:
            # Ollama stopped responding
            llm_duration = time.time() - t0
            status.record_llm_call(llm_duration, len(context))
            log(f"ROUND {round_no}: TIMEOUT after {llm_duration:.1f}s")
            log(f"ROUND {round_no}: {str(e)}")

            print(f"\n{'='*70}")
            print(f"❌ OLLAMA TIMEOUT")
            print(f"Ollama stopped responding after {llm_duration:.1f}s")
            print(f"This usually means Ollama has hung or crashed.")
            print(f"{'='*70}\n")

            # Generate failure report and exit
            report_path = generate_failure_report(
                goal,
                _ctx,
                f"Ollama timeout: No response for 30s on round {round_no}"
            )
            print(f"📊 Failure report: {report_path}\n")
            sys.exit(1)

        except ResponseError as e:
            # Handle JSON parsing errors from malformed tool calls
            llm_duration = time.time() - t0
            status.record_llm_call(llm_duration, len(context))
            log(f"ROUND {round_no}: chat() {llm_duration:.2f}s")
            log(f"ROUND {round_no}: Ollama ResponseError (malformed tool call): {str(e)[:200]}")

            # Add error message to context and retry next round
            messages.append({
                "role": "assistant",
                "content": "I encountered an error generating the tool call. Let me try a different approach."
            })
            messages.append({
                "role": "user",
                "content": "ERROR: The previous tool call had invalid JSON formatting. Please ensure all strings in JSON use double quotes and proper escaping (e.g., \\\" for quotes, not \\')."
            })
            round_no += 1
            continue

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

                # Set activity based on tool
                if tool_name == "write_file":
                    status.set_activity(f"writing file: {c['function'].get('arguments', {}).get('path', '...')}")
                elif tool_name == "read_file":
                    status.set_activity(f"reading file: {c['function'].get('arguments', {}).get('path', '...')}")
                elif tool_name == "run_cmd":
                    cmd_args = c['function'].get('arguments', {})
                    if isinstance(cmd_args, str):
                        import json as json_lib
                        try:
                            cmd_args = json_lib.loads(cmd_args)
                        except:
                            cmd_args = {}
                    cmd = cmd_args.get('cmd', '...')
                    status.set_activity(f"running: {cmd}")
                elif tool_name == "mark_subtask_complete":
                    status.set_activity("marking subtask complete")
                else:
                    status.set_activity(f"calling tool: {tool_name}")

                # Execute tool
                try:
                    tool_result = dispatch(c)

                    # Check if force_escalate flag is set
                    if isinstance(tool_result, dict) and tool_result.get("force_escalate"):
                        log(f"Force escalation triggered for blocked action")
                        # Trigger escalation immediately
                        current_task = _ctx._get_current_task()
                        if current_task:
                            current_subtask = current_task.active_subtask()
                            if current_subtask:
                                # Force rounds to max to trigger escalation
                                current_subtask_rounds = MAX_ROUNDS_PER_SUBTASK
                                current_subtask.rounds_used = current_subtask_rounds
                                _ctx._save_state()
                                log(f"Forced subtask rounds to {MAX_ROUNDS_PER_SUBTASK} to trigger escalation")

                except Exception as e:
                    tool_result = {"error": f"dispatch-failed: {e}"}
                    log(f"Dispatch error: {e}")

                # Record action in context manager (only for non-completion tools)
                if tool_name != "mark_subtask_complete":
                    try:
                        args_dict = json.loads(c["function"]["arguments"]) if isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]
                        result_status = "success" if "result" in tool_result else "error"
                        error_msg = str(tool_result.get("error", ""))

                        # Extract result content for hash computation
                        result_content = ""
                        if "result" in tool_result:
                            # For list_dir and read_file, use the result content
                            if tool_name in ["list_dir", "read_file"]:
                                result_content = str(tool_result.get("result", ""))
                            # For other tools, use a shorter representation
                            else:
                                result_content = str(tool_result.get("result", ""))[:1000]

                        # Don't let record_action fail - just log if it does
                        try:
                            _ctx.record_action(tool_name, args_dict, result_status, error_msg, result_content)
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
                    sys.exit(0)  # Exit with success code for orchestrator

            # Increment round counters at end of round
            # Use the subtask that was active at START of round, not end
            # (mark_subtask_complete may have changed the active subtask mid-round)
            # Turn counter always increments by 1 per turn
            if round_start_subtask and round_start_sig:
                current_subtask_rounds += 1
                round_start_subtask.rounds_used = current_subtask_rounds
                task_total_rounds += 1
                _ctx._save_state()
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
            # Increment round counters at end of round
            # Use the subtask that was active at START of round
            # Turn counter always increments by 1 per turn
            if round_start_subtask and round_start_sig:
                current_subtask_rounds += 1
                round_start_subtask.rounds_used = current_subtask_rounds
                task_total_rounds += 1
            continue

        # No completion signal - this is a real final answer
        print("\n=== Agent Reply ===")
        print(msg["content"])
        sys.exit(1)  # Exit with failure - agent gave up without completing

    # Hit max rounds
    print(f"\n[stopped] Hit MAX_ROUNDS ({MAX_ROUNDS}) without completion.")
    print(f"Current task: {_ctx._get_current_task().description if _ctx._get_current_task() else 'none'}")
    sys.exit(1)  # Exit with failure code for orchestrator

if __name__ == "__main__":
    main()
