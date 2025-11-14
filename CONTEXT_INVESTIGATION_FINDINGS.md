# Context Investigation - Why L4+ Tasks Failed

## Summary

Investigated "empty workspace" failures and timeouts. Found **three distinct failure modes**, none related to tool calling bugs:

1. **Filename Spec Adherence** (rest_api_mock)
2. **Task Complexity Timeout** (command_parser, blog_system)
3. **Unknown/Context Overwrite** (config_loader)

## Detailed Analysis

### Case 1: rest_api_mock (FILENAME MISMATCH)

**Status:** "Failed - empty workspace" ❌
**Reality:** Workspace has 8 files including Flask app ✓

**Files Created:**
- `app.py` (Flask application - 1170 bytes)
- `test_app.py`, `test_api.py` (tests)
- `requirements.txt`, `README.md`
- `__pycache__/` (ran successfully)

**The Problem:**
- Task spec: "Create **api.py** with Flask app..."
- What agent created: `app.py`
- Validation looks for: `from api import app`
- Result: ModuleNotFoundError (wrong filename)

**Root Cause:** LLM didn't carefully read filename requirement. Used conventional name `app.py` instead of specified `api.py`.

**Evidence:**
```python
# Task definition
goal="Create api.py with Flask app..."
expected_files=["api.py"]

# What got created
app.py  # ← Wrong filename
```

### Case 2: Timeouts (command_parser, blog_system)

**command_parser:** Timeout at 10 minutes
**blog_system:** Timeout at 12 minutes

**Pattern:**
- Orchestrator delegated to architect
- Architect created architecture docs
- Orchestrator delegated to task_executor
- Task_executor ran many rounds but didn't complete

**Example - blog_system:**
- Task_executor ran 18 rounds
- Created architecture files (system-overview.md, module specs)
- Didn't finish implementation before timeout

**Root Cause:** Complex tasks require more coordination time than simple L3 tasks. The 10-12 minute timeouts may be too short for L4-L5 complexity with architect→executor flow.

### Case 3: config_loader (EMPTY WORKSPACE)

**Status:** Failed in 66 seconds, truly empty workspace

**What Happened:**
1. Orchestrator consulted architect (round 1)
2. Architect completed (round 2 - created architecture docs)
3. Orchestrator delegated to task_executor (round 3)
4. ???
5. Orchestrator called mark_complete claiming "Created config.py" (rounds 10-11)

**Context Evidence:**
```python
# Round 10-11: Orchestrator hallucinating
"mark_complete": {
  "summary": "Created a config.py file with Config class..."
}

# Workspace reality:
ls /tmp/orch_L4_config_loader_j8z905k0/
# Only .agent_context/ exists, no files
```

**Root Cause:** Unclear - possible causes:
1. Task_executor never started (delegation failed?)
2. Task_executor started but crashed immediately
3. Context inspection files from this run were overwritten by later tasks
4. Orchestrator hallucinated completion without waiting for task_executor

**Need to investigate:** Why orchestrator thinks task is done when workspace is empty.

## Pattern Analysis

### What's NOT the Problem

✅ **Tool calling works** - No crashes, no "NoneType" errors
✅ **File creation works** - L3 tasks create files reliably
✅ **Delegation works** - Orchestrator → Architect → TaskExecutor flow happens

### What IS the Problem

❌ **Filename spec adherence** - LLM uses conventional names vs. specified names
❌ **Complex task timeout** - L4+ tasks need >10 min with architect involvement
❌ **Completion hallucination** - Orchestrator marks complete before work is done

## Recommendations

1. **Increase timeouts for L4+**
   - L4: 15 minutes (currently 10)
   - L5: 20 minutes (currently 12)
   - L6-L7: 25-30 minutes

2. **Improve spec adherence**
   - Add filename validation in system prompt
   - Emphasize "EXACTLY as specified" in prompts
   - Consider adding filename checking before mark_complete

3. **Fix completion verification**
   - Orchestrator should verify files exist before marking complete
   - Add workspace validation tool
   - Prevent hallucinated completion

4. **Context inspection collision**
   - Context files are overwritten between tasks
   - Need per-task directories or unique filenames with task ID

## Success Rate by Complexity

- **L3 (Basic):** 83.3% ✅ - Tool calling and simple delegation works great
- **L4 (Intermediate):** 16.7% ⚠️ - Complexity/timeout/spec issues
- **L5 (Complex):** 0% ❌ - Timeout (only ran 1 task before stopping)

The drop-off at L4 is NOT a tool calling issue - it's a combination of:
- Task complexity requiring more time
- Spec adherence failures
- Orchestrator completion verification gaps
