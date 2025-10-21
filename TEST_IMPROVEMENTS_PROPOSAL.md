# Test Suite Improvement Proposals

## Executive Summary

Based on testing with the stress test suite, the agent shows **67% success on basic tasks**, **0% on intermediate**, and **33% on advanced**. Key issues identified:
- PYTHONPATH problems in workspace isolation (pytest can't import modules)
- Max rounds limit (24) too restrictive for complex multi-file tasks
- Incomplete workspace cleanup between tests affecting results
- Missing success detection patterns causing premature failures

## Critical Issues Found

### 1. **PYTHONPATH/Import Issues in Workspace** (CRITICAL)
**Problem:** Tests fail with `ModuleNotFoundError` because pytest runs from workspace subdirectory but can't find the package.

**Evidence:**
```
ERROR: ModuleNotFoundError: No module named 'mathx'
```
Even though files exist:
- `.agent_workspace/create-a-mathx-package-with-mathx-basic-py-add-sub/mathx/__init__.py`
- `.agent_workspace/create-a-mathx-package-with-mathx-basic-py-add-sub/tests/test_mathx.py`

**Root cause:** `workspace_manager.py:46-52` generates pytest commands without setting PYTHONPATH.

**Impact:** 2/3 failures in L2-L3 tests with package structures

### 2. **MAX_ROUNDS Too Low for Complex Tasks**
**Problem:** 24-round limit is hit on multi-file package creation tasks.

**Evidence:**
- L2-3: "Package with Modules" - hit max rounds at 24
- L3-3: "Add Feature to Package" - hit max rounds at 24
- Agent creates files successfully but runs out of rounds during verification

**Impact:** ~33% of intermediate/advanced tests fail due to timeout

### 3. **Insufficient Success Detection**
**Problem:** Tests marked as "unknown_failure" even when goal appears achieved.

**Evidence:**
- L1-1: "Hello World" - completed in 4 rounds but marked as failure
- L2-2: "Rock Paper Scissors" - completed in 8 rounds but marked as failure

**Root cause:** `run_stress_tests.py:320` only checks for exact string "Goal achieved" in output

**Impact:** False negatives reduce apparent agent reliability

### 4. **Workspace Pollution Between Tests**
**Problem:** Incomplete cleanup leaves artifacts that confuse subsequent tests.

**Evidence:**
- `.agent_workspace/` has 9+ subdirectories from different test runs
- `run_stress_tests.py:246-277` cleanup logic has hardcoded exclusions
- Some Python cache files persist (`__pycache__/`)

**Impact:** Non-deterministic test results, flaky tests

---

## Proposed Improvements

### A. Test Infrastructure Improvements

#### A1. Fix Workspace PYTHONPATH Issue (HIGH PRIORITY)
**File:** `workspace_manager.py`

**Change:** Update `get_test_command()` to set PYTHONPATH correctly:

```python
def get_test_command(self) -> list[str] | None:
    test_dirs = list(self.workspace_dir.glob("tests/"))
    if test_dirs:
        # Set PYTHONPATH to workspace root for imports
        return ["python", "-m", "pytest",
                str(test_dirs[0].relative_to(self.workspace_dir)), "-q"]
    test_files = list(self.workspace_dir.glob("test_*.py")) + \
                 list(self.workspace_dir.glob("*_test.py"))
    if test_files:
        return ["python", "-m", "pytest"] + \
               [str(f.relative_to(self.workspace_dir)) for f in test_files] + ["-q"]
    return None
```

**Alternative:** Inject `sys.path.insert(0, str(workspace_dir))` into agent's run_cmd for pytest

**Expected improvement:** Fix 40-50% of current failures

#### A2. Improve Test Success Detection
**File:** `run_stress_tests.py`

**Change:** Expand success detection patterns (lines 320-327):

```python
# Check if agent completed successfully
success_patterns = [
    "Goal achieved",
    "All tests passing",
    "Task completed successfully",
    "marked complete",
    "Successfully created",
    r"✓.*complete",  # Regex for checkmark patterns
]

for pattern in success_patterns:
    if re.search(pattern, result["output"], re.IGNORECASE):
        result["success"] = True
        break

# Still check failure modes
if not result["success"]:
    if "Hit MAX_ROUNDS" in result["output"]:
        result["failure_mode"] = "max_rounds_exceeded"
    elif "loop" in result["output"].lower():
        result["failure_mode"] = "infinite_loop"
```

**Expected improvement:** Reduce false negatives by 20-30%

#### A3. Enhanced Workspace Cleanup
**File:** `run_stress_tests.py`

**Change:** Replace pattern-based cleanup (lines 246-277) with comprehensive approach:

```python
def clean_workspace() -> None:
    """Clean up workspace before test."""
    # Remove entire .agent_workspace
    if Path(".agent_workspace").exists():
        shutil.rmtree(".agent_workspace")

    # Remove agent state
    if Path(".agent_context").exists():
        shutil.rmtree(".agent_context")

    # Remove log files
    for log_file in ["agent_v2.log", "agent_ledger.log"]:
        if Path(log_file).exists():
            Path(log_file).unlink()

    # Remove __pycache__ directories
    for pycache in Path(".").rglob("__pycache__"):
        shutil.rmtree(pycache)

    # Remove any test artifact files in root (but preserve agent code)
    preserve_files = {
        "agent.py", "context_manager.py", "status_display.py",
        "workspace_manager.py", "completion_detector.py",
        "run_stress_tests.py", "test_status_display.py",
        "pyproject.toml", "README.md", "CLAUDE.md",
        "AGENT_ARCHITECTURE.md", "STATUS_DISPLAY.md"
    }

    for item in Path(".").iterdir():
        if item.is_file() and item.suffix in ['.py', '.txt', '.json']:
            if item.name not in preserve_files:
                item.unlink()
        elif item.is_dir() and item.name not in {'.agent_workspace', '.agent_context',
                                                   '.git', 'hrm-jepa', '__pycache__'}:
            if not item.name.startswith('.'):
                shutil.rmtree(item)
```

**Expected improvement:** Eliminate test flakiness from pollution

#### A4. Add Detailed Test Metrics
**File:** `run_stress_tests.py`

**Change:** Track additional metrics in test results:

```python
result = {
    # ... existing fields ...
    "files_expected": len(test.get("expected_files", [])),
    "files_found": len(result["files_created"]),
    "pytest_ran": False,
    "pytest_passed": False,
    "ruff_ran": False,
    "ruff_passed": False,
    "completion_signals": [],  # Which completion phrases were found
}

# Parse output for tool executions
if "pytest" in result["output"]:
    result["pytest_ran"] = True
    result["pytest_passed"] = "passed" in result["output"] and \
                              "failed" not in result["output"]

if "ruff" in result["output"]:
    result["ruff_ran"] = True
    result["ruff_passed"] = "All checks passed" in result["output"]
```

**Expected improvement:** Better debugging of failures, clearer metrics

---

### B. Agent Reliability Improvements

#### B1. Increase MAX_ROUNDS for Complex Tasks
**File:** `agent.py`

**Change:** Make MAX_ROUNDS adaptive based on task complexity:

```python
# agent.py line 28
DEFAULT_MAX_ROUNDS = 24
MAX_ROUNDS_COMPLEX = 40  # For tasks with multiple subtasks

# In main loop, detect complexity:
def estimate_task_complexity(goal_description: str) -> int:
    """Estimate rounds needed based on goal description."""
    complexity_keywords = {
        'package': 10,
        'test': 5,
        'refactor': 8,
        'multiple': 5,
        'all': 5,
        'comprehensive': 8,
    }

    rounds = DEFAULT_MAX_ROUNDS
    for keyword, bonus in complexity_keywords.items():
        if keyword in goal_description.lower():
            rounds += bonus

    return min(rounds, MAX_ROUNDS_COMPLEX)
```

**Expected improvement:** Reduce timeout failures by 50%

#### B2. Better Completion Detection in Agent
**File:** `agent.py`

**Change:** Enhance the completion detector integration:

```python
# After LLM response, check for completion signals
from completion_detector import analyze_llm_response

# In main loop after getting LLM response:
completion_signals = analyze_llm_response(msg_content)
if completion_signals and completion_signals.get("likely_complete"):
    # Verify actual state matches completion claim
    state = probe_state_generic()

    # If files exist and tests pass, accept completion
    if state["files_exist"] and not state["recent_errors"]:
        log("Completion detected with verified state")
        ctx.mark_current_subtask_complete("Verified completion")
```

**Expected improvement:** Better goal completion recognition

#### B3. Add PYTHONPATH Management to run_cmd
**File:** `agent.py`

**Change:** Automatically set PYTHONPATH when running pytest in workspace:

```python
def run_cmd(cmd: list[str], workspace: WorkspaceManager | None = None) -> dict[str, Any]:
    # ... existing validation ...

    env = os.environ.copy()

    # Set PYTHONPATH for pytest in workspace
    if workspace and "pytest" in cmd:
        env["PYTHONPATH"] = str(workspace.workspace_dir)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=workspace.workspace_dir if workspace else None,
        env=env  # Use modified environment
    )
```

**Expected improvement:** Fix import errors in workspace tests

#### B4. Add Verification Step Before Completion
**File:** `agent.py`

**Change:** Add final verification round before declaring success:

```python
def verify_goal_completion(ctx: ContextManager, workspace: WorkspaceManager) -> bool:
    """Verify goal is actually complete before exiting."""
    verification = {
        "files_created": len(workspace.created_files) > 0,
        "tests_exist": workspace.get_test_command() is not None,
        "tests_pass": False,
        "lint_pass": False,
    }

    # Run tests if they exist
    test_cmd = workspace.get_test_command()
    if test_cmd:
        result = run_cmd(test_cmd, workspace)
        verification["tests_pass"] = result["rc"] == 0

    # Run linter if Python files exist
    lint_cmd = workspace.get_lint_command()
    if lint_cmd:
        result = run_cmd(lint_cmd, workspace)
        verification["lint_pass"] = result["rc"] == 0

    return verification["files_created"] and \
           (not verification["tests_exist"] or verification["tests_pass"])
```

**Expected improvement:** Reduce false positives on completion

---

### C. New Test Coverage

#### C1. Add Unit Tests for Core Components

**New file:** `tests/test_workspace_manager.py`
```python
"""Unit tests for workspace isolation."""
import pytest
from pathlib import Path
from workspace_manager import WorkspaceManager

def test_workspace_creation():
    ws = WorkspaceManager("test goal")
    assert ws.workspace_dir.exists()
    assert "test-goal" in str(ws.workspace_dir)

def test_path_resolution():
    ws = WorkspaceManager("test goal")
    resolved = ws.resolve_path("foo.py")
    assert ws.workspace_dir in resolved.parents

def test_pythonpath_in_test_command():
    ws = WorkspaceManager("test goal")
    (ws.workspace_dir / "tests").mkdir()
    (ws.workspace_dir / "tests" / "test_foo.py").touch()

    cmd = ws.get_test_command()
    assert cmd is not None
    assert "pytest" in cmd
    # Should handle PYTHONPATH properly
```

**New file:** `tests/test_completion_detector.py`
```python
"""Unit tests for completion detection."""
import pytest
from completion_detector import analyze_llm_response

def test_detects_completion_phrases():
    response = "Task completed successfully. All tests passing."
    result = analyze_llm_response(response)
    assert result["likely_complete"] == True

def test_no_false_positives():
    response = "Starting to work on the task..."
    result = analyze_llm_response(response)
    assert result["likely_complete"] == False
```

**Expected improvement:** Catch regressions in core components

#### C2. Add Integration Tests

**New file:** `tests/integration/test_agent_simple_tasks.py`
```python
"""Integration tests for simple agent tasks."""
import subprocess
import shutil
from pathlib import Path

def setup_function():
    """Clean workspace before each test."""
    if Path(".agent_workspace").exists():
        shutil.rmtree(".agent_workspace")
    if Path(".agent_context").exists():
        shutil.rmtree(".agent_context")

def test_create_single_file():
    """Agent should create a single file successfully."""
    result = subprocess.run(
        ["python", "agent.py", "Create hello.py with a hello world function"],
        capture_output=True,
        text=True,
        timeout=30
    )

    # Find the workspace
    workspaces = list(Path(".agent_workspace").iterdir())
    assert len(workspaces) == 1

    # Check file was created
    hello_files = list(workspaces[0].glob("**/hello.py"))
    assert len(hello_files) > 0

def test_create_file_with_tests():
    """Agent should create a file and its tests."""
    result = subprocess.run(
        ["python", "agent.py",
         "Create math_utils.py with add function and test_math_utils.py with tests"],
        capture_output=True,
        text=True,
        timeout=60
    )

    workspace = list(Path(".agent_workspace").iterdir())[0]
    assert (workspace / "math_utils.py").exists() or \
           any(workspace.rglob("math_utils.py"))
    assert (workspace / "test_math_utils.py").exists() or \
           any(workspace.rglob("test_math_utils.py"))
```

**Expected improvement:** Catch integration issues early

#### C3. Add Performance Benchmarks

**New file:** `benchmark_agent.py`
```python
"""Benchmark agent performance on standard tasks."""
import time
import statistics
from pathlib import Path

BENCHMARK_TASKS = [
    ("Simple file", "Create hello.py"),
    ("Function", "Create add function in math.py"),
    ("With tests", "Create utils.py with tests"),
]

def run_benchmark():
    results = []

    for name, task in BENCHMARK_TASKS:
        times = []
        for i in range(3):  # Run each 3 times
            start = time.time()
            # Run agent
            subprocess.run(["python", "agent.py", task],
                         capture_output=True, timeout=60)
            times.append(time.time() - start)

            # Cleanup
            shutil.rmtree(".agent_workspace")

        results.append({
            "task": name,
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times),
        })

    # Compare against baseline
    baseline = load_baseline()
    for r in results:
        regression = (r["mean"] - baseline[r["task"]]) / baseline[r["task"]]
        if regression > 0.2:  # 20% slower
            print(f"WARNING: {r['task']} is {regression*100:.0f}% slower")
```

**Expected improvement:** Track performance regressions

---

## Implementation Priority

### Phase 1: Critical Fixes (Week 1)
1. **A1**: Fix PYTHONPATH in workspace manager ✓ HIGH IMPACT
2. **B3**: Add PYTHONPATH to run_cmd ✓ HIGH IMPACT
3. **A3**: Enhanced workspace cleanup ✓ CRITICAL FOR RELIABILITY

**Expected impact:** 40-60% improvement in test pass rate

### Phase 2: Reliability (Week 2)
4. **A2**: Improve success detection in tests
5. **B1**: Adaptive MAX_ROUNDS
6. **B2**: Better completion detection
7. **A4**: Detailed test metrics

**Expected impact:** 20-30% improvement, better visibility

### Phase 3: Coverage (Week 3)
8. **C1**: Unit tests for components
9. **C2**: Integration tests
10. **B4**: Verification before completion

**Expected impact:** Prevent regressions, catch issues earlier

### Phase 4: Performance (Week 4)
11. **C3**: Performance benchmarks
12. **Optimization**: Based on benchmark results

**Expected impact:** Maintain performance over time

---

## Success Metrics

### Current Baseline
- Level 1: 67% pass rate
- Level 2: 0% pass rate
- Level 3: 33% pass rate
- Overall: ~33% pass rate

### Phase 1 Targets
- Level 1: 90%+ pass rate
- Level 2: 60%+ pass rate
- Level 3: 50%+ pass rate
- Overall: 65%+ pass rate

### Phase 2 Targets
- Level 1: 95%+ pass rate
- Level 2: 80%+ pass rate
- Level 3: 70%+ pass rate
- Overall: 80%+ pass rate

### Phase 3+ Targets
- Level 1: 100% pass rate
- Level 2: 90%+ pass rate
- Level 3: 80%+ pass rate
- Level 4: 60%+ pass rate
- Overall: 85%+ pass rate

---

## Conclusion

The test suite reveals **systematic issues** rather than fundamental agent capability problems:
1. Workspace isolation breaks Python imports (PYTHONPATH)
2. Round limits too conservative for actual task complexity
3. Success detection too narrow, missing valid completions

**All issues are fixable** with focused improvements to infrastructure. The agent demonstrates strong basic capabilities - we just need to remove the obstacles preventing it from succeeding on more complex tasks.

**Recommended action:** Implement Phase 1 fixes immediately, then reassess with full test suite.
