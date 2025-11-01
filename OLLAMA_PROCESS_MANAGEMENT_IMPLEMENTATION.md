# Ollama Process Management - Implementation Summary

**Date**: 2025-11-01
**Status**: ✅ Implemented and Testing
**Components**: OllamaManager, ProcessTracker, safe_test_runner.py

---

## Executive Summary

Implemented Phase 1 of Ollama process management to prevent stuck states when running background tests. The system now:
- Detects when Ollama is busy processing requests
- Clears stuck contexts before starting tests
- Tracks test subprocess PIDs for safe termination
- Prevents accidentally killing Claude process

---

## Root Cause Analysis

### The Problem Chain

1. **Background tests start** → Long-running evaluation spawns Ollama requests
2. **User requests stop** → Claude tries to kill test processes
3. **Wrong process killed** → `pkill -f python` kills Claude's own process
4. **Claude restarts** → User restarts Claude session
5. **Ollama not refreshed** → Ollama still has stuck contexts from previous tests
6. **New tests hang** → New tests timeout because Ollama is processing old requests

### Why Generic `pkill` Fails

```bash
pkill -f "run_three_level_eval.py"  # Too specific, might not match
pkill -f "python.*eval"              # Too broad, kills Claude process
```

**Problems**:
- No way to identify which Python process is the test vs Claude
- No tracking of spawned subprocess PIDs
- No Ollama context cleanup on abnormal termination
- No detection of Ollama busy state before starting tests

---

## Implemented Components

### 1. OllamaManager (`ollama_manager.py`)

**Purpose**: Detect and manage Ollama state to prevent stuck contexts.

**Key Methods**:

```python
class OllamaManager:
    @staticmethod
    def is_ollama_busy() -> bool:
        """Check if Ollama is processing requests (>1 model loaded)."""

    @staticmethod
    def wait_for_ollama(timeout: int = 60) -> bool:
        """Wait for Ollama to become idle."""

    @staticmethod
    def clear_all_contexts(model: str = None):
        """Clear all Ollama contexts for specified model(s)."""

    @staticmethod
    def get_loaded_models() -> list:
        """Get list of currently loaded models."""

    @staticmethod
    def get_ollama_status() -> dict:
        """Get detailed status (models, busy state, etc.)."""
```

**Usage**:
```python
from ollama_manager import OllamaManager

# Check status
OllamaManager.print_status()

# Before starting tests
if OllamaManager.is_ollama_busy():
    OllamaManager.wait_for_ollama(timeout=60)

# Clear stuck contexts
OllamaManager.clear_all_contexts("gpt-oss:20b")
```

**Output Example**:
```
======================================================================
OLLAMA STATUS
======================================================================
Status: IDLE
Loaded models: 0
  (none)
======================================================================
```

---

### 2. ProcessTracker (`process_tracker.py`)

**Purpose**: Track spawned test processes for safe termination without killing Claude.

**Key Methods**:

```python
class ProcessTracker:
    TRACKER_FILE = Path(".agent_context/running_processes.json")

    @classmethod
    def register_process(cls, pid: int, process_type: str, description: str):
        """Register a process as running."""

    @classmethod
    def unregister_process(cls, pid: int):
        """Unregister a process (it completed)."""

    @classmethod
    def kill_all_tracked_processes(cls, exclude_ppid: int = None) -> list:
        """Kill all tracked processes, excluding children of specified parent."""

    @classmethod
    def list_running_processes(cls) -> list:
        """List all tracked running processes."""

    @classmethod
    def cleanup_dead_processes(cls) -> int:
        """Remove dead processes from tracker."""
```

**Registry Format** (`.agent_context/running_processes.json`):
```json
{
  "1647": {
    "pid": 1647,
    "type": "test_suite",
    "description": "Test: run_three_level_eval",
    "started": "2025-11-01T22:18:17.123456",
    "ppid": 1234
  }
}
```

**Usage**:
```python
from process_tracker import ProcessTracker
import os

# Register at test start
pid = os.getpid()
ProcessTracker.register_process(pid, "evaluation", "3-level test suite")

try:
    run_tests()
finally:
    ProcessTracker.unregister_process(pid)
```

**Safe Kill** (from another process/Claude):
```python
# Kill all tracked test processes, but exclude Claude's children
claude_pid = os.getppid()
killed = ProcessTracker.kill_all_tracked_processes(exclude_ppid=claude_pid)
print(f"Killed {len(killed)} test processes safely")
```

---

### 3. Safe Test Runner (`safe_test_runner.py`)

**Purpose**: Launch tests with proper Ollama checks and process tracking.

**4-Step Process**:

1. **Check Ollama Status** - Detect if Ollama is busy
2. **Clear Old Contexts** - Prevent stuck state from previous runs
3. **Launch Test Subprocess** - Track PID, stream output
4. **Monitor & Cleanup** - Handle interruption, always unregister

**Usage**:
```bash
# Instead of:
python run_three_level_eval.py

# Use:
python safe_test_runner.py run_three_level_eval.py
```

**Output Example**:
```
======================================================================
SAFE TEST RUNNER: Test: run_three_level_eval
======================================================================

[1/4] Checking Ollama status...
✓ Ollama is idle

[2/4] Clearing Ollama contexts...
[ollama_manager] ✓ Cleared context for gpt-oss:20b
✓ Cleared contexts for gpt-oss:20b

[3/4] Launching test: run_three_level_eval.py
[process_tracker] Registered PID 1647: Test: run_three_level_eval
✓ Test running as PID 1647
  (To stop: python stop_tests.py or Ctrl+C)

[4/4] Test output:
----------------------------------------------------------------------
[... test output streams in real-time ...]
----------------------------------------------------------------------

✓ Test completed successfully
✓ Cleaned up PID 1647
```

**Features**:
- ✅ Pre-checks Ollama before starting
- ✅ Clears contexts automatically
- ✅ Tracks subprocess PID
- ✅ Streams output in real-time
- ✅ Handles Ctrl+C gracefully
- ✅ Always cleans up on exit (normal or error)

---

## How It Prevents the Root Cause

### Before (Problem State)

1. User runs: `python run_three_level_eval.py` in background
2. Test hangs, user tries: `pkill -f python`
3. **Kills Claude process** (same `python` command)
4. Claude restarts, Ollama still has old contexts
5. New tests hang on stuck Ollama state

### After (With Safe Runner)

1. User runs: `python safe_test_runner.py run_three_level_eval.py`
2. Safe runner:
   - Checks Ollama status first
   - Clears any stuck contexts
   - Registers test PID (1647)
   - Starts test as subprocess
3. If user stops test:
   - `ProcessTracker.kill_all_tracked_processes()` kills PID 1647
   - Excludes Claude's PPID → **Claude process safe**
4. On Claude restart:
   - OllamaManager can clear contexts again
   - No stuck state

---

## Testing Results

### Component Tests

**OllamaManager**:
```bash
$ python ollama_manager.py
======================================================================
OLLAMA STATUS
======================================================================
Status: IDLE
Loaded models: 0
  (none)
======================================================================
```
✅ Successfully detects Ollama state

**ProcessTracker**:
```bash
$ python process_tracker.py
======================================================================
TRACKED PROCESSES
======================================================================
[process_tracker] No tracked processes running

Cleaning up dead processes...
✓ All tracked processes are alive
```
✅ Successfully tracks processes

**Safe Test Runner**:
```bash
$ python safe_test_runner.py run_three_level_eval.py
======================================================================
SAFE TEST RUNNER: Test: run_three_level_eval
======================================================================

[1/4] Checking Ollama status...
✓ Ollama is idle

[2/4] Clearing Ollama contexts...
[ollama_manager] ✓ Cleared context for gpt-oss:20b
✓ Cleared contexts for gpt-oss:20b

[3/4] Launching test: run_three_level_eval.py
[process_tracker] Registered PID 1647: Test: run_three_level_eval
✓ Test running as PID 1647
```
✅ Successfully launches tests with proper checks

### Integration Test

**Currently Running**: Three-level evaluation suite via safe_test_runner
- Status: ✅ Running (PID 1647 tracked)
- Ollama: Contexts cleared before start
- Output: Streaming in real-time
- Level 1 tests: In progress

---

## Benefits

### Safety
- ✅ **Never kills Claude process** - Tracks PIDs, excludes parent process
- ✅ **Clean shutdown** - Always unregisters processes
- ✅ **Graceful interruption** - Handles Ctrl+C properly

### Reliability
- ✅ **Prevents stuck Ollama** - Clears contexts before starting
- ✅ **Detects busy state** - Waits for Ollama to be idle
- ✅ **Survives restarts** - Tracker file persists

### Observability
- ✅ **List running tests** - `python process_tracker.py`
- ✅ **Check Ollama status** - `python ollama_manager.py`
- ✅ **Real-time output** - Safe runner streams test output

---

## Future Enhancements (Not Implemented)

### Solution 4: stop_tests.py (Proposed)

User-facing command to safely stop all tests:

```python
# stop_tests.py
from process_tracker import ProcessTracker
from ollama_manager import OllamaManager

def stop_all_tests():
    """Safely stop all running test processes without killing Claude."""
    ProcessTracker.list_running_processes()
    claude_pid = os.getppid()
    killed = ProcessTracker.kill_all_tracked_processes(exclude_ppid=claude_pid)
    OllamaManager.clear_all_contexts()
    print(f"✓ Stopped {len(killed)} test process(es)")
```

### Solution 5: startup_check() (Proposed)

Auto-cleanup on Claude restart:

```python
# In agent initialization
def startup_check():
    """Run on Claude restart to clean up stuck state."""
    print("Claude restarted - checking for stuck Ollama state...")
    ProcessTracker.cleanup_dead_processes()
    OllamaManager.clear_all_contexts()
    print("✓ Cleanup complete")
```

---

## Files Created

1. **ollama_manager.py** (185 lines)
   - OllamaManager class
   - Methods: is_ollama_busy, wait_for_ollama, clear_all_contexts, get_loaded_models
   - Standalone script: `python ollama_manager.py` prints status

2. **process_tracker.py** (178 lines)
   - ProcessTracker class
   - Methods: register_process, unregister_process, kill_all_tracked_processes, list_running_processes
   - Standalone script: `python process_tracker.py` lists tracked processes

3. **safe_test_runner.py** (131 lines)
   - safe_run_test_suite() function
   - 4-step launch process
   - Standalone script: `python safe_test_runner.py <script.py>`

---

## Usage Guide

### For Running Tests

**Old way** (unsafe):
```bash
python run_three_level_eval.py
```

**New way** (safe):
```bash
python safe_test_runner.py run_three_level_eval.py
```

### For Stopping Tests

**Old way** (unsafe):
```bash
pkill -f python  # Kills Claude!
```

**New way** (safe - coming in Solution 4):
```bash
python stop_tests.py  # Only kills tracked test processes
```

**Current workaround**:
```python
from process_tracker import ProcessTracker
import os

claude_pid = os.getppid()
killed = ProcessTracker.kill_all_tracked_processes(exclude_ppid=claude_pid)
```

### For Checking Status

**Check Ollama**:
```bash
python ollama_manager.py
```

**Check Running Tests**:
```bash
python process_tracker.py
```

---

## Summary

Phase 1 of Ollama process management is **fully implemented and testing**. The system now prevents the root cause of stuck Ollama states by:

1. **Pre-checking** Ollama status before tests
2. **Clearing** stuck contexts automatically
3. **Tracking** test process PIDs safely
4. **Protecting** Claude process from accidental termination

**Current Status**: Three-level evaluation running via safe_test_runner (PID 1647)

**Next Steps**:
- Monitor evaluation results
- Implement stop_tests.py (Solution 4)
- Add startup_check() to agent initialization (Solution 5)

---

*Implementation Date: 2025-11-01*
*Status: ✅ Testing in progress*
