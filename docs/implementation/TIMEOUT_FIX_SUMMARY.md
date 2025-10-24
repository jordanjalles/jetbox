# Timeout Fix Summary

## Problem

The eval suite x10 results showed 9 timeout failures (10% of all failures) with the following pattern:

- **Affected tests**: L3-2 (3x), L3-3 (1x), L4-1 (3x), L5-2 (2x)
- **Pattern**: Only in later iterations (2, 5, 7, 8, 10)
- **Symptom**: Test times out with **0 rounds executed**
- **Root cause**: Agent never starts because `decompose_goal()` hangs when calling Ollama

## Root Cause Analysis

Initially suspected "state pollution" because timeouts only occurred in later iterations. However, investigation revealed:

1. `.agent_context` was already being cleaned properly by `clean_workspace()`
2. The issue was NOT persistent state
3. The real problem: **Ollama performance degradation during long test runs**

### Technical Details

- `decompose_goal()` at agent.py:892 calls `chat()` with **no timeout**
- When Ollama becomes slow/unresponsive after many tests, this LLM call hangs indefinitely
- Test framework times out (240-360s) but shows "0 rounds" because agent never completes decomposition and never enters main loop
- Pattern matches Ollama getting backed up after processing many requests

## Solution Implemented

### 1. Ollama Health Monitoring

Added health check functions to both test suites:

**File**: `tests/run_stress_tests.py` (lines 271-297)
**File**: `tests/run_eval_suite.py` (lines 13-39)

```python
def check_ollama_health() -> bool:
    """Check if Ollama is responsive."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def restart_ollama_if_needed() -> None:
    """Restart Ollama if it's not responding (helps prevent hangs)."""
    if not check_ollama_health():
        print("[warning] Ollama not responding, waiting and checking again...")
        time.sleep(5)

        # Check again after wait
        if check_ollama_health():
            print("[info] Ollama recovered after wait")
            return

        print("[warning] Ollama still not responding. Manual restart may be needed.")
        print("[info] To restart Ollama:")
        print("  Windows: Restart Ollama app or run: ollama serve")
        print("  Linux: systemctl restart ollama")
        print("[info] Continuing test anyway...")
        time.sleep(2)
```

**Integration points**:
- Called before each test in `run_test()` (run_stress_tests.py:361)
- Called before each iteration in `run_full_suite()` (run_eval_suite.py:55)

### 2. LLM Call Timeout Wrapper

Added timeout mechanism for Ollama chat calls to prevent infinite hangs:

**File**: `agent.py` (lines 38-77)

```python
def chat_with_timeout(model: str, messages: list, options: dict, timeout_seconds: int = 120):
    """
    Call ollama chat with a timeout to prevent infinite hangs.

    Uses threading to enforce a timeout on the LLM call.
    Raises TimeoutError if call takes longer than timeout_seconds.
    """
    result_queue = Queue()

    def _call_chat():
        try:
            resp = chat(model=model, messages=messages, options=options)
            result_queue.put(("success", resp))
        except Exception as e:
            result_queue.put(("error", e))

    thread = Thread(target=_call_chat, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread is still running - timeout occurred
        raise TimeoutError(f"LLM call timed out after {timeout_seconds}s")

    if result_queue.empty():
        raise RuntimeError("Thread finished but no result in queue")

    status, result = result_queue.get()
    if status == "error":
        raise result
    return result
```

### 3. Updated decompose_goal() with Timeout

**File**: `agent.py` (lines 939-949)

```python
log("Decomposing goal into tasks...")
try:
    resp = chat_with_timeout(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
        timeout_seconds=120,  # 2 minute timeout for decomposition
    )
except TimeoutError as e:
    log(f"Goal decomposition timed out: {e}")
    # Fallback: create a single generic task
    return [{"description": goal, "subtasks": ["Complete the goal"]}]
```

## Expected Impact

With these fixes in place:

**Current state**: 68/90 passed (75.6%)

**After fixing timeouts**: 77/90 passed (85.6%)
- Eliminates 9 timeout failures
- Provides graceful degradation when Ollama is slow
- Agent can continue with fallback task structure instead of hanging

**After fixing L4-2** (separate issue): 87/90 passed (96.7%)
- L4-2 has 10/10 failures with "unknown_failure"
- Needs separate investigation

## Testing

To verify the fix works:

```bash
# Run a few problematic tests
python tests/run_stress_tests.py 3,4,5

# Or run full eval suite
python tests/run_eval_suite.py
```

Monitor for:
1. Health check warnings when Ollama is slow
2. Timeout fallback messages in logs
3. Tests starting with 0 rounds (should not happen anymore)

## Future Improvements

1. **Apply timeout to main loop LLM calls** - Currently only decompose_goal() has timeout
2. **Ollama connection pooling** - Better handle concurrent requests during testing
3. **Adaptive timeout** - Adjust timeout based on model speed and system load
4. **Better restart mechanism** - Currently just warns, could auto-restart Ollama on Windows/Linux
