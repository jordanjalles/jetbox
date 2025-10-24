# Timeout Fix Proposal

## Problem Analysis

The L3-L7 stress tests show **59% of failures are timeouts** where the agent process hangs with **0 rounds executed**. This indicates the agent never enters the main loop.

### Root Cause

Based on code analysis, the timeout happens during **agent initialization**, specifically:

1. **Line 1162-1165**: `wait_for_ollama(max_wait=30)` - Health check passes ✅
2. **Line 1169-1179**: Workspace initialization - Fast, unlikely to hang ✅
3. **Line 1182-1183**: `ContextManager._load_or_init(goal)` - May block if state loading fails ⚠️
4. **Line 1189-1211**: **Goal decomposition via LLM call** - **LIKELY CULPRIT** ⚠️⚠️⚠️

**Key Evidence:**
- L4-3 **passed iteration 1** but **timed out iteration 2**
- Same goal decomposition happens every time (no caching)
- `decompose_goal()` makes an LLM call that can hang if Ollama degrades

### Why Goal Decomposition Can Hang

```python
# Line 1191 in agent.py
if not _ctx.state.goal or not _ctx.state.goal.tasks:
    tasks_data = decompose_goal(goal)  # ← LLM call with NO timeout protection
```

The `decompose_goal()` function likely uses `chat()` directly without the inactivity timeout wrapper. If Ollama becomes unresponsive during this call, the agent hangs indefinitely.

---

## Proposed Solution: Multi-Layer Timeout Protection

### Layer 1: Add Heartbeat Logging ⭐ HIGH PRIORITY

**Goal:** Let test framework detect progress even if agent hangs later

**Implementation:**
```python
def main() -> None:
    # ... existing code ...
    
    print("[HEARTBEAT] Agent starting")
    sys.stdout.flush()  # Force immediate output
    
    # Check Ollama health
    print("[HEARTBEAT] Checking Ollama health")
    sys.stdout.flush()
    if not wait_for_ollama(max_wait=30, check_interval=5):
        # ... error handling ...
    
    print("[HEARTBEAT] Initializing workspace")
    sys.stdout.flush()
    _workspace = WorkspaceManager(goal, ...)
    
    print("[HEARTBEAT] Loading context")
    sys.stdout.flush()
    _ctx = ContextManager()
    _ctx.load_or_init(goal)
    
    print("[HEARTBEAT] Decomposing goal")
    sys.stdout.flush()
    if not _ctx.state.goal or not _ctx.state.goal.tasks:
        tasks_data = decompose_goal(goal)
        # ...
    
    print("[HEARTBEAT] Entering main loop")
    sys.stdout.flush()
    
    for round_no in range(1, MAX_ROUNDS + 1):
        print(f"[HEARTBEAT] ROUND {round_no} started")
        sys.stdout.flush()
        # ... existing loop code ...
```

**Benefit:** Test framework can detect which stage failed:
- "ROUND X started" → Agent is running (current detection works)
- "Decomposing goal" but no "Entering main loop" → Goal decomposition hung
- "Checking Ollama" but no "Initializing workspace" → Ollama check hung

---

### Layer 2: Wrap Goal Decomposition with Timeout ⭐ CRITICAL FIX

**Goal:** Prevent indefinite hang during goal decomposition

**Current Code (agent.py ~line 150-180):**
```python
def decompose_goal(goal: str) -> list[dict[str, Any]]:
    """Decompose goal into tasks and subtasks."""
    # ... builds prompt ...
    
    resp = chat(  # ← NO TIMEOUT PROTECTION
        model=MODEL,
        messages=[{"role": "user", "content": decomp_prompt}],
        options={"temperature": TEMP},
    )
    # ... parse response ...
```

**Fixed Code:**
```python
def decompose_goal(goal: str) -> list[dict[str, Any]]:
    """Decompose goal into tasks and subtasks."""
    # ... builds prompt ...
    
    try:
        # Use inactivity timeout wrapper
        resp = chat_with_inactivity_timeout(
            model=MODEL,
            messages=[{"role": "user", "content": decomp_prompt}],
            options={"temperature": TEMP},
            inactivity_timeout=45  # Longer than default for complex goals
        )
    except TimeoutError as e:
        log(f"[error] Goal decomposition timed out: {e}")
        # Fallback: Create single simple task
        log("[fallback] Creating simple single-task fallback")
        return [{
            "description": goal,
            "subtasks": [
                "Understand the requirements",
                "Implement the solution", 
                "Verify it works"
            ]
        }]
    
    # ... parse response ...
```

**Benefit:**
- Goal decomposition can never hang indefinitely
- Falls back to simple task structure if Ollama is slow
- Agent can still attempt the task even if decomposition fails

---

### Layer 3: Check Ollama Before EACH LLM Call ⚠️ MODERATE PRIORITY

**Goal:** Detect Ollama degradation before making expensive LLM calls

**Implementation:**
```python
def chat_with_health_check(model: str, messages: list, options: dict, inactivity_timeout: int = 30):
    """
    Chat with pre-flight health check to avoid hanging on dead Ollama.
    """
    # Quick health check before expensive call
    if not check_ollama_health():
        log("[warning] Ollama health check failed before LLM call")
        # Wait briefly for recovery
        time.sleep(2)
        if not check_ollama_health():
            raise RuntimeError("Ollama is not responding - cannot proceed with LLM call")
    
    # Proceed with timeout-protected call
    return chat_with_inactivity_timeout(model, messages, options, inactivity_timeout)
```

**Usage:**
Replace all `chat_with_inactivity_timeout()` calls with `chat_with_health_check()`.

**Benefit:**
- Fails fast if Ollama is dead
- Provides clear error message instead of silent timeout
- Reduces wasted time waiting for inevitable timeout

---

### Layer 4: Test Framework Improvements 🔧 INFRASTRUCTURE

**Goal:** Better timeout detection in test framework

**Current Code (run_stress_tests.py:469-474):**
```python
proc = subprocess.run(
    ["python", "agent.py", test["task"]],
    capture_output=True,  # ← Can cause deadlock on large output
    text=True,
    timeout=test["timeout"],
)
```

**Issue:** `capture_output=True` buffers ALL output. If agent produces tons of output (e.g., during loop), the buffer can fill and cause a deadlock.

**Fixed Code:**
```python
# Use Popen for better control
proc = subprocess.Popen(
    ["python", "agent.py", test["task"]],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,  # Line buffered
)

# Monitor output with timeout
output_lines = []
start_time = time.time()
last_activity = start_time
heartbeat_timeout = 60  # Fail if no output for 60s

try:
    for line in proc.stdout:
        output_lines.append(line)
        last_activity = time.time()
        
        # Check for heartbeat
        if "[HEARTBEAT]" in line:
            print(f"  {line.strip()}")  # Show progress
        
        # Check total timeout
        if time.time() - start_time > test["timeout"]:
            proc.kill()
            result["failure_mode"] = "timeout"
            result["error"] = f"Total timeout after {test['timeout']}s"
            break
        
        # Check inactivity timeout
        if time.time() - last_activity > heartbeat_timeout:
            proc.kill()
            result["failure_mode"] = "inactivity_timeout"
            result["error"] = f"No output for {heartbeat_timeout}s"
            break
    
    proc.wait(timeout=5)
    result["output"] = "".join(output_lines)
    
except Exception as e:
    proc.kill()
    result["error"] = str(e)
```

**Benefit:**
- No buffer deadlock
- Detects both total timeout AND inactivity
- Shows real-time progress via heartbeat messages
- Distinguishes "agent hung" from "task taking long time"

---

## Implementation Priority

### Must Have (Fixes 59% of failures)

1. ✅ **Layer 2**: Wrap goal decomposition with timeout (5 min)
2. ✅ **Layer 1**: Add heartbeat logging (10 min)

### Should Have (Improves debugging)

3. ⚠️ **Layer 4**: Improve test framework timeout detection (15 min)

### Nice to Have (Defense in depth)

4. 🔧 **Layer 3**: Health check before each LLM call (20 min)

---

## Expected Results

After implementing Layers 1-2:

- **Timeout failures should drop from 59% to <10%**
- Failed tests will show exactly WHERE they hung:
  - "Decomposing goal" → Goal decomposition issue
  - "ROUND X" → Actual agent logic issue
- Tests won't hang silently - will fail fast with clear errors

After implementing all layers:

- **Robust against Ollama degradation**
- **Clear failure messages** instead of silent timeouts
- **Real-time progress visibility** during long tests

---

## Testing Plan

1. **Unit test** `decompose_goal()` timeout:
   ```bash
   python -c "from agent import decompose_goal; decompose_goal('test goal')"
   ```

2. **Integration test** with mock slow Ollama:
   - Temporarily make Ollama slow (via network throttling or mock)
   - Run L4-3 (the test that passed iteration 1, failed iteration 2)
   - Should fail fast with clear error, not silent 360s timeout

3. **Full regression test**:
   - Re-run complete L3-L7 test suite (75 tests)
   - Expect timeout failures to drop from 10/17 to <2/17

---

## Risk Assessment

**Low Risk:**
- Changes are defensive (add timeouts where none existed)
- Fallback behavior maintains functionality
- Heartbeat logging is non-invasive (just print statements)

**Testing Required:**
- Verify goal decomposition fallback works for complex goals
- Ensure heartbeat messages don't interfere with parsing
- Test with both fast and slow Ollama responses

