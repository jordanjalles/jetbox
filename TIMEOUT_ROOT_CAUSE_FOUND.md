# Root Cause Found: Main Loop LLM Call Has No Timeout

## The Smoking Gun

**Line 1394 in agent.py:**
```python
resp = chat(              # ← NO TIMEOUT PROTECTION!
    model=MODEL,
    messages=context,
    tools=tool_specs(),
    options={"temperature": TEMP},
    stream=False,
)
```

This is the **ONLY** `chat()` call in the entire codebase that doesn't use `chat_with_inactivity_timeout()`.

## Why This Causes 59% Timeout Failures

### Timeline of a Timeout

1. ✅ Agent starts, checks Ollama health (has timeout)
2. ✅ Goal decomposition happens (uses `chat_with_inactivity_timeout()` - line 971)
3. ✅ Agent enters main loop, prints "INITIAL TASK TREE"
4. ✅ Round 1 starts, calls `probe_state_generic()`, displays status
5. ✅ Builds context for LLM call
6. ❌ **Calls `chat()` WITHOUT timeout wrapper**
7. ❌ **Ollama hangs or degrades → Agent hangs indefinitely**
8. ❌ Test times out after 240-480s with **0 rounds recorded**

### Why 0 Rounds Are Recorded

The test framework counts rounds by looking for output patterns. But:
- Round count is incremented AFTER the LLM call (line 1417)
- If LLM call hangs, the round never completes
- Test sees 0 rounds and times out

### Why L4-3 Passed Iteration 1 But Failed Iteration 2

**Iteration 1:**
- Ollama is fresh, responds quickly
- Agent completes in 29 rounds (106.2s)

**Iteration 2 (same test):**
- Ollama has been running for a while, may be degraded
- First LLM call in main loop hangs
- Test times out after 360s with 0 rounds

This is EXACTLY the pattern we see in the failure data!

## The Fix (2 lines)

Change line 1394 from:
```python
        try:
            resp = chat(
                model=MODEL,
                messages=context,
                tools=tool_specs(),
                options={"temperature": TEMP},
                stream=False,
            )
```

To:
```python
        try:
            resp = chat_with_inactivity_timeout(
                model=MODEL,
                messages=context,
                tools=tool_specs(),
                options={"temperature": TEMP},
                inactivity_timeout=30,  # Fail if Ollama stops responding
            )
```

**Note:** `chat_with_inactivity_timeout()` doesn't support `stream=False` parameter, but it implements streaming internally, so we just remove that parameter.

## Why This Was Missed

Looking at the code history:
- `chat_with_inactivity_timeout()` was added to fix timeout issues (line 38)
- Goal decomposition was updated to use it (line 971)
- But the main loop was missed, still using raw `chat()` (line 1394)

This is a classic "we fixed it in one place but not the other" bug.

## Expected Impact

**Before fix:**
- 10/17 failures (59%) are timeouts with 0 rounds
- Silent hangs, no error messages
- Tests waste 240-480s waiting for timeout

**After fix:**
- Timeouts drop to near zero (only genuine Ollama crashes)
- Fast failure with clear error: "No response from Ollama for 30s"
- Tests fail in ~30s instead of 240-480s
- Agent can retry or fail gracefully

## Why Heartbeat Logging Is Still Useful

Even with the timeout fix, heartbeat logging helps debug:
- WHERE in initialization something went wrong
- Progress visibility during long-running tests
- Distinguishes "agent working" from "agent hung"

So we should still implement Layer 1 (heartbeat logging) as a diagnostic tool.

## Implementation Plan

1. **Fix main loop LLM call** (2 lines) - CRITICAL
2. **Add heartbeat logging** (10 min) - Diagnostic aid
3. **Test with L4-3** specifically (the test that failed iteration 2)
4. **Re-run full L3-L7 suite** (expect timeout failures to drop from 10 to ~0)

