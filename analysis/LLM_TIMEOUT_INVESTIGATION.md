# LLM Timeout Investigation

**Date**: 2025-10-29
**Issue**: Round 13 of L5-L7 benchmark took 11+ minutes for single LLM call
**Status**: Root cause identified

---

## Incident Summary

During L5-L7 strategy benchmark execution, round 13 of the hierarchical strategy on L5_blog_system task experienced an 11-minute LLM call:

- **Round 12**: 49.6s total runtime, 4.46s avg LLM call
- **Round 13**: 12m 26s total runtime, 62.19s avg LLM call
- **Delta**: ~696 seconds (~11.6 minutes) added in single round
- **LLM calls**: 12 calls total, but only 11 actions executed (one call produced no tool calls)

---

## Root Cause

**The timeout protection did NOT fail - Ollama legitimately took 11+ minutes to generate a response while actively streaming chunks.**

### How Current Timeout System Works

Located in `llm_utils.py:18-104`, the `chat_with_inactivity_timeout()` function:

1. **Uses inactivity detection, not total time limits**
2. **Default inactivity timeout: 30 seconds**
3. **Streams responses from Ollama in chunks**
4. **Only times out if NO chunks received for 30 seconds**

```python
def chat_with_inactivity_timeout(
    model: str,
    messages: list,
    options: dict,
    inactivity_timeout: int = 30,  # Only detects inactivity
    tools: list | None = None,
) -> dict[str, Any]:
    # ...
    for chunk in OLLAMA_CLIENT.chat(**chat_args):
        result_queue.put(("chunk", chunk))  # Activity signal
    # ...
    msg_type, data = result_queue.get(timeout=inactivity_timeout)
    # Times out only if no chunk for 30s
```

### Why Protection Didn't Trigger

The timeout **correctly did NOT trigger** because:

1. **Ollama was actively streaming**: Continuously sending chunks throughout the 11 minutes
2. **No period of inactivity >30s**: Each chunk reset the inactivity timer
3. **System working as designed**: Inactivity detection is for detecting hung/crashed Ollama, not slow generation

### What Actually Happened

Ollama's `gpt-oss:20b` model took 11+ minutes to generate a response, likely due to:

1. **Large context size**: Round 13 had accumulated significant context
2. **Complex tool calls**: Model was deciding between multiple tools
3. **Model thinking deeply**: 20B parameter model doing extensive reasoning
4. **Hierarchical context**: Full goal/task/subtask hierarchy + all messages

---

## Where Timeout Is (Not) Configured

### In `task_executor_agent.py`

**Line 297** - Main LLM call:
```python
response = chat_with_inactivity_timeout(
    model=self.model,
    messages=context,
    tools=self.get_tools(),
    options={"temperature": self.temperature},
    # ❌ NO inactivity_timeout parameter passed - uses default 30s
)
```

**Line 219-226** - Jetbox notes LLM call:
```python
def _llm_caller_for_jetbox(self, messages, temperature=0.2, timeout=30):
    """LLM caller for jetbox notes."""
    return chat_with_inactivity_timeout(
        model=self.model,
        messages=messages,
        options={"temperature": temperature},
        inactivity_timeout=timeout,  # ✅ Explicitly passes 30s
    )
```

### No Total Time Limit

**There is no maximum total time limit on LLM calls** - only inactivity detection. This is by design to allow complex tasks to take as long as needed.

---

## Why This Is a Problem

### 1. **User Experience**
- 11-minute LLM calls feel like the system is hung
- No feedback to user that model is still thinking
- Benchmarks become impractical (6 tasks × 11 min = 1+ hour)

### 2. **Resource Waste**
- Ollama consuming resources for extended periods
- Other processes blocked waiting for response
- Context window might be too large (inefficient generation)

### 3. **Practical Limits**
- Benchmark timeout (20 minutes) barely accommodates one slow call
- Multiple slow calls would exceed reasonable wait times
- Production use would be frustrating

---

## How It "Got Around" Protection

**It didn't.** The protection is working exactly as designed:

- **Design goal**: Detect when Ollama is hung/crashed (no activity)
- **Not designed for**: Limiting total generation time
- **Behavior**: Allows infinite time as long as chunks are flowing

This is actually a **feature** for complex tasks, but becomes problematic when:
1. Context is too large (model slows down)
2. Model gets into inefficient generation pattern
3. No user feedback during long generation

---

## Evidence from Logs

### Round 12 → Round 13 Transition

**Round 12** (`l5_l7_detailed_output.log`):
```
AGENT STATUS - Round 12 | Runtime: 49.6s
PERFORMANCE:
  Avg LLM call:      4.46s
  LLM calls:         11
  Actions executed:  11
```

**Round 13** (immediately after):
```
AGENT STATUS - Round 13 | Runtime: 12m 26s
PERFORMANCE:
  Avg LLM call:      62.19s  # (11*4.46 + 696) / 12 ≈ 62s
  LLM calls:         12
  Actions executed:  11  # ⚠️ Call 12 produced no action
```

### Key Observations

1. **LLM call 12 took ~696 seconds** (~11.6 minutes)
2. **No timeout exception** - call completed successfully
3. **No action resulted** - LLM responded but didn't call tools (possibly just text)
4. **Status shows "Tokens (est): 0"** - token estimation might be broken

---

## Recommendations

### Short Term (Immediate)

1. **Add total time limit parameter** to `chat_with_inactivity_timeout()`:
   ```python
   def chat_with_inactivity_timeout(
       model: str,
       messages: list,
       options: dict,
       inactivity_timeout: int = 30,
       max_total_time: int | None = None,  # NEW: Optional total time limit
       tools: list | None = None,
   ) -> dict[str, Any]:
   ```

2. **Configure per-call total timeout** in `task_executor_agent.py`:
   ```python
   response = chat_with_inactivity_timeout(
       model=self.model,
       messages=context,
       tools=self.get_tools(),
       options={"temperature": self.temperature},
       inactivity_timeout=30,  # Still detect hung Ollama
       max_total_time=180,     # NEW: Fail if >3 minutes total
   )
   ```

3. **Add progress feedback** during long LLM calls:
   - Print "Thinking..." after 10s
   - Print "Still thinking..." every 30s
   - Show token generation rate

### Medium Term

4. **Fix token estimation** (showing 0 in logs):
   - `estimate_context_size()` appears broken
   - Should help identify context explosion issues

5. **Add context size warnings**:
   - Warn if context >8K tokens
   - Consider aggressive compaction if context >16K tokens
   - Alert if context approaching model limit

6. **Benchmark timeout protection**:
   - Per-task timeout (not just per-test-suite)
   - Fail fast and record timeout as metric
   - Continue to next task on timeout

### Long Term

7. **Context management improvements**:
   - More aggressive message compaction
   - Prune old tool results after summarization
   - Hierarchical strategy should clear more aggressively

8. **Model selection**:
   - Consider smaller models (7B instead of 20B) for benchmarks
   - 20B models are slower but not necessarily better for coding

9. **Streaming feedback to user**:
   - Show partial LLM output as it streams
   - Display token generation rate
   - Visual indicator that system is active

---

## Related Issues

1. **Token estimation broken**: `Tokens (est): 0` in all status displays
   - Located in `context_strategies.py` `estimate_context_size()` methods
   - Not being called or returning 0

2. **No max_rounds enforcement visible**: Benchmark allowed 40 rounds for L5 but no clear enforcement shown

3. **Context isolation not preventing growth**: Hierarchical strategy clears messages but context still grows

---

## Testing Plan

To validate fixes:

1. **Add max_total_time=180** (3 minutes) to main LLM call
2. **Re-run L5 task with hierarchical strategy**
3. **Verify timeout triggers if >3 minutes**
4. **Check that inactivity detection still works** (test with stopped Ollama)

---

## Conclusion

The 11-minute LLM call was NOT a bug in the timeout system - the system worked as designed. However, the design lacks a **total time limit**, only detecting **inactivity**.

**Immediate action**: Add `max_total_time` parameter to prevent indefinitely long LLM calls while preserving inactivity detection for hung processes.

**Root cause**: Design assumption that "as long as it's streaming, it's making progress" doesn't account for pathologically slow generation due to context size or model behavior.

---

## Files to Modify

1. **`llm_utils.py:18-104`** - Add `max_total_time` parameter and enforcement
2. **`task_executor_agent.py:297`** - Pass `max_total_time=180` to LLM calls
3. **`context_strategies.py`** - Fix `estimate_context_size()` to return actual token counts

---

## Open Questions

1. **Why did Ollama take 11 minutes?**
   - Context size? (token estimation broken so can't tell)
   - Model behavior? (20B model thinking deeply)
   - Ollama issue? (unlikely - completed successfully)

2. **Why no tool calls on call 12?**
   - Did LLM just return text?
   - Did it refuse to call tools?
   - Was response truncated?

3. **Is context growing unbounded?**
   - Hierarchical clears messages but keeps system prompt + goal/task/subtask
   - Need to verify total context size over rounds
