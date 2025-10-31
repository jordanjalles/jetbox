# Timeout Instrumentation and Context Dumping

**Date**: 2025-10-29
**Issue**: Need to diagnose 11+ minute LLM calls and detect hung Ollama
**Status**: ✅ Implemented and tested

---

## Overview

Added comprehensive timeout instrumentation to detect and diagnose:
1. **Hung Ollama** - No response chunks for 30+ seconds
2. **Slow generation** - Total time exceeds 3 minutes (context too large or model struggling)

When timeouts occur, the system automatically dumps the full context to `.agent_context/timeout_dumps/` for diagnosis.

---

## Features

### 1. Dual Timeout System

**Inactivity Timeout** (30 seconds):
- Detects when Ollama stops sending response chunks
- Indicates hung/crashed Ollama process
- Default: 30 seconds

**Total Time Limit** (180 seconds):
- Enforces maximum generation time regardless of activity
- Detects pathologically slow generation (large context, complex reasoning)
- Default: 3 minutes (configurable)

### 2. Automatic Context Dumping

When timeout occurs, saves complete diagnostic dump to:
```
.agent_context/timeout_dumps/timeout_{type}_{timestamp}.json
```

**Dump Contents**:
```json
{
  "timestamp": "20251029_233934",
  "timeout_type": "max_total_time",
  "elapsed_time_seconds": 3.01,
  "model": "gpt-oss:20b",
  "context_stats": {
    "message_count": 2,
    "total_chars": 92,
    "estimated_tokens": 23
  },
  "messages": [...],  // Full context sent to LLM
  "tools": [...]      // Tool definitions
}
```

### 3. User Feedback

Clear error messages with context location:
```
TimeoutError: LLM call exceeded max_total_time of 180s (elapsed: 183.4s).
Context dumped to .agent_context/timeout_dumps/
```

---

## Implementation

### Modified Files

**1. `llm_utils.py`** - Core timeout logic

Added `max_total_time` parameter:
```python
def chat_with_inactivity_timeout(
    model: str,
    messages: list,
    options: dict,
    inactivity_timeout: int = 30,
    tools: list | None = None,
    max_total_time: int | None = None,  # NEW
) -> dict[str, Any]:
```

Added timeout checking:
```python
# Track total time
start_time = time.time()

while True:
    msg_type, data = result_queue.get(timeout=inactivity_timeout)

    if msg_type == "chunk":
        # Check total time on each chunk
        elapsed = time.time() - start_time
        if max_total_time and elapsed > max_total_time:
            _dump_timeout_context(model, messages, tools, elapsed, "max_total_time")
            raise TimeoutError(...)
        continue
```

Added context dumping function:
```python
def _dump_timeout_context(
    model: str,
    messages: list,
    tools: list | None,
    elapsed_time: float,
    timeout_type: str,
) -> None:
    """Dump context to file when timeout occurs for diagnosis."""
    # Creates .agent_context/timeout_dumps/ directory
    # Saves JSON with messages, tools, stats, and metadata
```

**2. `task_executor_agent.py`** - Applied timeout to main LLM call

```python
# BEFORE
response = chat_with_inactivity_timeout(
    model=self.model,
    messages=context,
    tools=self.get_tools(),
    options={"temperature": self.temperature},
)

# AFTER
response = chat_with_inactivity_timeout(
    model=self.model,
    messages=context,
    tools=self.get_tools(),
    options={"temperature": self.temperature},
    inactivity_timeout=30,     # 30s inactivity = hung Ollama
    max_total_time=180,        # 3 minutes total = too slow
)
```

---

## Testing

Created comprehensive test suite: `test_timeout_instrumentation.py`

**Test 1: Normal completion (no timeout)**
```
messages = [{"role": "user", "content": "Say 'hello' in one word"}]
max_total_time=180  # Plenty of time

✓ Completed normally: "Hi"
✅ Normal completion working correctly!
```

**Test 2: Timeout with context dump**
```
messages = [{"role": "user", "content": "Write a very long detailed essay..."}]
max_total_time=3  # Will timeout during generation

✓ Timeout triggered correctly after 3.0s
✓ Context dump created: timeout_max_total_time_20251029_233934.json
✓ Dump contains 2 messages, ~23 tokens
✅ Timeout instrumentation working correctly!
```

**Results**:
```
✅ ALL TESTS PASSED
```

---

## Usage

### In Agent Code

Timeout is automatically applied to all main LLM calls in `TaskExecutorAgent.run()`.

### Configuring Timeout

To adjust timeout for specific scenarios:

```python
# Shorter timeout for simple tasks
response = chat_with_inactivity_timeout(
    model="gpt-oss:20b",
    messages=context,
    tools=tools,
    options={"temperature": 0.2},
    inactivity_timeout=30,
    max_total_time=60,  # 1 minute
)

# Longer timeout for complex reasoning
response = chat_with_inactivity_timeout(
    model="gpt-oss:20b",
    messages=context,
    tools=tools,
    options={"temperature": 0.2},
    inactivity_timeout=30,
    max_total_time=300,  # 5 minutes
)

# No total time limit (original behavior)
response = chat_with_inactivity_timeout(
    model="gpt-oss:20b",
    messages=context,
    tools=tools,
    options={"temperature": 0.2},
    inactivity_timeout=30,
    max_total_time=None,  # No limit
)
```

### Diagnosing Timeouts

When timeout occurs:

1. **Check error message** for elapsed time:
   ```
   TimeoutError: LLM call exceeded max_total_time of 180s (elapsed: 183.4s)
   ```

2. **Find dump file**:
   ```bash
   ls -lt .agent_context/timeout_dumps/
   ```

3. **Analyze dump**:
   ```bash
   cat .agent_context/timeout_dumps/timeout_max_total_time_20251029_233934.json
   ```

4. **Check context stats**:
   ```json
   "context_stats": {
     "message_count": 12,
     "total_chars": 45000,
     "estimated_tokens": 11250  // ← Large context!
   }
   ```

5. **Review messages** to understand what caused large context

---

## Timeout Types

### `inactivity` Timeout

**Trigger**: No chunks received for 30 seconds
**Cause**: Ollama hung, crashed, or network issue
**Action**:
- Check Ollama is running: `ollama list`
- Restart Ollama if needed
- Check network connectivity

**Dump filename**: `timeout_inactivity_{timestamp}.json`

### `max_total_time` Timeout

**Trigger**: Total time exceeds limit (default 180s)
**Cause**:
- Context too large (many messages, long tool results)
- Model struggling with complex reasoning
- Model size too large for hardware (20B on slow CPU)

**Action**:
- Review context stats in dump
- If tokens >8K, reduce context:
  - Clear old messages more aggressively
  - Summarize tool results
  - Use append strategy instead of hierarchical
- Consider smaller model (7B instead of 20B)
- Increase timeout if legitimately complex task

**Dump filename**: `timeout_max_total_time_{timestamp}.json`

---

## Context Stats Interpretation

From dump file:

```json
"context_stats": {
  "message_count": 25,
  "total_chars": 120000,
  "estimated_tokens": 30000
}
```

**Guidelines**:
- **< 2K tokens**: Normal, fast generation
- **2K-8K tokens**: Acceptable, moderate speed
- **8K-16K tokens**: Large, slow generation expected
- **16K-32K tokens**: Very large, consider compaction
- **> 32K tokens**: Excessive, will cause very slow generation

**Actions by token count**:

| Tokens | Speed | Action |
|--------|-------|--------|
| < 2K | Fast | No action needed |
| 2K-8K | Moderate | Monitor, consider clearing old messages |
| 8K-16K | Slow | Compact context, summarize tool results |
| 16K-32K | Very slow | Aggressive compaction required |
| > 32K | Pathological | Clear most messages, keep only essential context |

---

## Integration with Existing Systems

### Token Estimation

Works with fixed token estimation from `analysis/TOKEN_ESTIMATION_FIX.md`:
- Real-time token counts now displayed in status
- Can correlate timeout with token count
- Example: "Round 13 had 11K tokens, took 11 minutes"

### Status Display

Timeout errors are caught and reported in agent status:
```
AGENT STATUS - Round 13 | Runtime: 3m 4s
[timeout] LLM call exceeded max_total_time of 180s
Context dumped to .agent_context/timeout_dumps/timeout_max_total_time_20251029_233934.json
```

### Benchmarks

Prevents infinite benchmarks:
- L5-L7 tasks now fail fast if >3 minutes per call
- Allows benchmark to continue to next task
- Provides diagnostic data for performance investigation

---

## Preventing Timeouts

### 1. Context Management

**Use Append Strategy**:
- More efficient than hierarchical
- 48% fewer rounds on simple tasks
- Better context reuse

**Clear Old Messages**:
```python
# In hierarchical strategy
if self.context_strategy.should_clear_on_transition():
    self.clear_messages()
```

**Summarize Tool Results**:
```python
# Instead of keeping full 5000-char file contents
tool_result = "File created successfully: example.py (150 lines)"
```

### 2. Model Selection

- **7B models**: Faster generation, suitable for most tasks
- **20B models**: Better quality but 3x slower, use for complex reasoning only
- **3B models**: Very fast, good for simple tasks

### 3. Task Decomposition

Break large tasks into smaller subtasks:
- Reduces context per subtask
- Allows progress checkpoints
- Prevents single slow generation from blocking entire goal

---

## Future Enhancements

### 1. Adaptive Timeouts

Adjust timeout based on context size:
```python
# Rough estimate: 1 second per 1K tokens
estimated_tokens = len(str(messages)) // 4
adaptive_timeout = max(60, estimated_tokens // 1000 * 60)
```

### 2. Streaming Progress

Show user that generation is active:
```
[LLM] Generating... (15s elapsed, 450 tokens)
[LLM] Still generating... (30s elapsed, 890 tokens)
```

### 3. Automatic Compaction

If approaching timeout, interrupt and compact:
```python
if elapsed > max_total_time * 0.8:
    # Cancel current generation
    # Compact context
    # Retry with smaller context
```

### 4. Timeout Telemetry

Track timeout frequency:
- Which tasks timeout most often?
- What context sizes cause timeouts?
- Which models timeout more?

---

## Troubleshooting

### Frequent Timeouts

**Problem**: Every call timing out after 3 minutes

**Diagnosis**:
1. Check dump files for context stats
2. Look for pattern: always >15K tokens?
3. Review recent changes to context building

**Solutions**:
- Reduce `HISTORY_KEEP` in agent config
- Use append strategy instead of hierarchical
- Clear jetbox notes if too large
- Increase timeout to 5 minutes for this model

### No Dump File Created

**Problem**: Timeout occurs but no dump in `.agent_context/timeout_dumps/`

**Causes**:
1. Permission error writing to `.agent_context/`
2. Disk full
3. Exception during dump write

**Check**:
```bash
ls -la .agent_context/
df -h .agent_context/
```

### Dump Too Large

**Problem**: Dump files are huge (>100MB)

**Cause**: Messages contain large tool results (file contents, command output)

**Solution**: Truncate large content before dumping:
```python
# In _dump_timeout_context
for msg in messages:
    content = str(msg.get("content", ""))
    if len(content) > 10000:
        msg["content"] = content[:10000] + f"... [truncated {len(content)-10000} chars]"
```

---

## Related Issues

1. **11-minute LLM call** (`analysis/LLM_TIMEOUT_INVESTIGATION.md`)
   - Root cause: No total time limit
   - Fixed by: This implementation

2. **Token estimation** (`analysis/TOKEN_ESTIMATION_FIX.md`)
   - Now shows real token counts
   - Helps identify when context is too large

3. **Context strategy comparison** (`analysis/CONTEXT_STRATEGY_COMPARISON.md`)
   - Append strategy is more efficient
   - Helps prevent large context buildup

---

## Summary

**Before**:
- No total time limit on LLM calls
- 11-minute calls possible
- No diagnostic information
- Benchmarks could run forever

**After**:
- 3-minute total timeout (configurable)
- 30-second inactivity detection
- Automatic context dumps for diagnosis
- Clear error messages with stats
- Prevents runaway generation

**Result**: Faster failure detection, better diagnostics, improved user experience.
