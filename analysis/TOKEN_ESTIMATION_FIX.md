# Token Estimation Fix

**Date**: 2025-10-29
**Issue**: Token estimation always showing 0 in status display
**Status**: ✅ Fixed and tested

---

## Problem

During benchmarks and agent runs, the status display always showed:
```
Tokens (est):      0
```

This made it impossible to:
- Detect context size issues
- Identify when context was growing too large
- Understand why LLM calls were slow
- Debug the 11-minute LLM timeout incident

---

## Root Cause Analysis

### Issue 1: Token Counts Not Preserved in `llm_utils.py`

**Location**: `llm_utils.py:60-78`

The `chat_with_inactivity_timeout()` function streams responses from Ollama but only preserved the message content and tool calls:

```python
# BEFORE (broken)
full_response = {"message": {"role": "assistant", "content": ""}}

for chunk in OLLAMA_CLIENT.chat(**chat_args):
    result_queue.put(("chunk", chunk))

    content = chunk.get("message", {}).get("content", "")
    if content:
        full_response["message"]["content"] += content

    tool_calls = chunk.get("message", {}).get("tool_calls")
    if tool_calls:
        full_response["message"]["tool_calls"] = tool_calls

# ❌ Token counts (prompt_eval_count, eval_count) were discarded!
result_queue.put(("done", full_response))
```

**Ollama Response Structure**:
```python
{
    'model': 'gpt-oss:20b',
    'done': True,
    'prompt_eval_count': 69,      # ← Input tokens (context size)
    'eval_count': 34,              # ← Output tokens (generation)
    'prompt_eval_duration': 35811300,
    'eval_duration': 426102200,
    'message': {
        'role': 'assistant',
        'content': 'Hello! 🌟',
        'tool_calls': [...]
    }
}
```

### Issue 2: Status Display Used Wrong Field

**Location**: `status_display.py:181-186`

The status display expected `eval_count` (output tokens only) and used a rough estimation:

```python
# BEFORE (broken)
def record_llm_call(self, duration: float, messages_count: int) -> None:
    """Record an LLM call for statistics."""
    self.stats.llm_call_times.append(duration)
    self.stats.messages_sent += messages_count
    # Rough token estimation: ~100 tokens per message
    self.stats.total_tokens_estimated += messages_count * 100  # ❌ Completely wrong!
    self._save_stats()
```

**Called from** `task_executor_agent.py:307`:
```python
self.status_display.record_llm_call(duration, response.get("eval_count", 0))
#                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                              Defaults to 0 if not present!
```

Since `llm_utils.py` didn't preserve `eval_count`, this always defaulted to 0.

---

## Solution

### Fix 1: Preserve Token Counts in `llm_utils.py`

**File**: `llm_utils.py:77-82`

Added code to extract token counts from the final chunk:

```python
# AFTER (fixed)
for chunk in OLLAMA_CLIENT.chat(**chat_args):
    result_queue.put(("chunk", chunk))

    content = chunk.get("message", {}).get("content", "")
    if content:
        full_response["message"]["content"] += content

    tool_calls = chunk.get("message", {}).get("tool_calls")
    if tool_calls:
        full_response["message"]["tool_calls"] = tool_calls

    # ✅ NEW: Preserve token counts from final chunk
    if chunk.get("done"):
        chunk_dict = dict(chunk) if hasattr(chunk, "__dict__") or hasattr(chunk, "keys") else chunk
        for key in ["prompt_eval_count", "eval_count", "total_duration", "prompt_eval_duration", "eval_duration"]:
            if key in chunk_dict:
                full_response[key] = chunk_dict[key]

result_queue.put(("done", full_response))
```

### Fix 2: Use Actual Token Counts in `status_display.py`

**File**: `status_display.py:181-194`

Updated to accept and use real token counts:

```python
# AFTER (fixed)
def record_llm_call(self, duration: float, eval_count: int, prompt_eval_count: int = 0) -> None:
    """
    Record an LLM call for statistics.

    Args:
        duration: Time taken for LLM call
        eval_count: Number of tokens generated (output)
        prompt_eval_count: Number of tokens in prompt (input)
    """
    self.stats.llm_call_times.append(duration)
    self.stats.messages_sent += 1
    # ✅ Use actual token counts from Ollama
    self.stats.total_tokens_estimated += prompt_eval_count + eval_count
    self._save_stats()
```

### Fix 3: Pass Both Token Counts from Agent

**File**: `task_executor_agent.py:307-311`

Updated call site to pass both input and output tokens:

```python
# AFTER (fixed)
if self.status_display:
    self.status_display.record_llm_call(
        duration,
        response.get("eval_count", 0),        # Output tokens
        response.get("prompt_eval_count", 0)  # Input tokens (context size)
    )
```

---

## Testing

Created and ran test: `test_token_estimation.py`

**Results**:
```
AGENT STATUS - Round 2 | Runtime: 1.3s
PERFORMANCE:
  Avg LLM call:      1.27s
  Tokens (est):      1,421   ✅ Working!

AGENT STATUS - Round 3 | Runtime: 2.1s
PERFORMANCE:
  Avg LLM call:      1.00s
  Tokens (est):      2,895   ✅ Growing as expected!

✓ LLM calls: 3
✓ Total tokens: 4,400
✓ Messages sent: 3

✅ Token estimation is working!
```

**Token Breakdown**:
- Round 2: 1,421 tokens (prompt + response)
- Round 3: 2,895 tokens (context grew)
- Total: 4,400 tokens across 3 LLM calls

---

## Impact

### Before Fix
- Token counts always showed 0
- No visibility into context size growth
- Could not debug context-related performance issues
- No warning when approaching model limits

### After Fix
- Accurate token counts displayed in real-time
- Can track context growth across rounds
- Can identify when context optimization is needed
- Helps debug slow LLM calls (large context = slow generation)

---

## Related Issues Fixed

This fix also helps diagnose the **11-minute LLM call issue** from the L5-L7 benchmark:
- We can now see if context size was the root cause
- Can detect context explosions early
- Can set thresholds for context size warnings

---

## Files Modified

1. **`llm_utils.py:77-82`** - Preserve token counts from Ollama chunks
2. **`status_display.py:181-194`** - Accept and use actual token counts
3. **`task_executor_agent.py:307-311`** - Pass both prompt and output tokens

---

## Validation

Run any agent task and verify status display shows non-zero token counts:

```bash
python agent_v2.py "Create a hello.py file"
```

Expected output:
```
PERFORMANCE:
  Tokens (est):      1,234   # Should be non-zero
```

---

## Notes

- Token counts are **exact** from Ollama, not estimates
- `prompt_eval_count` = input context size (what we send)
- `eval_count` = output tokens (what model generates)
- Total tokens = prompt_eval_count + eval_count
- Useful for tracking context growth and identifying optimization opportunities

---

## Future Enhancements

1. **Add context size warnings**:
   - Warn if context >8K tokens
   - Alert if context >16K tokens
   - Force compaction if approaching model limit

2. **Track token efficiency**:
   - Tokens per action
   - Context growth rate
   - Compare strategies by token usage

3. **Display breakdown**:
   - Show input vs output token split
   - Track token usage per subtask
   - Identify which operations are most expensive
