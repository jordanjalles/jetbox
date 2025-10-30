# JSON Parsing Error Fix: Analysis and Solution

**Date:** 2025-10-30
**Issue:** L3-L7 evaluation hitting JSON parsing errors in tool calls
**Error Pattern:** `error parsing tool call: raw='...' err=invalid character ... (status code: 500/-1)`

## Root Cause Analysis

### The Problem

The LLM (gpt-oss:20b) occasionally generates text **before** the JSON in tool call responses:

**Example 1 (L7_rate_limiter):**
```
We wrote the file. Need to create tests.{"path":"tests/test_rate_limiter.py","content":"..."}
                                        ^--- JSON starts here
```

**Example 2 (L4_todo_list):**
```
{"path":"todo.py","content":"..."},"append":false,"encoding":"utf-8","line_end":null,"overwrite":true}
                                   ^--- Extra parameters invented by LLM
```

**Example 3 (L7_rate_limiter bash):**
```
{"command":"python - <<'PY'\n...code...\nPY"]}
                       ^--- Single quotes in heredoc break JSON
```

### How Ollama Tool Calling Works

1. **LLM generates response** with tool calls
2. **Ollama parses the response** to extract tool call JSON
3. **If parsing fails**, Ollama raises `ResponseError` with:
   - `error`: String like "error parsing tool call: raw='...' err=invalid character ..."
   - `status_code`: 500 (server error) or -1 (parse error)
4. **Our code catches exception** and converts to string, which becomes the error in results

### Why This Happens

**Normal Ollama response format:**
```python
{
  "message": {
    "role": "assistant",
    "content": "",  # Empty when using tools
    "tool_calls": [
      {
        "function": {
          "name": "write_file",
          "arguments": {"path": "...", "content": "..."}  # Already parsed as dict
        }
      }
    ]
  }
}
```

**When LLM misbehaves:**
- LLM puts commentary in `content` AND tries to generate tool call
- LLM generates malformed JSON (text before JSON, extra parameters, etc.)
- Ollama can't parse → raises ResponseError
- We get string error instead of structured tool calls

## Error Categories

### 1. Text Before JSON (Most Common)

**Pattern:**
```
We wrote the file. Need to create tests.{"path":...}
```

**Why it happens:**
- LLM is trained to be conversational
- Explains its actions before taking them
- Doesn't understand tool calls should be pure JSON

**Fix:** Prompt engineering to suppress commentary OR post-process to extract JSON

### 2. Invalid JSON (Parameter Invention)

**Pattern:**
```
{"path":"todo.py","content":"..."},"append":false,"encoding":"utf-8","line_end":null,"overwrite":true}
                                   ^--- Comma makes this invalid JSON
```

**Why it happens:**
- LLM invents parameters not in schema
- Tries to be helpful by adding options
- Creates syntactically invalid JSON (multiple top-level objects)

**Fix:** **kwargs approach (already implemented) OR strict prompt engineering

### 3. JSON String Escaping Issues

**Pattern:**
```
{"command":"python - <<'PY'\n..."}
                       ^--- Single quotes need escaping in JSON
```

**Why it happens:**
- Bash heredoc syntax uses single quotes: `<<'EOF'`
- Single quotes in JSON strings must be escaped
- LLM doesn't escape them properly

**Fix:** Revert bash migration (use write_file with **kwargs) OR teach LLM to escape

## Current State Analysis

### From BASH_HEREDOC_JSON_ERROR.md

The bash migration document identified:
- **67% failure rate** from parameter invention (before bash migration)
- **59% failure rate** from JSON parsing after bash migration
- **Marginal improvement** of only 8%
- **Trade-off:** Swapped one error type for another

### Affected Tasks

From `l3_l7_context_strategy_results.json`:

1. **L7_rate_limiter (hierarchical):** Text before JSON
2. **L4_todo_list (append):** Invalid JSON (extra parameters)
3. **L7_rate_limiter (append):** Heredoc quote escaping

**Pattern:** Both strategies affected, different error types

## Solution Options

### Option 1: Robust JSON Extraction (Recommended)

**Approach:** Post-process Ollama errors to extract JSON from mixed responses

**Implementation:**
```python
def extract_json_from_mixed_response(error_msg: str) -> dict | None:
    """
    Extract JSON from mixed text+JSON error messages.

    Example input:
        "error parsing tool call: raw='Text before.{"key":"value"}', err=..."

    Returns parsed JSON dict or None if extraction fails.
    """
    import re
    import json

    # Extract the raw string from error message
    match = re.search(r"raw='(.*?)'(?:,\s*err=|$)", error_msg, re.DOTALL)
    if not match:
        return None

    raw = match.group(1)

    # Try to find JSON in the string
    # Strategy 1: Look for {...} pattern
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Look for [...] pattern (array)
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
```

**Where to implement:**
- In `task_executor_agent.py` around line 449 (LLM call)
- Catch `ollama.ResponseError` specifically
- Extract JSON and construct tool_calls manually
- Log warning about fallback parsing

**Pros:**
- Handles text-before-JSON errors
- No prompt engineering needed
- Works with current model behavior
- Backward compatible

**Cons:**
- Heuristic approach (not guaranteed)
- Can't fix invalid JSON (extra parameters)
- May extract wrong JSON if multiple objects present

### Option 2: Strict Prompt Engineering

**Approach:** Modify system prompt to prevent commentary

**Changes to `agent_config.yaml`:**
```yaml
system_prompt: |
  ...

  CRITICAL TOOL CALLING RULES:
  1. When calling tools, provide ONLY the tool call - NO explanatory text
  2. Do NOT add commentary before or after tool calls
  3. Do NOT invent parameters not in the tool schema
  4. Ensure all JSON is properly escaped (use \" for quotes, not ')

  BAD: We wrote the file. {"path": "test.py", "content": "..."}
  GOOD: {"path": "test.py", "content": "..."}

  BAD: {"command": "cat > f.py <<'EOF'\ncode\nEOF"}
  GOOD: {"command": "cat > f.py <<EOF\ncode\nEOF"}
```

**Pros:**
- Addresses root cause (LLM behavior)
- Clean solution if it works
- No code changes needed

**Cons:**
- May not work reliably (LLM may ignore)
- Requires testing/iteration
- Model-specific (may break with other models)

### Option 3: Retry on Parse Error

**Approach:** Catch parse errors and retry with clarifying prompt

**Implementation:**
```python
MAX_RETRIES = 2

for attempt in range(MAX_RETRIES):
    try:
        response = chat_with_inactivity_timeout(...)
        break  # Success
    except ollama.ResponseError as e:
        if "error parsing tool call" in str(e) and attempt < MAX_RETRIES - 1:
            # Add clarifying message
            messages.append({
                "role": "user",
                "content": "Error: Tool call had invalid JSON. Please provide ONLY the JSON tool call with no explanatory text before or after."
            })
            continue  # Retry
        else:
            raise  # Give up
```

**Pros:**
- Gives LLM a chance to self-correct
- Can work for transient errors
- Educational for the model

**Cons:**
- Adds latency (retry takes time)
- May not help if model is confused
- Could loop if model keeps making same error

### Option 4: Hybrid Approach (Best)

**Combine Option 1 + Option 2:**

1. **Improve prompt** to reduce errors (Option 2)
2. **Add fallback parsing** for when errors still occur (Option 1)
3. **Log warnings** when fallback is used (for monitoring)

**Benefits:**
- Defense in depth (multiple layers)
- Reduces errors through prompt engineering
- Handles remaining errors gracefully
- Provides metrics on error frequency

## Recommended Implementation

### Phase 1: Add Robust JSON Extraction (Immediate)

**File:** `/workspace/llm_utils.py`

Add helper function:
```python
def extract_tool_call_from_error(error: ollama.ResponseError) -> dict | None:
    """Extract tool call JSON from Ollama parsing error."""
    # Implementation from Option 1
    ...
```

**File:** `/workspace/task_executor_agent.py` (line ~449)

Wrap LLM call:
```python
try:
    response = chat_with_inactivity_timeout(...)
except ollama.ResponseError as e:
    if "error parsing tool call" in str(e):
        # Try to extract JSON from error
        extracted = extract_tool_call_from_error(e)
        if extracted:
            print(f"[llm_fallback] Recovered tool call from parse error")
            # Construct response with extracted tool call
            response = {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": extracted}]
                }
            }
        else:
            raise  # Can't recover, re-raise
    else:
        raise  # Different error, re-raise
```

### Phase 2: Improve Prompt Engineering (Follow-up)

Update system prompt to add tool calling rules.

### Phase 3: Add Metrics (Monitoring)

Track how often fallback parsing is used:
```python
self.perf_stats.record_parse_error_recovery()
```

## Testing Plan

### 1. Unit Test for JSON Extraction

```python
def test_extract_json_from_mixed_response():
    # Test case 1: Text before JSON
    error1 = "error parsing tool call: raw='Text here.{\"key\":\"value\"}', err=..."
    result1 = extract_json_from_mixed_response(error1)
    assert result1 == {"key": "value"}

    # Test case 2: No JSON found
    error2 = "error parsing tool call: raw='Just text', err=..."
    result2 = extract_json_from_mixed_response(error2)
    assert result2 is None

    # Test case 3: Multiple JSON objects (take first)
    error3 = "error parsing tool call: raw='{\"a\":1},{\"b\":2}', err=..."
    result3 = extract_json_from_mixed_response(error3)
    assert result3 == {"a": 1}
```

### 2. Integration Test with Real Error

Reproduce the L7_rate_limiter error and verify fix:
```python
# Manually construct ResponseError
error = ollama.ResponseError(
    "error parsing tool call: raw='We wrote file.{\"path\":\"test.py\"}', err=..."
)
# Verify extraction works
extracted = extract_tool_call_from_error(error)
assert extracted["path"] == "test.py"
```

### 3. Re-run L3-L7 Evaluation

Run `tests/test_context_strategies_l3_l7.py` and compare:
- **Before:** 3/20 tasks pass (JSON errors kill progress)
- **After:** Expected >10/20 tasks pass (JSON errors recovered)

## Success Metrics

### Immediate (After Phase 1)

- JSON parsing errors reduced from 15% to <5%
- L3-L7 pass rate improved by 20-30%
- No crashes from recoverable parse errors

### Long-term (After Phase 2)

- JSON parsing errors <2% (rare edge cases only)
- L3-L7 pass rate >60%
- LLM learns cleaner tool call format

## Implementation Status

- [x] Root cause analysis complete
- [x] Solution options evaluated
- [ ] JSON extraction function implemented
- [ ] Error recovery integrated in task_executor_agent.py
- [ ] Unit tests written
- [ ] Integration tests pass
- [ ] L3-L7 evaluation shows improvement

## Next Steps

1. Implement `extract_tool_call_from_error()` in `llm_utils.py`
2. Add error recovery wrapper in `task_executor_agent.py`
3. Write unit tests for JSON extraction
4. Re-run L3-L7 evaluation
5. Measure improvement in pass rate
6. Document results in follow-up report

---

**Conclusion:** The JSON parsing errors are caused by the LLM generating text before JSON in tool calls, which breaks Ollama's parser. The recommended fix is a hybrid approach: (1) add robust JSON extraction as a fallback, (2) improve prompt engineering to reduce errors. This provides both immediate relief and long-term improvement.
