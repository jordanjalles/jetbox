# Timeout Fix V2 - Inactivity-Based Detection

## What Changed

### V1 (Broken - Caused Regressions)
- **Timeout type**: Total execution time (120 seconds)
- **Problem**: Cut off complex tasks that needed >120s to think
- **Result**: L5-2 dropped from 70% → 33% success rate

### V2 (Correct - Inactivity Detection)
- **Timeout type**: Inactivity timeout (30 seconds of no response)
- **Benefit**: Allows unlimited thinking time, only detects when Ollama is actually hung
- **Expected result**: Should NOT regress complex tasks

## The Key Insight

**Bad**: "You have 120 seconds to finish decomposing this task"
- Kills legitimate thinking for complex tasks
- Forces premature cutoff

**Good**: "You can think as long as you want, but if you go silent for 30 seconds, you're hung"
- Distinguishes "actively thinking" from "completely hung"
- Uses HTTP streaming to detect activity

## How It Works

```python
# Stream responses from Ollama
for chunk in chat(model=MODEL, messages=[...], stream=True):
    # Each chunk = Ollama is alive and working
    # If no chunk arrives for 30s = Ollama is hung
```

**Key difference**:
- **Thinking for 5 minutes**: ✅ OK if chunks arrive continuously
- **Silent for 30 seconds**: ❌ Timeout - Ollama is dead

## Implementation

**File**: `agent.py`

**Function**: `chat_with_inactivity_timeout()`
- Uses Ollama's streaming API
- Monitors chunk arrival
- Timeouts only if no activity for 30 seconds

**Usage**:
```python
resp = chat_with_inactivity_timeout(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.1},
    inactivity_timeout=30,  # 30s of silence = hung
)
```

## Better Fallback Too

Even if timeout does occur, the new fallback is more useful:

**V1 Fallback** (useless):
```python
[{"description": goal, "subtasks": ["Complete the goal"]}]
```

**V2 Fallback** (workable):
```python
[
    {"description": "Understand and plan",
     "subtasks": ["Read relevant files", "Identify what needs to be done"]},
    {"description": "Implement changes",
     "subtasks": ["Make necessary code changes"]},
    {"description": "Verify and test",
     "subtasks": ["Run tests", "Check code quality"]}
]
```

This gives the agent a concrete structure to work with.

## Expected Impact

### Complex Tasks (L5-2)
- **Before V1**: 70% success (no timeout)
- **After V1**: 33% success (premature cutoff)
- **After V2**: Should return to ~70% (no premature cutoff)

### Simple Tasks (L3-2, L4-1)
- **Before V1**: 70% success (some hangs)
- **After V1**: 80-90% success (timeout caught hangs)
- **After V2**: Should maintain 80-90% (still catches true hangs)

## Why This Is Correct

This matches how HTTP clients work:
- `requests.get(url, timeout=30)` means "no activity for 30s = timeout"
- NOT "entire request must complete in 30s"

Our original approach was like forcing a file download to complete in 30 seconds total, even if it's a huge file downloading steadily!

The inactivity approach correctly detects:
- ✅ Ollama process crashed
- ✅ Network connection lost
- ✅ Ollama completely hung/frozen
- ❌ Ollama thinking hard (as long as chunks arrive)
