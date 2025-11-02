# Tool Call Error Feedback Fix

**Date**: 2025-11-02
**Problem**: LLM generates malformed tool calls → agent fails silently → empty rounds → stuck loops
**Solution**: Immediate error feedback to LLM with actionable instructions

## The Problem

From the empty round analysis, we discovered:

```
[loop_detection] ⚠️ Empty round #1 - LLM did not call any tools
[loop_detection] LLM response: LLM call failed: error parsing tool call:
raw='{"content":"import json\nimport pytest\nfrom app import app...
```

**What was happening**:
1. LLM generates malformed tool call (e.g., adds text before JSON)
2. Ollama parser fails with `error parsing tool call`
3. Agent logs error but sends NO feedback to LLM
4. LLM doesn't know it did anything wrong
5. LLM tries again with same mistake
6. Loop continues → empty rounds → timeout

## The Root Cause

The LLM was generating output like:

```
Let me create a test file: {"name": "write_file", "arguments": {...}}
```

Instead of pure JSON:

```json
{"name": "write_file", "arguments": {...}}
```

**The agent was failing silently** - no feedback to the LLM about what went wrong.

## The Solution

**File**: `base_agent.py` lines 337-379

**Single point of implementation**: Exception handler in `call_llm()`

### Before (Silent Failure)

```python
except Exception as e:
    return {
        "message": {
            "role": "assistant",
            "content": f"LLM call failed: {e}",
        }
    }
```

Problems:
- Generic error message
- LLM doesn't understand what went wrong
- No instructions on how to fix it
- Role is "assistant" (not treated as feedback)

### After (Immediate Actionable Feedback)

```python
except Exception as e:
    error_str = str(e)

    # Check if it's a tool call parsing error
    if "error parsing tool call" in error_str.lower():
        # Extract the malformed output
        import re
        match = re.search(r"raw='(.*?)'", error_str, re.DOTALL)
        if match:
            malformed_output = match.group(1)[:200]
        else:
            malformed_output = "unknown"

        # Provide clear, actionable feedback
        feedback = (
            "ERROR: Your last response had a malformed tool call.\n\n"
            f"What you generated: {malformed_output}...\n\n"
            "PROBLEM: Tool calls must be pure JSON with NO text before or after.\n\n"
            "CORRECT FORMAT:\n"
            "  {\n"
            "    \"name\": \"tool_name\",\n"
            "    \"arguments\": {\"arg1\": \"value1\"}\n"
            "  }\n\n"
            "INCORRECT (what you did):\n"
            "  Let me do this: {\"name\": \"tool_name\", ...}  ← NO TEXT BEFORE JSON\n\n"
            "Try again with ONLY the JSON tool call, no explanatory text."
        )

        return {
            "message": {
                "role": "user",  # Treated as feedback/instruction
                "content": feedback,
            }
        }

    # For other errors, provide generic feedback
    return {
        "message": {
            "role": "user",
            "content": f"ERROR: LLM call failed with: {error_str}\n\nPlease try again.",
        }
    }
```

### Key Improvements

1. **Detects tool call parsing errors specifically**
   - Checks for "error parsing tool call" in exception message
   - Extracts the malformed output from the error

2. **Shows LLM what it did wrong**
   - Displays first 200 chars of malformed output
   - Makes it clear what the LLM generated

3. **Provides correct format example**
   - Shows proper JSON tool call structure
   - Contrasts with incorrect format

4. **Sends as user message, not assistant**
   - Role: "user" makes LLM treat it as feedback/instruction
   - LLM is more likely to correct behavior when feedback comes from "user"

5. **Actionable instructions**
   - Clear directive: "Try again with ONLY the JSON tool call"
   - Explains the rule: "NO text before or after"

## How It Works in Practice

### Previous Flow (Silent Failure)
```
Round 1: LLM generates "Let me do this: {tool_call}"
         → Parsing error
         → Error logged, no feedback

Round 2: LLM tries same thing (doesn't know it was wrong)
         → Parsing error again
         → Loop continues

Round 50: Max rounds reached → FAILURE
```

### New Flow (Immediate Feedback)
```
Round 1: LLM generates "Let me do this: {tool_call}"
         → Parsing error detected
         → Feedback injected into context:
            "ERROR: Your last response had a malformed tool call.
             What you generated: Let me do this: {tool_call}
             PROBLEM: Tool calls must be pure JSON...
             Try again with ONLY the JSON tool call."

Round 2: LLM sees feedback, corrects mistake
         → Generates pure JSON: {tool_call}
         → Success! Tool executes
```

## Why This Location?

**Question**: Why implement in `base_agent.py:call_llm()` exception handler?

**Answer**: This is the **single point** where all LLM call errors are caught.

- ✅ Every agent uses `call_llm()`
- ✅ Every LLM error passes through this handler
- ✅ Error is caught immediately after failure
- ✅ Feedback is added to message history automatically (line 1090 in run() loop)
- ✅ LLM sees feedback on next round
- ✅ No need to modify individual tools or behaviors

**Architecture Diagram**:
```
run() loop
  ├─> call_llm()  ← ERROR CAUGHT HERE
  │     ├─> chat_with_inactivity_timeout()
  │     │     └─> Ollama generates response
  │     │           └─> Parsing fails
  │     │                 └─> Exception raised
  │     └─> Exception handler ← FEEDBACK INJECTED HERE
  │           └─> Returns user message with feedback
  │
  └─> add_message(error_message)  ← Auto-added to history (line 1090)
        └─> LLM sees feedback on next round
```

## Testing

**Challenge**: Malformed tool calls are intermittent - they don't happen every time.

**Evidence it will work**:
1. Error messages ARE already added to history (line 1088-1090)
2. LLM DOES see them on next round
3. Problem was: error messages were too generic
4. Solution: Make error messages actionable
5. LLM should correct behavior when given clear feedback

**Next Steps**:
- Run full L5-L7 evaluation to capture natural occurrences
- Monitor logs for "ERROR: Your last response had a malformed tool call"
- Verify LLM corrects itself in subsequent rounds
- Measure reduction in empty rounds

## Expected Impact

### Metrics to Track

1. **Empty rounds** (before fix):
   - Avg consecutive: ~1-2 per occurrence
   - Max observed: 2
   - Caused by: Malformed tool calls

2. **Empty rounds** (after fix):
   - Expected: 0-1 per occurrence
   - LLM should self-correct immediately
   - No multi-round loops

3. **Success rate**:
   - Before: 100% (3/3) but small sample
   - After: Should maintain or improve
   - Fewer retries needed

4. **Time per task**:
   - Before: 41-512s (wide variance)
   - After: Should be more consistent
   - Fewer wasted rounds

## Comparison to Empty Round Recovery

**Empty Round Recovery** (LoopDetectionBehavior):
- Detects AFTER 3 consecutive empty rounds
- Generic recovery prompt
- Doesn't tell LLM what it did wrong
- Reactive approach

**Tool Call Error Feedback** (base_agent.py):
- Detects IMMEDIATELY on first error
- Specific feedback about the exact mistake
- Shows LLM what it generated wrong
- Proactive approach

**They work together**:
1. Tool call error feedback handles MOST cases (immediate correction)
2. Empty round recovery handles EDGE cases (if LLM still stuck after 3 rounds)
3. Defense in depth

## Conclusion

This fix implements immediate, actionable error feedback at the **single point** where all LLM errors are caught. Instead of failing silently, the agent now tells the LLM exactly what went wrong and how to fix it.

**Status**: ✅ **Implemented and ready for testing**

**Location**: `/workspace/base_agent.py` lines 337-379

**Impact**: Should eliminate most empty rounds caused by malformed tool calls

**Next**: Run full evaluation to measure effectiveness
