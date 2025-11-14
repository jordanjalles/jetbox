# Context Logging Improvements - Status Report

## What Was Requested

You asked for:
1. **Deeper investigation of "empty rounds"**
   - Were they truly empty or was the agent thinking?
   - Were there failed tool calls or wrong format?
   - Did the system properly nudge on empty rounds?

2. **Fix context inspection to log in workspaces**
   - Proper persistent detailed logs
   - Don't overwrite between tasks
   - Enable proper diagnosis of agent mistakes

## What Was Implemented

### ✅ Workspace-Relative Logging

**Before:**
- Context snapshots saved to `.context_inspection/` (global directory)
- Each task run overwrote previous files
- Investigating old runs was impossible

**After:**
```
<workspace>/.agent_context/context_snapshots/
  ├── orchestrator_round_000_initial.json
  ├── orchestrator_round_001_pre_llm.json
  ├── orchestrator_round_001_post_llm.json
  ├── task_executor_round_000_initial.json
  ...
```

**Benefits:**
- Each task run has its own persistent logs
- Never overwritten (tied to workspace)
- Can investigate failures after the fact
- Sub-agents (task_executor, architect) have separate logs

### ✅ Added on_llm_response Hook

Attempted to capture LLM responses immediately after they return:
- Captures content, tool_calls, empty status
- Runs after ToolCallingSyntaxBehavior parses responses
- Should show exactly what LLM returned

### ⚠️ LLM Response Capture Not Working

**Issue:** Post-LLM snapshots only contain metadata:
```json
{
  "agent_name": "orchestrator",
  "round": 1,
  "phase": "post_llm",
  "timestamp": 1763078635.6362162
}
```

Missing: `llm_response` field with actual response data

**Why:** The `on_llm_response` hook is being called but the snapshot doesn't include the response. Possible causes:
1. Exception in serialization (silent try/except)
2. Response structure different than expected
3. Timing issue (response not fully processed yet)

## What We Can Do Now

### ✅ Investigate Old Failures

Can now examine actual workspaces from failed tasks:
```bash
ls /tmp/orch_L4_config_loader_j8z905k0/.agent_context/context_snapshots/
cat .../orchestrator_round_001_pre_llm.json
```

Each round's pre-LLM context shows:
- Full message history
- Tools available
- Nudges and warnings
- System prompts

### ⚠️ Cannot Yet See LLM Responses

Until post-LLM capture is fixed, we can't see:
- What the LLM actually returned
- Whether it was empty or wrong format
- What tool calls it made (or didn't make)

## Next Steps to Fix

### 1. Debug on_llm_response Capture

Add explicit logging to see what's happening:
```python
def on_llm_response(self, agent, response):
    print(f"[DEBUG] Response keys: {response.keys()}")
    print(f"[DEBUG] Message keys: {response.get('message', {}).keys()}")
    # ... rest of capture logic
```

### 2. Simplify Serialization

Start with minimal capture:
```python
snapshot["llm_response"] = {
    "raw_response": str(response)[:500],  # Just dump it as string first
    "has_message": "message" in response,
    "has_tool_calls": "message" in response and "tool_calls" in response.get("message", {})
}
```

### 3. Alternative: Log to Messages

Instead of separate snapshot, add to pre-LLM file:
- Append post-LLM response to next round's pre-LLM context
- Shows full request-response cycle

## How to Use Current Implementation

### Check Workspace Context

```bash
WORKSPACE="/tmp/orch_L4_config_loader_j8z905k0"

# See all rounds
ls $WORKSPACE/.agent_context/context_snapshots/

# Check round 3 (where delegation happened)
cat $WORKSPACE/.agent_context/context_snapshots/orchestrator_round_003_pre_llm.json | jq .

# See what warnings/nudges were shown
cat .../*_pre_llm.json | jq '.context[] | select(.content | contains("EMPTY ROUNDS"))'
```

### Analyze "Empty Rounds"

From pre-LLM context:
1. Check for "CRITICAL: N CONSECUTIVE EMPTY ROUNDS" messages
2. See what nudges were shown ("You must call tools")
3. Check tool definitions (were right tools available?)
4. Look at message history (what did LLM see?)

## Commits

- `a2efe5a` - Workspace-relative logging + on_llm_response hook
- `d19902f` - Current state (response capture not working)

## Conclusion

**Progress: 50%**

✅ Fixed the logging location problem (workspace-relative)
✅ Made logs persistent (no more overwrites)
❌ LLM response capture needs debugging

**Can now answer:**
- What context was sent to LLM each round?
- What warnings/nudges were shown?
- What tools were available?

**Cannot yet answer:**
- What did LLM actually respond with?
- Was the response empty or malformed?
- What tool calls did it make?

**Recommendation:**
1. Debug on_llm_response with print statements
2. Or examine message history from pre-LLM snapshots (next round shows previous response)
3. Use this to investigate config_loader empty workspace case
