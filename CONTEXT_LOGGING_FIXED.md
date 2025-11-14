# Context Logging - FIXED

## Problem Summary

User requested: "Fix context inspection to log in workspaces with proper persistent detailed logs that don't overwrite themselves so we can properly diagnose agent mistakes."

The `on_llm_response` hook was configured but **not capturing LLM responses** - post-LLM snapshots only contained metadata (4 keys), no actual response data.

## Root Cause Analysis

Through thorough debugging with stack traces, I discovered **three critical bugs**:

### Bug 1: Duplicate Behavior Calling

The behavior system was calling `on_llm_response` **TWICE per round** from two places:

1. **base_agent.py:535-537** (legacy code):
   ```python
   for behavior in self.behaviors:
       if hasattr(behavior, 'on_llm_response'):
           response = behavior.on_llm_response(self, response)
   ```

2. **agent_lifecycle.py:315** (current architecture):
   ```python
   self.agent.event_system.trigger_llm_response(response["message"])
   ```

**Evidence**: Debug logging showed:
```
[DEBUG] on_llm_response called from /workspace/base_agent.py:537
[DEBUG] on_llm_response called from /workspace/src/agent_events.py:165
```

### Bug 2: Passing Wrong Data Structure

`agent_lifecycle.py:315` was passing **`response["message"]`** instead of the full response:

```python
# WRONG - passes just the message dict
self.agent.event_system.trigger_llm_response(response["message"])
```

This caused the second call to receive:
```python
# Second call got THIS (missing 'message' key):
{'role': 'assistant', 'content': '...', 'thinking': '...', 'tool_calls': [...]}

# Instead of THIS (full response):
{
  'message': {'role': 'assistant', 'content': '...', 'tool_calls': [...]},
  'prompt_eval_count': 123,
  'eval_count': 456,
  ...
}
```

### Bug 3: Event System Not Returning Modified Response

`agent_events.py` had return type `-> None` instead of `-> dict`:

```python
def trigger_llm_response(self, response: dict[str, Any]) -> None:  # WRONG
    for behavior in self.agent._behaviors:
        behavior.on_llm_response(agent=self.agent, response=response)  # Not captured
    # No return statement
```

This broke the behavior chain - behaviors that modify responses (like `ToolCallingSyntaxBehavior`) were having their changes ignored.

### Bug 4: File Overwriting

Two methods saved to the same filename:

1. **`on_llm_response`**: Saved immediately after LLM with full response
2. **`on_round_end`**: Saved after tools execute, overwrote file with minimal data

Both wrote to `{agent}_round_{N:03d}_post_llm.json`, causing the good data to be overwritten.

## Fixes Applied

### Fix 1: Remove Duplicate Calling

**File**: `base_agent.py:534-536`

```python
# BEFORE:
# Allow behaviors to post-process response (e.g., parse tool calls from XML)
for behavior in self.behaviors:
    if hasattr(behavior, 'on_llm_response'):
        response = behavior.on_llm_response(self, response)

# AFTER:
# NOTE: Behaviors are called via agent_lifecycle.py event system
# This method is only used by debug scripts
```

### Fix 2: Pass Full Response

**File**: `agent_lifecycle.py:313-315`

```python
# BEFORE:
if "message" in response:
    self.agent.event_system.trigger_llm_response(response["message"])

# AFTER:
if "message" in response:
    response = self.agent.event_system.trigger_llm_response(response)
```

### Fix 3: Return Modified Response

**File**: `agent_events.py:153-171`

```python
# BEFORE:
def trigger_llm_response(self, response: dict[str, Any]) -> None:
    for behavior in self.agent._behaviors:
        behavior.on_llm_response(agent=self.agent, response=response)
    # No return

# AFTER:
def trigger_llm_response(self, response: dict[str, Any]) -> dict[str, Any]:
    for behavior in self.agent._behaviors:
        response = behavior.on_llm_response(agent=self.agent, response=response)
    return response
```

### Fix 4: Separate Filenames

**File**: `behaviors/context_inspector.py`

```python
# on_llm_response saves to:
{agent}_round_{N:03d}_post_llm_immediate.json

# on_round_end saves to:
{agent}_round_{N:03d}_round_end.json
```

## Verification

Created test workspace with full LLM response capture:

```bash
$ cat /tmp/test_final_fix/.agent_context/context_snapshots/orchestrator_round_001_post_llm_immediate.json
{
  "agent_name": "orchestrator",
  "round": 1,
  "phase": "post_llm",
  "timestamp": 1763080331.3364496,
  "llm_response": {
    "content": "{\"name\": \"delegate_to_executor\", ...}",
    "content_length": 196,
    "tool_calls": [
      {
        "function": {
          "name": "delegate_to_executor",
          "arguments": {
            "task_description": "Create a math.py file with...",
            "workspace_mode": "existing"
          }
        }
      }
    ],
    "tool_call_count": 1,
    "is_empty": false
  }
}
```

✅ **Full LLM response captured**
✅ **Content preserved**
✅ **Tool calls preserved**
✅ **Empty status correct**
✅ **Files don't overwrite each other**

## What This Enables

Now we can properly investigate "empty rounds" from previous evaluation failures:

### For Each Workspace:

```bash
# Check what LLM actually returned (not inferred from context)
cat workspace/.agent_context/context_snapshots/orchestrator_round_003_post_llm_immediate.json

# See if response was:
# 1. Truly empty (no content, no tool_calls)
# 2. Malformed tool call (XML instead of JSON)
# 3. Failed parsing (content exists but tools not extracted)
# 4. Correct tool call but wrong arguments
```

### Analysis Questions Now Answerable:

1. **Were "empty rounds" truly empty?**
   - Check `llm_response.is_empty` field

2. **Was the agent thinking?**
   - Check `llm_response.content` for reasoning text

3. **Were there failed tool calls?**
   - Check `llm_response.content` for XML/JSON patterns
   - Compare to `llm_response.tool_calls` to see if parsing failed

4. **Did system properly nudge?**
   - Check `*_pre_llm.json` for "EMPTY ROUNDS" warnings
   - Check `*_pre_llm.json` for "You must call tools" messages

5. **What format did LLM use?**
   - Check `llm_response.content` for JSON vs XML vs plain text

## Next Steps

1. **Re-run failed evaluation task** (config_loader empty workspace)
2. **Examine actual LLM responses** from post_llm_immediate.json
3. **Determine root cause** of empty workspace:
   - LLM didn't call tools? → Context issue
   - LLM called wrong tools? → Prompt issue
   - LLM called tools with wrong format? → Syntax issue
   - Tools failed silently? → Dispatch issue

## Commit

```
commit cc64f72
fix(context_inspector): Fix LLM response capture in workspace snapshots

- Remove duplicate behavior calling from base_agent.py
- Fix agent_lifecycle.py to pass full response
- Fix agent_events.py to return modified response
- Separate post_llm_immediate and round_end filenames
```

## Status: ✅ COMPLETE

Context logging is now fully functional and ready for deep investigation of agent behavior.
