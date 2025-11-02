# L5 Web API Run 3 - 40 Minute Analysis

**Date**: 2025-11-02
**Duration**: ~40 minutes
**Status**: Eventually succeeded after multiple retries

## Timeline

1. **Orchestrator Round 1**: Delegates to task_executor
2. **Task_executor attempt 1**:
   - Rounds 1-4: Called tools (list_dir, list_dir, write_file, write_file)
   - **Rounds 5-50: NO TOOL CALLS** - Agent stuck in empty loop
   - Result: Max rounds reached, FAILURE
3. **Orchestrator Round 2-5**: Multiple retries
   - Each retry: task_executor hits max rounds (50) without completing
   - Pattern repeats: 3-4 tool calls in early rounds, then stuck
4. **Task_executor eventual success**: One attempt finally called `mark_complete`
5. **Orchestrator marks complete**: Goal achieved after ~40 minutes

## Root Cause: LLM Not Calling Tools

**Problem**: TaskExecutor gets stuck in rounds where it produces responses but NO tool calls.

**Evidence**:
```
[task_executor] Round 1/50
[task_executor] Executing 1 tool call(s)
[task_executor] -> write_file

[task_executor] Round 2/50
[task_executor] Executing 1 tool call(s)
[task_executor] -> write_file

[task_executor] Round 3/50

[task_executor] Round 4/50

[task_executor] Round 5/50
...
[task_executor] Round 50/50
[task_executor] Max rounds (50) reached without completion
```

**What's happening**:
- LLM responds successfully (no timeout)
- Response contains NO tool_calls
- Agent loops to next round
- Continues for 45+ rounds until max_rounds

## Why LLM Stops Calling Tools

**Hypothesis 1: Context Confusion**
- After a few tool calls, context becomes unclear
- LLM doesn't know what to do next
- Produces text response but no tools

**Hypothesis 2: Model Capability**
- `gpt-oss:20b` may struggle with complex Flask API task
- Model "gives up" but doesn't call mark_failed
- Just produces empty/text responses

**Hypothesis 3: Tool Schema Issues**
- LLM may not understand tool schemas after context compaction
- Tool documentation gets lost or corrupted
- LLM can't figure out which tool to call

## Current Agent Behavior

**When LLM produces no tool calls** (base_agent.py:1093-1094):
```python
if "tool_calls" in msg and msg["tool_calls"]:
    # Execute tools
else:
    # NO HANDLER - just continues to next round
    pass  # implicit
```

**Missing**: No detection or recovery when LLM stops calling tools.

## Why Orchestrator Eventually Succeeds

**Retry mechanism works**:
1. Orchestrator sees delegation failed
2. Retries with adjusted task description
3. Eventually one retry succeeds (luck/simpler prompt)
4. Orchestrator marks complete

**But inefficient**:
- 3-4 failed attempts × 50 rounds × 0.5s/round = ~100 seconds per failure
- Multiple failures = 40 minutes total

## Files Created

Despite the struggles, files WERE eventually created:
```
.agent_workspaces/rerun_l5_web_api_run3/
├── app.py (1626 bytes)
├── models.py (165 bytes)
├── tests/test_api.py (2053 bytes)
└── jetboxnotes.md
```

The task WAS completed, just inefficiently.

## Recommendations

### 1. Detect Empty Rounds
Add detection when LLM produces multiple rounds without tool calls:

```python
consecutive_empty_rounds = 0
for round_no in range(1, max_rounds + 1):
    response = self.call_llm(...)

    if "tool_calls" not in msg or not msg["tool_calls"]:
        consecutive_empty_rounds += 1
        if consecutive_empty_rounds >= 3:
            print(f"[{self.name}] LLM stuck - 3 rounds without tool calls")
            # Trigger intervention
    else:
        consecutive_empty_rounds = 0
```

### 2. Add Recovery Prompts
When stuck, inject a recovery message:

```python
if consecutive_empty_rounds >= 3:
    recovery_msg = {
        "role": "user",
        "content": "You haven't called any tools in 3 rounds. Please call a tool to make progress, or call mark_failed if you're stuck."
    }
    self.add_message(recovery_msg)
    consecutive_empty_rounds = 0  # Reset after intervention
```

### 3. Lower max_rounds for Delegation
Current: 50 rounds per task_executor
Problem: Takes 25+ seconds to fail

Suggestion:
- TaskExecutor max_rounds = 25 (faster failure)
- Orchestrator can retry more frequently
- Same total time, more retry opportunities

### 4. Add Tool Call Requirement
System prompt should emphasize:
```
CRITICAL: You MUST call tools in every round.
- If making progress: call appropriate tools
- If stuck: call mark_failed with explanation
- NEVER respond without calling a tool
```

### 5. Better Context Management
After initial tool calls, context may become unclear.
- Keep tool documentation visible
- Avoid compacting tool schemas
- Re-inject tool reminders periodically

## Conclusion

**Delegation works correctly** (execution, workspace coordination, summary extraction all functional).

**Problem is LLM capability** - Model gets stuck in loops where it doesn't know what tool to call next.

**Solution needed**: Detection and recovery when LLM stops calling tools, rather than silently looping for 50 rounds.
