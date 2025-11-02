# Empty Rounds Investigation - LLM Stops Calling Tools

**Date**: 2025-11-02
**Issue**: TaskExecutor getting stuck in loops where LLM produces responses but no tool calls
**Duration Impact**: L5 Web API Run 3 took 40 minutes due to multiple failed attempts

## Summary

After fixing delegation architecture (execution, workspace coordination, summary extraction), we discovered a secondary issue: **LLM intermittently stops calling tools** mid-task, causing agents to loop for 50 rounds before failing and requiring retry.

## Investigation Results

### What We Know

1. **Intermittent Behavior**: Empty rounds don't happen every time
   - Some runs complete successfully in 10-20 rounds
   - Other runs get stuck after 3-4 tool calls and loop for 50 rounds
   - Same task description, different outcomes (temperature/randomness)

2. **Pattern When It Occurs**:
   ```
   Round 1-4: LLM calls tools normally (write_file, list_dir, run_bash)
   Round 5-50: LLM produces responses but NO tool_calls
   Result: Max rounds reached → FAILURE
   Orchestrator: Retries delegation with adjusted prompt
   Eventually: One retry succeeds
   ```

3. **Context State**:
   - System prompt regenerated every round (includes tool docs) ✓
   - Tools always available via `get_system_prompt()` + `generate_tool_documentation()` ✓
   - Context compaction preserves system prompt ✓
   - No evidence of tools being lost

4. **What Gets Created**:
   - Despite multiple failures, task eventually completes
   - Files created: `app.py` (1626 bytes), `test_api.py` (2053 bytes)
   - Code quality is good - proper Flask API with comprehensive tests
   - Orchestrator retry mechanism works

### What We Don't Know (Couldn't Capture)

1. **LLM Output Content**: What is the LLM actually saying in empty rounds?
   - Is it producing text explanations without tools?
   - Is it confused about what to do next?
   - Is it silently failing to understand tool schemas?

2. **Context at Stuck Point**: What does context look like when LLM gets stuck?
   - Is there a specific pattern in message history?
   - Does specific tool result content trigger confusion?
   - Are there any errors in tool results?

3. **Why Retries Work**: Why does same task succeed after retry?
   - Different prompt phrasing breaks the stuck state?
   - Random seed variation in LLM?
   - Context reset helps?

## Hypotheses

### ✅ Likely: Hypothesis 1 - Context Confusion (LLM Capability)

**Evidence:**
- Intermittent behavior suggests LLM decision-making issue
- Same task succeeds sometimes, fails other times
- After initial success (3-4 tool calls), LLM loses direction
- Model (gpt-oss:20b) may struggle with complex multi-step tasks

**Mechanism:**
```
1. LLM starts well: understands task, calls appropriate tools
2. After 3-4 tools: context has tool results, partial progress
3. LLM analyzes situation: "what should I do next?"
4. LLM gets confused/uncertain
5. Instead of calling mark_failed, produces text response (no tools)
6. Agent loops hoping for tool call that never comes
```

### ❌ Unlikely: Hypothesis 3 - Tool Schema Issues

**Evidence Against:**
- System prompt regenerated every round
- Tools always present via `get_system_prompt()`
- Context compaction preserves system prompt
- Debug runs show tools working fine initially

**Why It Seemed Plausible:**
- Context compaction could lose information
- But tool schemas are in system prompt (not messages)
- System prompt never gets compacted

## Current Agent Behavior (Problem)

**When LLM produces no tool calls** (base_agent.py:1093-1094):
```python
if "tool_calls" in msg and msg["tool_calls"]:
    # Execute tools
    for tool_call in tool_calls:
        ...
else:
    # NO HANDLER - just continues silently
    pass  # implicit - loops to next round
```

**Missing:**
- No detection of empty rounds
- No recovery mechanism
- No feedback to LLM about the problem
- Just loops silently for 50 rounds

## Impact Analysis

### Time Breakdown for 40-Minute Run

```
Attempt 1: 3 tools → stuck → 50 rounds → FAIL    (~25s)
Attempt 2: 4 tools → stuck → 50 rounds → FAIL    (~25s)
Attempt 3: 2 tools → stuck → 50 rounds → FAIL    (~25s)
Attempt 4: Similar pattern → FAIL                (~25s)
...
(Multiple retries with different prompts)
...
Attempt N: 10 tools → mark_complete → SUCCESS    (~10s)

Total: ~40 minutes
```

**Orchestrator overhead:**
- Each failed delegation → prompt adjustment → new delegation
- Orchestrator rounds: ~3-5 seconds each
- Total orchestrator time: ~5-10 minutes
- Total task_executor time: ~30-35 minutes (mostly stuck loops)

## Recommendations

### 1. Detect Empty Rounds (HIGH PRIORITY)

Add tracking for consecutive rounds without tool calls:

```python
# In base_agent.py run() method
consecutive_empty_rounds = 0

for round_no in range(1, max_rounds + 1):
    response = self.call_llm(...)

    if "message" in response:
        msg = response["message"]

        if not msg.get("tool_calls"):
            consecutive_empty_rounds += 1
            if consecutive_empty_rounds >= 3:
                print(f"[{self.name}] ⚠️  LLM stuck - {consecutive_empty_rounds} rounds without tool calls")
                # Trigger intervention
        else:
            consecutive_empty_rounds = 0  # Reset on successful tool call
```

### 2. Add Recovery Mechanism (HIGH PRIORITY)

When stuck, inject a recovery message to help LLM:

```python
if consecutive_empty_rounds >= 3:
    recovery_msg = {
        "role": "user",
        "content": """You haven't called any tools in 3 consecutive rounds.

IMPORTANT: You MUST call a tool to make progress:
- If you know what to do next: call the appropriate tool
- If you're stuck or unsure: call mark_failed(reason="explanation")
- DO NOT respond without calling a tool

Available tools: write_file, read_file, list_dir, run_bash, mark_complete, mark_failed

What tool will you call next?"""
    }
    self.add_message(recovery_msg)
    consecutive_empty_rounds = 0  # Reset after intervention
    continue  # Give LLM another chance
```

### 3. Lower max_rounds for Subagents (MEDIUM PRIORITY)

**Current:** 50 rounds per task_executor delegation
**Problem:** 25+ seconds to fail when stuck

**Recommendation:**
```python
# In behaviors/delegation.py
execution_result = target_agent.run(max_rounds=25)  # Was: 50
```

**Benefits:**
- Faster failure detection (~12 seconds instead of 25)
- More retry opportunities in same time
- Orchestrator can try different approaches faster

### 4. Enforce Tool Requirement in System Prompt (LOW PRIORITY)

Update task_executor system prompt:

```
CRITICAL RULE: You MUST call at least one tool in every round.

If making progress: Call the next appropriate tool
If stuck/uncertain: Call mark_failed(reason="...") to explain
If complete: Call mark_complete(summary="...")

NEVER produce a response without calling a tool.
```

### 5. Add Diagnostic Logging (LOW PRIORITY)

Log message content when no tool calls:

```python
if not msg.get("tool_calls"):
    content_preview = msg.get("content", "")[:200]
    print(f"[DEBUG] Empty round - LLM said: {content_preview}...")
```

This helps diagnose what LLM is thinking when stuck.

### 6. Consider Circuit Breaker (FUTURE)

After N failed delegations, escalate or give up:

```python
# In delegation behavior
failed_attempts = 0
MAX_DELEGATION_RETRIES = 5

if delegation_failed:
    failed_attempts += 1
    if failed_attempts >= MAX_DELEGATION_RETRIES:
        return {
            "success": False,
            "error": f"Task failed after {failed_attempts} delegation attempts"
        }
```

## Implementation Priority

1. **Immediate (fixes stuck loops)**:
   - Empty round detection
   - Recovery prompts

2. **Short-term (improves performance)**:
   - Lower max_rounds to 25
   - Add diagnostic logging

3. **Future (nice to have)**:
   - Tool requirement enforcement
   - Circuit breaker for repeated failures

## Expected Improvement

**Before Fix:**
- Stuck attempts: 50 rounds × 0.5s = 25 seconds each
- 4 failed attempts = 100+ seconds wasted
- Total task time: 40 minutes

**After Fix (Empty Round Detection + Recovery):**
- Stuck detection: 3 rounds × 0.5s = 1.5 seconds
- Recovery prompt: 1-2 additional rounds
- If recovery works: ~2 seconds lost instead of 25
- If recovery fails: mark_failed after 5 rounds instead of 50
- Expected time savings: 80-90% reduction in stuck time

**After Fix (+ Lower max_rounds):**
- Failed attempts: 25 rounds × 0.5s = 12 seconds each
- Combined with recovery: ~3-5 seconds per stuck attempt
- Total task time estimate: 5-10 minutes (vs 40 minutes)

## Conclusion

The delegation architecture is **functionally correct** (all context management, workspace coordination, and summary extraction work properly).

The performance issue is **LLM capability** - the model intermittently gets stuck and stops calling tools, but the agent doesn't detect or recover from this state.

**Solution:** Add empty round detection + recovery prompts to help LLM get unstuck, and lower max_rounds to fail faster when recovery doesn't work.

**Expected impact:** 80%+ reduction in time spent on stuck loops, bringing 40-minute tasks down to 5-10 minutes.
