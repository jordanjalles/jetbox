# TaskExecutor Idle Loop Test Results

**Date:** 2025-10-31
**Fix Applied:** Reduced inactivity timeout from 30s to 15s
**Test Method:** Orchestrator delegation to TaskExecutor with SubAgentStrategy

## Test Results Summary

| Task | Complexity | Rounds | Duration | Context (tokens) | Status | mark_complete Called? |
|------|-----------|--------|----------|------------------|--------|---------------------|
| L5 CLI Calculator | Simple | 3 | ~14s | 6,188 | ✅ SUCCESS | ✅ Yes |
| L6 REST API Client | Medium | 31+ | 2m 49s+ | 176,151 | ❌ IDLE LOOP | ❌ No |

## Detailed Findings

### L5 Simple Task - SUCCESS ✅

**Goal:** "Create a CLI calculator with history, including Calculator class, evaluate, history, clear_history, support +,-,*,/, parentheses, sqrt, abs"

**Results:**
- Completed in **3 rounds**
- Duration: ~14 seconds
- Context: 6,188 tokens
- **Agent properly called mark_complete** with summary
- Files created successfully:
  - `calculator.py` - Full implementation with Calculator class
  - `jetboxnotes.md` - Summary generated
  - `agent_ledger.log` - Action log

**Completion Message:**
```
✓ DELEGATED TASK COMPLETE
Summary: Implemented Calculator class with safe expression evaluation, history tracking,
clear_history, and a CLI supporting history, clear, exit commands. Added comprehensive
documentation and example usage.
```

**Analysis:** For simple, well-defined tasks, the agent recognizes completion naturally and calls mark_complete without issues.

---

### L6 Medium Task - IDLE LOOP ❌

**Goal:** "Create REST API client library with APIClient class supporting get/post methods, AuthHandler for token management, rate limiting at 5 requests per second, and retry logic with exponential backoff"

**Results (at Round 31, still running):**
- Running for **31+ rounds**
- Duration: **2m 49s+** (approaching 3min timeout)
- Context: **176,151 tokens** (exceeds 128K limit by 37%)
- **Agent has NOT called mark_complete or mark_failed**
- Executing 1 action per round (not truly idle)
- LLM call time increasing: 2.56s → 5.47s (context growth impact)

**Behavior Pattern:**
```
Round 1-10:  Creating APIClient class, AuthHandler
Round 11-20: Adding rate limiting, retry logic, tests
Round 21-30: Refining implementation, adding edge cases
Round 31+:   Still working, no completion signal
```

**Analysis:**
The agent is actively working (not timing out) but doesn't recognize when it has done "enough" work. The LLM continues iterating, adding features, refining code, but never calls mark_complete. This is the core idle loop bug.

**Root Cause:** For complex tasks with multiple requirements, the LLM doesn't have clear criteria for "completion". It keeps trying to perfect the implementation rather than reporting reasonable progress.

---

## Key Insights

###  1. Simple vs Complex Task Behavior

**Simple tasks (L5):**
- Clear, focused goal
- LLM recognizes completion naturally
- mark_complete called within 3-5 rounds
- ✅ Works correctly

**Complex tasks (L6):**
- Multiple requirements (5+ features)
- LLM unclear when "good enough"
- Never calls mark_complete
- Context grows unbounded
- ❌ Idle loop bug manifests

### 2. Inactivity Timeout Impact

**Finding:** Reducing inactivity timeout from 30s to 15s did NOT fix the idle loop.

**Why:** The LLM is not timing out - it's actively responding and making progress. The issue is it never decides to call the completion tools (mark_complete/mark_failed).

**Conclusion:** Inactivity timeout is useful for detecting hung LLMs, but doesn't solve the completion decision problem.

### 3. Context Growth Pattern

L6 context growth exceeded the 128K token limit:
- Round 10: 26,776 tokens
- Round 20: 93,862 tokens
- Round 30: 176,151 tokens

SubAgentStrategy's compaction threshold (75% of 128K = 96K) was exceeded by Round 21, but compaction didn't prevent continued growth. The agent should have been nudged to complete before reaching this point.

### 4. Status Display Misleading

The "💤 idle" status is misleading - the agent is NOT idle. It's executing 1 action per round. The "idle" status just means no hierarchical subtasks are active (SubAgentStrategy doesn't use task decomposition).

**Recommendation:** Update status display for SubAgentStrategy to show "working" instead of "idle".

---

## Recommendations

Based on these test results, the idle loop bug requires a **proactive completion nudge** for complex tasks:

### Option 1: Completion Nudge at Context Threshold ⭐ RECOMMENDED

When context exceeds 60% of limit (~76K tokens for 128K model):
```python
if estimated_tokens > max_tokens * 0.6 and not has_called_completion_tools:
    inject_message: "You've been working for a while. If you've made meaningful
    progress on the core requirements, call mark_complete(summary) now. Don't
    wait for perfection - report reasonable progress."
```

**Pros:**
- Catches stuck agents before context explodes
- Natural trigger point (agent has done substantial work)
- Single nudge, not repetitive

**Cons:**
- May trigger too early for very complex tasks
- Threshold may need tuning per model

### Option 2: Round-Based Nudge

Every N rounds (e.g., N=15):
```python
if round_no % 15 == 0 and not has_called_completion_tools:
    inject_message: "Progress check - have you met the core requirements? ..."
```

**Pros:**
- Simple to implement
- Predictable trigger

**Cons:**
- May be too aggressive (nudges even when agent is close to natural completion)
- Fixed interval doesn't adapt to task complexity

### Option 3: Acceptance Criteria in Delegation

Orchestrator explicitly states minimum criteria:
```
Task: Create REST API client with APIClient class, AuthHandler, rate limiting, retry logic

Minimum Acceptance Criteria:
- APIClient class with get() and post() methods exists
- AuthHandler class exists
- Basic rate limiting present
- Basic retry logic present

Call mark_complete() when these 4 criteria are met. Don't over-engineer.
```

**Pros:**
- Clear success criteria upfront
- LLM can self-assess

**Cons:**
- Requires orchestrator to generate criteria
- Still relies on LLM judgment

---

## Next Steps

1. ✅ **Completed:** Reduce inactivity timeout to 15s (helps with hung LLMs, but doesn't fix idle loop)
2. ⬜ **TODO:** Implement Option 1 (context-threshold nudge)
3. ⬜ **TODO:** Retest L6 with nudge implemented
4. ⬜ **TODO:** Test L8 (with architect) to validate full integration

---

## Test Files

- L5 Implementation: `./implement-a-cli-calculator-with-a-calculator-class/`
- L6 (in progress): TBD after timeout
- Logs:
  - `/tmp/l5_orch_test.log` - L5 successful completion
  - `/tmp/l6_test.log` - L6 idle loop demonstration
