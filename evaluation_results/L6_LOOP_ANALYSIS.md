# L6 Loop Analysis - Actual Behavior

## What Actually Happened

The L6 agent was **NOT idle** and **NOT in a completion decision loop**. It was stuck in a **failure recovery loop** trying to fix pytest import errors.

### Timeline from agent_ledger.log:

**Rounds 1-4: Good Progress ✅**
```
1. WRITE api_client/__init__.py
2. WRITE api_client/auth.py
3. WRITE api_client/client.py (with rate limiting, retry logic, AuthHandler)
4. WRITE tests/test_api_client.py
```

**Round 5: First pytest attempt ❌**
```
BASH pytest -q -> rc=2 (FAILED)
ERROR run_bash rc=2
```

**Rounds 6-31: STUCK IN LOOP trying to fix pytest** 🔁
```
Repeated ~15-20 times:
  - WRITE sitecustomize.py (trying to fix Python path)
  - BASH pytest -q -> rc=2 (STILL FAILS)
  - ERROR run_bash rc=2
  - WRITE sitecustomize.py (rewrite same file with slight variation)
  - BASH pytest -q -> rc=2 (STILL FAILS)
  - ...
```

### The Loop Pattern

The agent:
1. ✅ Created excellent implementation code (client.py has rate limiting, exponential backoff, etc.)
2. ❌ Tests failed due to import/path issues
3. 🔁 Got stuck trying the SAME solution repeatedly (rewriting sitecustomize.py)
4. ❌ Never recognized "this isn't working, try something different"
5. ❌ Never called mark_complete() even though core requirements were met
6. ⏱️ Eventually hit 3-minute timeout

### Files Created (ALL REQUIREMENTS MET!)

**api_client/client.py (62 lines):**
- ✅ APIClient class
- ✅ get() and post() methods
- ✅ Rate limiting (5 req/sec with timing logic)
- ✅ Exponential backoff retry (backoff_factor * 2^attempt)
- ✅ AuthHandler integration

**api_client/auth.py (52 lines):**
- ✅ AuthHandler class
- ✅ Token management

**tests/test_api_client.py (71 lines):**
- ✅ Tests for API client

**The task was FUNCTIONALLY COMPLETE** - the agent just couldn't get pytest to run due to import path issues.

## Root Cause Analysis

This is NOT an "idle loop" or "doesn't know when to mark complete" bug.

This is a **"can't accept failure and move on"** bug.

The agent:
1. Has no concept of "I've tried this 5 times, time to try something different"
2. Doesn't recognize when a fix attempt pattern is repeating
3. Can't decide "tests failing but code is complete enough - mark_complete anyway"

## Loop Detection

The codebase HAS loop detection (`loop_counts` in context_manager), but:
1. It may not have triggered because writes to sitecustomize.py had slightly different content each time
2. Even if detected, the agent doesn't have a mechanism to BREAK OUT of the loop

From the status display, we saw:
```
Round 31: Actions executed: 31, Tokens: 176K
```

But no "LOOP DETECTION WARNING" appeared in the log, suggesting the loop detector didn't catch this pattern.

## What Should Have Happened

**Option A: Give up on tests**
```
After 3-5 pytest failures:
"Tests are failing due to environment issues, but core implementation is complete.
Calling mark_complete() to report finished implementation."
```

**Option B: Try different approach**
```
After 3 sitecustomize.py rewrites:
"sitecustomize.py approach isn't working. Let me try installing as editable package
with `pip install -e .` instead."
```

**Option C: Ask for help** (not available in current tools)
```
"I've created APIClient with all required features but pytest won't run.
Implementation is complete - should I mark complete anyway?"
```

## Context Compaction

**Question:** Did context compaction trigger?

**Answer:** No evidence in the logs. SubAgentStrategy threshold is 75% of 128K = 96K tokens.

At Round 21: 93,862 tokens (just under threshold)
At Round 31: 176,151 tokens (way over limit!)

**Conclusion:** Either:
1. Compaction didn't trigger (bug in compaction logic)
2. Compaction triggered but didn't help (context still grew)
3. Token estimation is wrong (actual context < estimated)

Need to grep for "context_compaction" in logs to confirm.

## Token Estimation Issues

The estimated tokens grew linearly:
- Round 10: 28,696 tokens
- Round 20: 93,862 tokens (~3.3x)
- Round 30: 176,151 tokens (~6x)

This is suspicious - with only 1 action per round and messages being tool results, context shouldn't grow THIS fast. Possible issues:
1. Token estimation formula is wrong
2. Messages aren't being deduplicated
3. Tool results contain huge strings (pytest error output?)

## Recommendations

### Fix #1: Failure Pattern Detection ⭐ HIGH PRIORITY

When the same tool+args fails 3 times in a row:
```python
if action_signature in recent_failures and count >= 3:
    inject_message: "This approach has failed 3 times. Either:
    1. Try a COMPLETELY DIFFERENT approach
    2. If task is substantially complete, call mark_complete()
    3. If truly blocked, call mark_failed(reason)"
```

### Fix #2: "Good Enough" Completion Criteria

For complex tasks, define "minimum viable completion":
```
Task: REST API client with get/post, auth, rate limiting, retry

Minimum criteria:
- ✅ APIClient class exists
- ✅ get() and post() methods exist
- ✅ AuthHandler exists
- ✅ Rate limiting logic present
- ✅ Retry logic present
- ⚠️ Tests exist but may not run

If criteria met, call mark_complete() even if tests fail.
```

### Fix #3: Loop Detector Enhancement

Current loop detector tracks action signatures. Enhance to:
- Track failure patterns (same command failing repeatedly)
- Track file rewrite patterns (same file edited repeatedly)
- Surface warnings more aggressively

### Fix #4: Context Compaction Debugging

Add logging to confirm when/if compaction triggers:
```python
if estimated_tokens > threshold:
    print(f"[context_compaction] Triggered at {estimated_tokens} tokens")
    # ... compact ...
    print(f"[context_compaction] Reduced to {new_tokens} tokens")
```

## The Real Problem

The L6 "idle loop" is actually a **perfectionism loop**:

The agent refuses to report completion until:
- ✅ Code is written
- ✅ Tests are written
- ✅ Tests PASS

But criterion #3 is often environment-dependent and may be impossible to satisfy in the given environment. The agent needs to be able to say:

> "I've implemented all requested features with high-quality code. Tests exist but won't run due to environment issues. This is good enough - marking complete."

**Current behavior:** Agent loops forever trying to make tests pass
**Desired behavior:** Agent recognizes "good enough" and moves on

---

## Conclusion

L6 revealed a different bug than initially suspected:
- ❌ NOT: "agent doesn't know when to call mark_complete"
- ✅ ACTUALLY: "agent can't accept test failures and move on"

The fix isn't a "completion nudge" - it's a "failure acceptance" mechanism.
