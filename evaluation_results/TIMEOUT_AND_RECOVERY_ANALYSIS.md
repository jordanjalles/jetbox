# Timeout and Recovery Failure Analysis

**Date**: 2025-11-03
**Issue**: Agents timing out during L5-L7 evaluation despite loop detection and recovery mechanisms

---

## Executive Summary

**ROOT CAUSE**: The loop detection recovery mechanism **is working** (injecting prompts every 5 empty rounds), but the LLM **does not respond** to recovery prompts. It gets stuck in an infinite loop of empty rounds until hitting max_rounds=50, causing timeouts.

**KEY FINDING**: The architect tries to call `write_file` (which doesn't exist in its toolset), realizes it can't complete the task, but instead of calling `mark_failed()`, it just keeps having empty rounds.

**IMPACT**: 40% of L5 tests (2/5) and 100% of L6 tests timeout due to this issue.

---

## Detailed Analysis

### Test Results Summary

**L5 Tests (from rerun)**:
- ✅ L5_run2: SUCCESS (34.5s) - Fast completion
- ✅ L5_run4: SUCCESS (114.5s) - Normal completion
- ✅ L5_run5: SUCCESS (67.0s) - Normal completion
- ⏱️ L5_run1: TIMEOUT (300.0s) - Architect stuck in empty rounds
- ⏱️ L5_run3: TIMEOUT (300.0s) - Architect stuck in empty rounds

**Success Pattern**: 60% success rate when architect completes normally
**Failure Pattern**: 40% timeout when architect gets stuck

---

## Root Cause: Empty Round Infinite Loop

### Timeline of L5_run1 (Timeout Example)

```
Round 1-5: Architect creates architecture docs successfully
  [architect] -> write_architecture_doc     ✅
  [architect] -> write_module_spec (x3)    ✅
  [architect] -> write_task_list           ✅

Round 6: Architect tries to call write_file (DOESN'T EXIST!)
  [architect] -> write_file                ❌ (Tool doesn't exist)
  [loop_detection] ⚠️ Empty round #1

Round 7-8: LLM realizes it can't create files
  [loop_detection] ⚠️ Empty round #2-3
  LLM response: "I'm unable to create the Flask application...
                 workspace does not provide a file-creation tool"

Round 9: First recovery injection
  [loop_detection] Injecting empty round recovery (round 3)

Round 10-13: LLM still having empty rounds
  [loop_detection] ⚠️ Empty round #4-7
  LLM keeps saying "No tool available"

Round 14-50: Recovery injected every 5 rounds, LLM never recovers
  Injection at rounds: 9, 14, 19, 24, 29, 34, 39, 44, 49
  LLM continues empty rounds: 31+ consecutive empty rounds
  Eventually hits max_rounds=50
```

**Total Time**: ~150 seconds for architect to burn through 50 rounds
**Orchestrator Time**: Additional time waiting/delegating
**Result**: Timeout at 300s before task completes

---

## Why Recovery Doesn't Work

### The Recovery Mechanism (Current Behavior)

**Code**: `behaviors/loop_detection.py:229-305`

```python
# Recovery injection logic
should_inject = (
    not self.recovery_prompt_injected or
    (self.consecutive_empty_rounds % 5 == 0)  # Repeat every 5 empty rounds
)
```

**What it does**:
1. Detects when consecutive_empty_rounds >= 3
2. Injects recovery prompt into context
3. Re-injects every 5 empty rounds (at 5, 10, 15, 20, 25, 30...)
4. Recovery prompt includes:
   - Current goal
   - Available tools list
   - Instructions to call mark_failed if blocked
   - Urgent warnings

**Recovery Prompt Content** (line 277-301):
```
🚨 WARNING: 10 consecutive empty rounds - NO TOOLS CALLED!

GOAL:
  Create a Flask REST API with CRUD endpoints...

YOUR TOOLS:
  write_architecture_doc, write_module_spec, write_task_list, mark_complete, mark_failed

ACTION REQUIRED NOW:
  • If work is DONE: call mark_complete or mark_failed
  • If work is BLOCKED: call mark_failed(reason="...")
  • If work CONTINUES: call the next tool needed

YOU CANNOT PROCEED WITHOUT CALLING A TOOL.
Look at the tool list above and call one NOW.
```

### Why LLM Doesn't Respond

**Evidence from logs** (L5_run1.log:75):
```
[loop_detection] LLM response: I'm unable to create the Flask application
and its test suite because the workspace does not provide a file‑creation
tool. The only available operations are for writing architecture
documentation and ...
```

**The Problem**:
1. LLM **understands** it can't complete the task
2. LLM **sees** the recovery prompt listing available tools
3. LLM **knows** `mark_failed` is available
4. BUT LLM **doesn't call** `mark_failed()`
5. Instead, LLM just responds with text explaining why it can't proceed
6. This creates another empty round, triggering recovery again
7. Infinite loop ensues

**Root Cause**: The LLM is responding to the prompt conversationally instead of calling the tool. This is likely because:
- The system prompt doesn't emphasize tool calls strongly enough
- The LLM is trained to explain problems rather than signal failure
- The recovery prompt is interpreted as a question, not a command

---

## Missing Mechanism: Auto-Fail After N Empty Rounds

**Current Behavior**:
- No hard limit on consecutive empty rounds
- Agent burns through all max_rounds (50) doing nothing
- Takes ~150s to exhaust rounds
- Eventually times out

**What's Missing**:
```python
# This doesn't exist!
if self.consecutive_empty_rounds >= AUTO_FAIL_THRESHOLD:  # e.g., 20
    raise AutoFailException("Too many consecutive empty rounds")
```

**Impact**:
- Agents waste time in infinite loops
- Timeouts occur instead of graceful failures
- No clear signal to parent agent that subagent is stuck

---

## Why Some Tests Succeed and Others Timeout

### Success Pattern (L5_run2, L5_run4, L5_run5)

```
Round 1: Orchestrator -> Architect
Architect rounds 1-9: Creates architecture docs successfully
Architect round 10: Calls mark_complete ✅

Round 2: Orchestrator -> Task Executor
Task Executor rounds 1-20: Implements Flask API + tests
Task Executor round 21: Calls mark_complete ✅

Round 3: Orchestrator calls mark_goal_complete ✅
Total time: 30-120s
```

**Key**: Architect completes quickly (6-10 rounds) and doesn't try to create files

### Timeout Pattern (L5_run1, L5_run3)

```
Round 1: Orchestrator -> Architect
Architect rounds 1-5: Creates architecture docs successfully
Architect round 6: Tries to call write_file ❌
Architect rounds 7-50: Empty rounds (44 consecutive!)
Architect hits max_rounds, returns to orchestrator

Round 2: Orchestrator might delegate to Task Executor
But architect took 150s, now only 150s left before 300s timeout
Task Executor runs out of time...
Timeout at 300s ⏱️
```

**Key**: Architect burns 150s in empty rounds, leaving insufficient time for task executor

---

## Why Architect Tries to Call write_file

### Architect's Available Tools

**Defined in**: `architect_config.yaml` via `ArchitectToolsBehavior`

```yaml
behaviors:
  - type: ArchitectToolsBehavior
```

**Tools provided**:
- `write_architecture_doc` - Create system overview
- `write_module_spec` - Create module specifications
- `write_task_list` - Create task breakdown
- `mark_complete` - Signal completion
- `mark_failed` - Signal failure

**Tools NOT available**:
- `write_file` - Generic file creation (Task Executor tool)
- `read_file` - File reading (Task Executor tool)
- `run_bash` - Command execution (Task Executor tool)

### Why LLM Hallucinates write_file

**Hypothesis 1**: System prompt confusion
- Architect's system prompt may not clearly distinguish its role
- LLM may think it's a general-purpose agent

**Hypothesis 2**: Tool description ambiguity
- `write_architecture_doc` and `write_module_spec` create files
- LLM may generalize to "I can create any file"

**Hypothesis 3**: Goal phrasing
- Goal says "Create a Flask REST API"
- LLM interprets this as "I should create the Flask code"
- Doesn't understand it should only create architecture docs

**Evidence from log**:
- Architect successfully creates architecture docs (rounds 1-5)
- Then tries to create the actual Flask code (round 6)
- Realizes it can't, but doesn't know how to signal failure

---

## Impact Analysis

### Time Breakdown (L5_run1 Timeout)

| Phase | Rounds | Time | Status |
|-------|--------|------|--------|
| Architect initialization | - | 5s | ✅ |
| Architect work (rounds 1-5) | 5 | 15s | ✅ |
| Architect stuck (rounds 6-50) | 44 | 130s | ⏱️ |
| Orchestrator overhead | - | 10s | ✅ |
| Task Executor (if started) | ? | 140s | ⏱️ Insufficient |
| **Total** | 50+ | **300s** | **TIMEOUT** |

**Wasted Time**: 130s (43% of total timeout) spent in empty rounds

### Comparison: Success vs Timeout

| Metric | Success (L5_run2) | Timeout (L5_run1) |
|--------|-------------------|-------------------|
| Architect rounds | 9 | 50 |
| Architect time | 15s | 150s |
| Task Executor time | 20s | Not enough |
| Total time | 34.5s | 300s (timeout) |
| Wasted time | 0s | 130s |

**10x Time Difference**: Empty rounds cause 10x slowdown

---

## Proposed Fixes

### Fix 1: Auto-Fail After N Empty Rounds (CRITICAL)

**Priority**: P0 - Prevents infinite loops

**Implementation**: Add to `behaviors/loop_detection.py`

```python
class LoopDetectionBehavior(AgentBehavior):
    def __init__(self, max_repeats: int = 5, max_empty_rounds: int = 3,
                 auto_fail_threshold: int = 20):
        self.max_empty_rounds = max_empty_rounds
        self.auto_fail_threshold = auto_fail_threshold  # NEW

    def on_round_start(self, context, **kwargs):
        # Existing recovery injection code...

        # NEW: Auto-fail after too many empty rounds
        if self.consecutive_empty_rounds >= self.auto_fail_threshold:
            print(f"[loop_detection] ❌ AUTO-FAIL: {self.consecutive_empty_rounds} "
                  f"consecutive empty rounds exceeded threshold ({self.auto_fail_threshold})")

            # Force agent to fail
            agent = kwargs.get('agent')
            if agent and hasattr(agent, 'force_fail'):
                agent.force_fail(
                    reason=f"Auto-failed after {self.consecutive_empty_rounds} "
                           f"consecutive empty rounds without tool calls"
                )

            # Or raise exception to break out of run loop
            raise RuntimeError(
                f"Auto-fail: {self.consecutive_empty_rounds} consecutive empty rounds"
            )
```

**Expected Impact**:
- Agents fail after 20 empty rounds instead of 50
- Saves 60s per stuck agent
- Orchestrator can retry or move forward

**Configuration**:
```yaml
# agent_config.yaml
behavior_defaults:
  loop_detection:
    max_repeats: 5
    max_empty_rounds: 3
    auto_fail_threshold: 20  # NEW
```

### Fix 2: Improve Architect System Prompt (HIGH PRIORITY)

**Priority**: P0 - Prevents write_file hallucination

**Current Issue**: Architect doesn't understand it should ONLY create architecture docs

**Proposed Change**: `architect_config.yaml`

```yaml
system_prompt: |
  You are the ARCHITECT agent. Your role is DESIGN ONLY - you create architecture
  documentation, NOT implementation code.

  YOUR TOOLS (ONLY THESE):
  - write_architecture_doc: Create system overview
  - write_module_spec: Create module specifications
  - write_task_list: Create task breakdown
  - mark_complete: Signal you've finished architecture design
  - mark_failed: Signal if you cannot create adequate architecture

  CRITICAL: You CANNOT create implementation files (app.py, tests, etc.)!

  Your architecture will be handed to a TASK EXECUTOR agent who will implement it.

  WORKFLOW:
  1. Understand the requirements
  2. Create architecture docs (system overview, module specs, task breakdown)
  3. Call mark_complete() when architecture is ready
  4. If requirements are unclear or impossible, call mark_failed(reason="...")

  DO NOT try to implement the actual code - that's not your job!
```

**Expected Impact**:
- Architect won't try to call write_file
- Architect will call mark_complete after creating architecture
- Eliminates primary cause of empty round loops

### Fix 3: Stronger Tool Call Enforcement in Recovery (MEDIUM PRIORITY)

**Priority**: P1 - Makes recovery more effective

**Current Issue**: LLM responds with text instead of calling tools

**Proposed Change**: `behaviors/loop_detection.py:297-301`

```python
recovery.extend([
    "",
    "🚨 CRITICAL: YOU MUST CALL A TOOL IN YOUR NEXT RESPONSE!",
    "",
    "DO NOT respond with text explaining the situation.",
    "DO NOT try to discuss or analyze.",
    "DO NOT wait for further instructions.",
    "",
    "REQUIRED ACTION (choose ONE):",
    "1. If work is complete: call mark_complete(summary='...')",
    "2. If blocked/impossible: call mark_failed(reason='...')",
    "3. If continuing: call the next appropriate tool",
    "",
    "Your response MUST contain a tool call. No exceptions."
])
```

**Expected Impact**:
- LLM more likely to call tools instead of explaining
- Faster recovery from empty rounds
- Reduced wasted rounds

### Fix 4: Early Completion Detection for Architect (LOW PRIORITY)

**Priority**: P2 - Optimization

**Observation**: Architect often completes architecture in 5-9 rounds but doesn't call mark_complete

**Proposed**: Add completion heuristic

```python
# In ArchitectToolsBehavior
def on_tool_call(self, tool_name, args, result, **kwargs):
    if tool_name == 'write_task_list':
        # Task list is usually the last architecture artifact
        self.task_list_written = True

    # Check if architecture is complete
    if (self.task_list_written and
        self.architecture_doc_written and
        len(self.module_specs) >= 2):

        print("[architect_tools] Architecture appears complete. "
              "Suggesting mark_complete...")
        # Inject suggestion in next context
```

**Expected Impact**:
- Faster architect completion
- Fewer unnecessary rounds
- Clearer signal to LLM that work is done

---

## Recommended Action Plan

### Immediate (Today)

1. **Implement Fix 1**: Auto-fail after 20 empty rounds
   - Prevents infinite loops
   - Saves ~60s per stuck agent
   - Critical for all agent types

2. **Implement Fix 2**: Improve architect system prompt
   - Prevents write_file hallucination
   - Eliminates root cause of empty rounds
   - Critical for L5-L7 success

3. **Test with single L5 task**: Verify fixes work
   ```bash
   python orchestrator_main.py "Create a Flask REST API for Users" --once
   ```

### Short-term (This Week)

4. **Implement Fix 3**: Stronger recovery enforcement
   - Makes existing recovery more effective
   - Helps all agent types

5. **Rerun L5-L7 x5 evaluation**: Measure improvement
   - Expected: 80%+ L5 success
   - Expected: Faster completion times
   - Expected: Fewer timeouts

6. **Increase timeout for L6/L7**: 300s → 600s
   - L6/L7 tasks are legitimately complex
   - May need extra time even with fixes

### Long-term (Next Sprint)

7. **Implement Fix 4**: Early completion detection
8. **Add timeout telemetry**: Track where time is spent
9. **Optimize LLM calls**: Reduce unnecessary context

---

## Success Metrics

### Before Fixes

- L5 success rate: 60% (3/5)
- L5 timeout rate: 40% (2/5)
- Average L5 time (success): 72s
- Average L5 time (timeout): 300s
- Wasted time per timeout: 130s

### After Fixes (Projected)

- L5 success rate: 80-100% (4-5/5)
- L5 timeout rate: 0-20% (0-1/5)
- Average L5 time (success): 60s
- Average L5 time (timeout): 0s or quick fail at 40s
- Wasted time per timeout: 0s

**Expected Improvement**: 2-3x faster, 33% higher success rate

---

## Conclusion

The timeout issue is caused by a **combination of two problems**:

1. **Architect role confusion**: Tries to implement code instead of just designing architecture
2. **Missing auto-fail mechanism**: Gets stuck in infinite empty rounds instead of failing fast

Both problems are **easily fixable** with the proposed changes. The recovery mechanism is actually working (injecting prompts every 5 rounds), but the LLM doesn't respond to it appropriately.

**Priority Actions**:
1. ✅ Add auto-fail after 20 empty rounds
2. ✅ Clarify architect role in system prompt
3. ✅ Strengthen recovery prompt language
4. ✅ Test and rerun evaluation

**Expected Outcome**: 80-100% success rate on L5 tests after fixes.
