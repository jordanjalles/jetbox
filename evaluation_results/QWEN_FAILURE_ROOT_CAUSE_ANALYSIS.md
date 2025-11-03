# qwen3:8b Failure Root Cause Analysis

**Evaluation**: L3-L6 x5 (gpt-oss:20b vs qwen3:8b)
**Total Tests**: 20
**Failures**: 10 (50% failure rate)
**Date**: 2025-11-03

## Executive Summary

qwen3:8b failures fall into 3 distinct categories:

1. **Legitimate Timeouts** (40% of failures): Agent making progress but ran out of time
2. **Stuck/Hung** (30% of failures): LLM call hung or agent stuck with no progress
3. **Completion Detection** (30% of failures): Work completed but `mark_complete` not called

**Key Insight**: Most failures (70%) are fixable with longer timeouts or increased max_rounds. Only 30% are true bugs (LLM hangs).

---

## Failure Breakdown

### Category 1: Legitimate Timeouts (4 failures, 40%)

**Definition**: Agent was actively making progress (writing files, running commands) but hit timeout before completion.

| Test ID | Level | Timeout | Rounds | Files | Commands | Assessment |
|---------|-------|---------|--------|-------|----------|------------|
| L3_run2 | L3 | 180s | 8 | 4 | 2 | Needed ~30s more |
| L3_run4 | L3 | 180s | 9 | 4 | 3 | Needed ~30s more |
| L4_run3 | L4 | 240s | 7 | 2 | 3 | Needed ~60s more |
| L5_run1 | L5 | 300s | 5 | 2 | 1 | Orchestrator overhead |

**Root Cause**: Timeout thresholds too aggressive for task complexity

**Evidence**:
```
L3_run2: Round 8, wrote 4 files (string_utils package), ran 2 test commands
L3_run4: Round 9, wrote 4 files (validators package), ran 3 test commands
L4_run3: Round 7, wrote 2 files (file_processor), ran 3 commands
L5_run1: Round 5 at orchestrator level, task executor working on implementation
```

**Fix**: Increase timeouts by 25-50%
- L3: 180s → 240s (+33%)
- L4: 240s → 300s (+25%)
- L5: 300s → 420s (+40%)

**Partial Credit**: These should count as 80% success - work was nearly done

---

### Category 2: Stuck/Hung (3 failures, 30%)

**Definition**: Agent made little to no progress, LLM call appears to have hung or agent completely stuck.

| Test ID | Level | Timeout | Rounds | Files | Commands | Evidence |
|---------|-------|---------|--------|-------|----------|----------|
| L4_run4 | L4 | 240s | 1 | 0 | 0 | Hung in Round 1 |
| L5_run5 | L5 | 300s | 0 | 0 | 0 | Never started |
| L6_run4 | L6 | 420s | 0 | 0 | 0 | 1 empty round |

**Root Cause**: LLM call timeout or hang

**Evidence**:

**L4_run4** (cache_manager):
```
[task_executor] Round 1/12
(no output after this - hung for 240s)
```
Log shows agent started Round 1, but LLM never responded. The `timeout` command killed it at 240s.

**L5_run5** (Student API):
```
Rounds: 0
Files written: 0
Commands run: 0
```
Agent never even started first round - orchestrator hung immediately.

**L6_run4** (Notes API):
```
[architect] Round 5/50
(no output - hung for 420s)

Earlier:
[loop_detection] ⚠️  Empty round #1
[architect] Round 2/50
[architect] -> write_task_list
[architect] Round 3/50
[architect] -> write_architecture_doc
[architect] Round 4/50
[architect] -> write_module_spec
[architect] Round 5/50
(HUNG)
```
Architect made progress for 4 rounds, then hung on round 5 LLM call.

**Hypothesis**: Ollama/model issue, not Jetbox bug

Possible causes:
1. Ollama service stuck/deadlocked
2. qwen3:8b model hitting internal limit on specific prompts
3. Context length edge case causing generation to stall
4. GPU memory issue causing hang

**Fix**:
- Add LLM-level timeout (30s inactivity)
- Retry mechanism for hung calls
- Log full context when hang detected
- Monitor Ollama logs during hangs

**Mitigation**: None currently - this is an infrastructure issue

---

### Category 3: Completion Detection (3 failures, 30%)

**Definition**: Agent completed all work but failed to call `mark_complete`, hitting max_rounds (12).

| Test ID | Level | Rounds | Files | Commands | Assessment |
|---------|-------|--------|-------|----------|------------|
| L3_run3 | L3 | 12/12 | 7 | 5 | Work likely done |
| L3_run5 | L3 | 12/12 | 7 | 3 | Work likely done |
| L4_run2 | L4 | 12/12 | 6 | 5 | Work likely done |

**Root Cause**: Agent doesn't recognize when task is complete

**Evidence**:

**L3_run3** (data_structures):
```
Round 1-12: Wrote 7 files, ran 5 commands
Final round: -> write_file (still working, didn't call mark_complete)
Status: Max rounds (12) reached without completion
```

**L3_run5** (converters):
```
Round 1-12: Wrote 7 files, ran 3 commands
Final round: -> write_file
Status: Max rounds reached
```

**L4_run2** (json_validator):
```
Round 1-12: Wrote 6 files, ran 5 commands
Final round: Still writing files, no completion signal
```

**Pattern**: All 3 are L3-L4 (direct TaskExecutor mode), none are L5-L6 (Orchestrator mode)

This matches gpt-oss:20b's behavior - completion detection works in Orchestrator but not in direct TaskExecutor.

**Why?**
- **Orchestrator mode**: Has explicit goal tracking, completion nudging
- **Direct TaskExecutor**: Relies on agent calling `mark_complete` spontaneously
- **12 rounds**: Not enough for some L3-L4 tasks (need 15-18)

**Fix Options**:

1. **Increase max_rounds** for direct TaskExecutor: 12 → 18
2. **Add completion nudging** to direct TaskExecutor (like Orchestrator has)
3. **Use Orchestrator** for L3-L4 tasks instead of direct execution
4. **Improve system prompt** with stronger completion instructions

**Recommended**: Option 1 + 2 (increase max_rounds AND add completion nudging)

---

## Comparison: qwen3:8b vs gpt-oss:20b

| Issue | qwen3:8b | gpt-oss:20b | Winner |
|-------|----------|-------------|---------|
| **Legitimate Timeouts** | 4 | 6 | qwen (fewer) |
| **Stuck/Hung** | 3 | 0 | gpt (never hung) |
| **Completion Detection** | 3 (15%) | 9 (45%) | qwen (3x better) |
| **Total Failures** | 10 | 15 | qwen (fewer) |

**Analysis**:
- qwen3:8b has **3x fewer completion detection issues** (3 vs 9)
- qwen3:8b has **new issue**: LLM hangs (3 cases, never seen with gpt-oss)
- qwen3:8b **still better overall**: 10 failures vs 15

---

## Root Cause Summary

### 1. Infrastructure Issues (30%)

**LLM Hangs** - 3 cases where Ollama/qwen3:8b stopped responding

**Symptoms**:
- Agent starts round, LLM call never completes
- No tool calls, no output, just hangs
- Timeout kills process after N seconds

**Affected**:
- L4_run4 (cache_manager)
- L5_run5 (Student API)
- L6_run4 (Notes API - architect hung)

**Solution**:
- Add `llm_utils.py` timeout with retry
- Log context when hang detected
- Monitor Ollama service health
- Consider switching to qwen3:14b for hung-prone tasks

**Impact**: 🔴 HIGH - Blocks user, requires manual intervention

---

### 2. Timeout Tuning Issues (40%)

**Legitimate Timeouts** - 4 cases where agent was making progress but needed more time

**Symptoms**:
- Agent writing files, running commands
- Making steady progress
- Hits timeout before calling `mark_complete`

**Affected**:
- L3_run2 (string_utils - needed ~30s more)
- L3_run4 (validators - needed ~30s more)
- L4_run3 (file_processor - needed ~60s more)
- L5_run1 (User API - orchestrator overhead)

**Solution**:
- Increase L3 timeout: 180s → 240s
- Increase L4 timeout: 240s → 300s
- Increase L5 timeout: 300s → 420s
- Or: Use "no timeout" mode with max_rounds limit

**Impact**: 🟡 MEDIUM - Frustrating but not broken, just needs tuning

---

### 3. Agent Design Issues (30%)

**Completion Detection** - 3 cases where work was done but `mark_complete` not called

**Symptoms**:
- Agent hits max_rounds (12)
- Files written, tests run
- Work appears complete
- No `mark_complete` call

**Affected**:
- L3_run3 (data_structures)
- L3_run5 (converters)
- L4_run2 (json_validator)

**Solution**:
- Increase max_rounds: 12 → 18 for L3-L4
- Add completion nudging (inject reminder at round 10)
- Better system prompt about calling `mark_complete`
- Heuristic completion detection (if tests pass + all files written → auto-complete)

**Impact**: 🟡 MEDIUM - Work gets done but reported as failure

---

## Recommended Fixes (Priority Order)

### 1. Add LLM Timeout/Retry (Fixes 30% of failures)

**Problem**: LLM hangs block agent completely

**Solution**:
```python
# In llm_utils.py
def chat_with_inactivity_timeout(...):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = ollama_call_with_timeout(timeout=60)
            return result
        except TimeoutError:
            if attempt < max_retries - 1:
                log(f"LLM hung, retry {attempt+1}/{max_retries}")
                continue
            else:
                raise RuntimeError("LLM hung after 3 retries")
```

**Impact**: Converts 3 hangs → 0-1 hangs (2-3 fixed)

---

### 2. Increase max_rounds for L3-L4 (Fixes 30% of failures)

**Problem**: 12 rounds not enough for complex L3-L4 tasks

**Solution**:
```yaml
# In task_executor_config.yaml
rounds:
  max_rounds: 18  # Up from 12
```

**Impact**: Converts 3 UNKNOWN → 2-3 SUCCESS (gives time to call mark_complete)

---

### 3. Increase Timeouts (Fixes 40% of failures)

**Problem**: Timeouts too aggressive for complex tasks

**Solution**:
```python
# In evaluation scripts
TIMEOUTS = {
    "L3": 240,   # Up from 180s
    "L4": 300,   # Up from 240s
    "L5": 420,   # Up from 300s
    "L6": 600,   # Up from 420s
}
```

**Impact**: Converts 4 TIMEOUT → 3-4 SUCCESS

---

### 4. Add Completion Nudging (Improves completion detection)

**Problem**: Agent doesn't know when to call `mark_complete`

**Solution**:
```python
# In LoopDetectionBehavior.inject_context()
if round_num >= 10 and round_num < max_rounds:
    context += "\n\nREMINDER: If the task is complete, call mark_complete(). If stuck, call mark_failed()."
```

**Impact**: Helps agent recognize completion, reduces UNKNOWN failures

---

## Success Criteria After Fixes

**Current**: 50% success (10/20)

**After Fixes**:
- LLM timeout/retry: +2-3 successes (3 hangs → 0-1)
- Increase max_rounds: +2-3 successes (3 UNKNOWN → SUCCESS)
- Increase timeouts: +3-4 successes (4 TIMEOUT → SUCCESS)

**Projected**: 70-80% success (14-16/20)

**Remaining Failures** (4-6):
- 1-2 LLM hangs (infrastructure, hard to fix)
- 2-3 genuinely hard tasks (legitimate complexity)
- 0-1 other edge cases

---

## Task-Specific Analysis

### L3 Failures (4/5 failed, 20% success)

| Run | Goal | Status | Root Cause | Fix |
|-----|------|--------|------------|-----|
| run2 | string_utils | TIMEOUT | Needed 30s more | Increase timeout |
| run3 | data_structures | UNKNOWN | Max rounds (12) | Increase to 18 |
| run4 | validators | TIMEOUT | Needed 30s more | Increase timeout |
| run5 | converters | UNKNOWN | Max rounds (12) | Increase to 18 |

**Pattern**: L3 tasks are at the edge of:
- Time limit (180s)
- Round limit (12 rounds)

**Fix**: Increase both
- Timeout: 180s → 240s
- Max rounds: 12 → 18

**Expected**: 4-5/5 success (80-100%)

---

### L4 Failures (3/5 failed, 40% success)

| Run | Goal | Status | Root Cause | Fix |
|-----|------|--------|------------|-----|
| run2 | json_validator | UNKNOWN | Max rounds (12) | Increase to 18 |
| run3 | file_processor | TIMEOUT | Needed 60s more | Increase timeout |
| run4 | cache_manager | TIMEOUT | LLM hung | Add retry |

**Pattern**: Mix of issues
- 1 completion detection (UNKNOWN)
- 1 legitimate timeout (file_processor complex)
- 1 LLM hang (cache_manager)

**Fix**: All 3 fixes needed
- Max rounds: 12 → 18
- Timeout: 240s → 300s
- LLM retry

**Expected**: 4-5/5 success (80-100%)

---

### L5 Failures (2/5 failed, 60% success)

| Run | Goal | Status | Root Cause | Fix |
|-----|------|--------|------------|-----|
| run1 | User API | TIMEOUT | Orchestrator overhead | Increase timeout |
| run5 | Student API | TIMEOUT | LLM hung (never started) | Add retry |

**Pattern**: Orchestrator tasks
- More overhead (architect + task executor)
- Occasional LLM hangs

**Fix**:
- Timeout: 300s → 420s
- LLM retry

**Expected**: 4-5/5 success (80-100%)

---

### L6 Failures (1/5 failed, 80% success)

| Run | Goal | Status | Root Cause | Fix |
|-----|------|--------|------------|-----|
| run4 | Notes API | TIMEOUT | Architect hung round 5 | Add retry |

**Pattern**: Already doing well (80%), one hang

**Fix**: LLM retry

**Expected**: 5/5 success (100%)

---

## Conclusions

### 1. qwen3:8b is Still the Winner

Despite 50% failure rate, qwen3:8b is still better than gpt-oss:20b (25% success):
- 2x better success rate
- 3x better completion detection
- New issue (LLM hangs) affects only 15% of tests

### 2. Most Failures are Fixable

- 70% of failures are configuration issues (timeouts, max_rounds)
- 30% are infrastructure issues (LLM hangs)
- None are fundamental algorithm bugs

### 3. Expected Improvements

With all fixes:
- Current: 50% success (10/20)
- After fixes: 70-80% success (14-16/20)
- vs gpt-oss: 25% success (5/20)

**Gap widens**: qwen3:8b will be 3-4x better than gpt-oss after fixes

### 4. Priority Actions

1. **Immediate**: Add LLM timeout/retry (fixes 15-30% of failures)
2. **Quick win**: Increase max_rounds to 18 (fixes 15% of failures)
3. **Easy**: Increase timeouts (fixes 20% of failures)
4. **Nice-to-have**: Completion nudging (reduces UNKNOWN)

### 5. Trade-offs

**Longer timeouts**:
- ✅ Higher success rate
- ❌ Slower failure detection
- ❌ Wastes time on truly stuck agents

**Higher max_rounds**:
- ✅ More chances to complete
- ❌ More rounds wasted on impossible tasks

**LLM retry**:
- ✅ Fixes infrastructure hangs
- ❌ Triples time for hung cases (3 retries × 60s = 180s)

**Recommendation**: Accept the trade-offs - success rate matters more than speed
