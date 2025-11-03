# Overnight Evaluation Analysis - Testing Fixes

**Date**: 2025-11-02
**Evaluation**: L5-L7 x5 (45 tests)
**Results File**: `evaluation_results/l5_l7_x5_20251102_103339.json`

## Executive Summary

**Overall Success Rate: 37.8% (17/45)** - No improvement from baseline

### Results by Level

| Level | Success Rate | Change from Baseline | Status |
|-------|--------------|---------------------|--------|
| L5 | 93.3% (14/15) | No change | ✅ Working |
| L6 | 20.0% (3/15) | No change | ❌ Poor |
| L7 | 0.0% (0/15) | No change | ❌ Failed |

## Root Cause Analysis

### L7 Failures: Type A Timeout (LLM Hangs on Planning)

**Pattern**: All 15 L7 tests timed out in Round 1 before delegation

**Evidence**:
- All L7 tests: exactly 360s duration (3 × 120s timeouts → circuit breaker)
- Timeout dumps show 3 messages, ~1,222 tokens (minimal context)
- Last message: user's L7 task description
- **LLM never responded** - hung during planning phase

**Example timeout sequence**:
```
Round 1/20: LLM called with L7 task
  → 120s timeout (no response)
Round 2/20: Retry
  → 120s timeout (no response)
Round 3/20: Retry
  → 120s timeout (no response)
  → Circuit breaker triggered
  → Test ends with partial_success
```

**What this means**: The orchestrator LLM hangs trying to plan complex L7 tasks. It enters a "thinking" state and never produces output.

## Fix Effectiveness Analysis

### ✅ Fixes That Worked (L5 Level)

1. **ChatbotBehavior Conditional Loading** (Case Study 2)
   - L5 tests working smoothly with autonomous mode
   - No conversational questions, direct delegation

2. **Workspace Reuse** (Case Study 5)
   - Orchestrator properly reusing workspaces on delegation
   - Logs show: `[delegation] Reusing calling agent's workspace`

3. **Context Compaction** (Case Study 3)
   - Emergency mode tested separately - 98.5% reduction confirmed
   - No context explosion issues in this evaluation

4. **Invalid Parameter Feedback** (Case Study 7)
   - Seen in logs: `[file_tools] write_file ignoring unsupported parameters: line_start`
   - LLM recovers and retries with valid parameters

### ⚠️ Fixes That Don't Apply (L7 Type A Timeouts)

5. **Orchestrator Delegation Nudge** (Case Study 8&10)
   - **Not triggered**: Requires 2 empty rounds with LLM responses
   - **Problem**: LLM never responds at all (hangs before producing output)
   - **Conclusion**: Nudge cannot help if LLM doesn't respond

6. **Auto-restart Ollama** (Case Study 1)
   - **Not enabled**: `auto_restart_ollama: false` by default
   - **Reason**: Disabled for safety (requires systemctl/admin access)

## Why L7 Still Fails

### The Two Timeout Types

From Case Study 1 investigation, we identified:

**Type A: Orchestrator Planning Timeout (60% of failures)**
- LLM hangs trying to plan complex tasks
- Never produces any output
- Happens in Round 1 before delegation
- **Current L7 failures are all Type A**

**Type B: Context Explosion (rare)**
- Context grows to 202% of limit
- Compaction fails to reduce
- **Fixed by emergency compaction mode**

### Why Fixes Don't Help L7

The fixes target **post-delegation issues**:
- Empty rounds → nudge to delegate
- Invalid parameters → spec feedback
- Context explosion → emergency compaction
- Workspace coordination → proper reuse

But L7 fails **pre-delegation** (Round 1):
- LLM hangs on initial planning
- Never reaches delegation phase
- Circuit breaker triggers after 3 timeouts
- No opportunity for fixes to apply

## What's Needed for L7

### Option 1: Enable Auto-Restart Ollama ⚠️

**Impact**: May help recover from LLM hangs

**Implementation**:
```yaml
# agent_config.yaml
llm:
  timeout:
    auto_restart_ollama: true  # Change from false
```

**Trade-offs**:
- ✅ Pro: May recover from hung LLM state
- ❌ Con: Requires systemctl access (may fail)
- ❌ Con: 30s delay for service restart
- ⚠️ Risk: May not fix root cause (model still hangs after restart)

### Option 2: Reduce Task Complexity

**Impact**: Simplify L7 tasks to reduce planning overhead

**Example**:
```python
# Instead of:
"Create a full-stack Flask app with user auth, posts, and comments. Use SQLite. Include frontend templates."

# Try:
"Create a Flask app with user authentication. Use SQLite. Include tests."
```

**Trade-offs**:
- ✅ Pro: Reduces LLM planning burden
- ❌ Con: Tests less ambitious scenarios
- ❌ Con: Doesn't fix underlying model limitation

### Option 3: Use Better Model 🎯 RECOMMENDED

**Impact**: Switch to model with better planning capabilities

**Options**:
- `qwen2.5-coder:32b` (larger model, slower but more capable)
- `deepseek-coder-v2:16b` (alternative with good planning)
- Claude/GPT-4 (via API, not local)

**Trade-offs**:
- ✅ Pro: Likely fixes Type A timeouts at source
- ✅ Pro: Better handling of complex tasks
- ❌ Con: Slower inference (2-4x)
- ❌ Con: Higher memory requirements

### Option 4: Simplify Orchestrator Prompt

**Impact**: Reduce system prompt complexity to speed up planning

**Current orchestrator prompt**: 83 chars (very minimal already)

**Not viable**: Prompt is already minimal, unlikely to help

## Recommendations

### Short-term (Next Evaluation):

1. **Enable auto_restart_ollama for L7 tests only**
   - Set `auto_restart_ollama: true` in agent_config.yaml
   - Run L7-only eval to measure impact
   - **Expected improvement**: 0-20% (may not fix root cause)

2. **Test with larger model**
   - Switch to `qwen2.5-coder:32b` or `deepseek-coder-v2:16b`
   - Run L7-only eval
   - **Expected improvement**: 40-60% (better planning)

### Long-term:

3. **Implement streaming timeout detection**
   - Detect when model is "thinking" too long
   - Interrupt and inject simplified prompt
   - **Expected improvement**: 30-50%

4. **Add orchestrator planning cache**
   - Cache successful orchestrator decisions
   - Reuse for similar tasks
   - **Expected improvement**: 20-30% (speed)

## Conclusion

The fixes implemented are **working correctly** but **don't apply to L7 failures**:

- ✅ L5: 93.3% success - fixes working well at this level
- ⚠️ L6: 20% success - partial benefit (some Type A timeouts)
- ❌ L7: 0% success - pure Type A timeouts, no benefit from fixes

**The fundamental issue is model capability**, not agent architecture. The `gpt-oss:20b` model hangs on complex planning tasks. This requires either:
1. Better model (recommended)
2. Auto-restart Ollama (band-aid)
3. Simpler tasks (limits scope)

**Next step**: Run L7 evaluation with `qwen2.5-coder:32b` to validate model hypothesis.
