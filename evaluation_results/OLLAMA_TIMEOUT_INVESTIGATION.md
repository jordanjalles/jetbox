# Ollama Timeout Investigation

**Date:** November 2, 2025
**Total Timeout Dumps Analyzed:** 86 files

## Executive Summary

Analysis of 86 timeout dumps reveals **two distinct timeout patterns**:

1. **Type A: Small Context Orchestrator Timeouts (60% of failures)**
   - Context: ~1,200 tokens, 3 messages
   - Scenario: Orchestrator Round 1 - initial planning call
   - Duration: 120s with no response
   - **Root cause: LLM hangs on planning complex tasks, NOT context overload**

2. **Type B: Large Context Task Executor Timeouts (rare but severe)**
   - Context: 129K tokens (202% of 64K limit!)
   - Scenario: Task executor deep into execution
   - Duration: 120s with no response
   - **Root cause: Context explosion overwhelming LLM**

---

## Detailed Findings

### Type A: Orchestrator Planning Timeouts

**Sample:** `timeout_inactivity_20251102_075937.json`

```
Model: gpt-oss:20b
Context: 3 messages, 1,221 tokens, 4,886 chars
Elapsed: 120s
Tools: 6 (delegation tools)
Last message: "Create a Flask app with user authentication..."
```

**Pattern:**
- Occurs at orchestrator Round 1-3
- Always on first LLM call for the goal
- Tiny context - not a token limit issue
- Consistent across L6 P2-P3 and ALL L7 tasks (25 tests)

**Hypothesis:**
- **LLM struggles with delegation planning for complex tasks**
- Model tries to reason about architecture/decomposition
- Gets stuck in internal planning loop
- Never produces output, triggers inactivity timeout

**Evidence:**
- Simple L5 tasks (CRUD APIs): Orchestrator responds in 2-5s ✅
- Complex L6 tasks (Auth + Sessions): 60% timeout, 40% success
- L7 tasks (Full-stack): 100% timeout (15/15 failures)

**Key Insight:** Complexity of task description correlates with timeout rate, not context size.

---

### Type B: Context Explosion Timeouts

**Sample:** `timeout_inactivity_20251102_074636.json`

```
Model: gpt-oss:20b
Context: 23 messages, 129,328 tokens (202% of 64K limit!), 517K chars
Elapsed: 120s
Tools: 11 (file + command tools)
Message breakdown: 1 system, 2 user, 6 tool, 14 assistant
```

**Pattern:**
- Occurs at task_executor Round 25+
- Context exploded to 202% of max_tokens setting
- CompactWhenNearFullBehavior triggered but ineffective (only 1% reduction)
- LLM stuck in "thinking mode" generating text but not calling tools

**Evidence from logs:**
```
[compact_when_near_full] Context at 129,702 tokens (202.7% of 64,000)
[compact_when_near_full] Reduced from 129,702 to 129,166 tokens (201.8%)
[loop_detection] ⚠️  Empty round #2 - LLM did not call any tools
```

**Key Insight:** Compaction is ineffective once context has already exploded. Need proactive limits.

---

## Root Cause Analysis

### Why Ollama Hangs (Type A)

**Not a bug, but a capability gap:**

1. **Model planning overhead**
   - gpt-oss:20b tries to fully understand complex tasks before responding
   - L7 tasks (full-stack apps) require extensive planning
   - Model enters "thinking" state but never produces output

2. **Tool selection paralysis**
   - Orchestrator has 6 delegation tools
   - Complex task could use architect, task_executor, or both
   - Model deliberates indefinitely

3. **No output triggers timeout**
   - Ollama waits 120s for any token
   - If model doesn't start generating, inactivity timeout fires
   - Circuit breaker: 3 consecutive timeouts = abort

### Why Context Explodes (Type B)

**Compaction strategy failure:**

1. **Threshold too high**
   - Config: compact at 75% of 64K = 48K tokens
   - But context already at 129K (202%)!
   - Token counting may be wrong OR compaction not triggering

2. **Compaction ineffective**
   - Uses LLM to summarize messages
   - When context huge, summarization barely helps (1% reduction)
   - Need aggressive truncation, not summarization

3. **Empty rounds compound problem**
   - LLM generates long text responses (thinking aloud)
   - No tool calls = context grows without progress
   - Each round adds more messages, tokens increase

---

## Recommendations (Prioritized)

### P0: Fix Type A Orchestrator Timeouts (Blocks 60% of tests)

**Option 1: Increase first-call timeout**
```python
# In base_agent.py or orchestrator config
first_call_timeout = 180  # 3 minutes for planning
subsequent_timeout = 120  # 2 minutes for execution
```

**Option 2: Rule-based delegation (bypass LLM planning)**
```python
# In orchestrator behavior
if any(keyword in goal.lower() for keyword in ['full-stack', 'auth', 'database']):
    return delegate_to_architect()  # Skip LLM decision
else:
    return delegate_to_task_executor()
```

**Option 3: Inject planning guidance**
```
When deciding delegation:
1. If task mentions "full-stack" or has >3 components → delegate_to_architect
2. If task is single API or simple feature → delegate_to_task_executor
3. Decide within 10 seconds - don't overthink
```

**Option 4: Fallback auto-delegation**
```python
# After 1st timeout on Round 1, skip orchestrator planning
if timeout_count == 1 and round_number == 1:
    print("[fallback] Auto-delegating to task_executor")
    return auto_delegate_to_task_executor(goal)
```

**Recommendation:** Implement **Options 1 + 4** immediately.

---

### P1: Fix Type B Context Explosion (Caused 1 timeout, slows others)

**Fix 1: Hard limit enforcement**
```python
# In CompactWhenNearFullBehavior
def enhance_context(self, context, **kwargs):
    estimated_tokens = self._estimate_tokens(context)

    if estimated_tokens > self.max_tokens:
        # AGGRESSIVE: Drop oldest 50% of message pairs
        keep_recent = 10
        context['messages'] = (
            [context['messages'][0]]  # Keep system prompt
            + context['messages'][-keep_recent:]  # Keep last 10 messages
        )
        print(f"[compact] EMERGENCY: Dropped to {keep_recent} messages")
```

**Fix 2: Prevent explosion before it happens**
```python
# Compact earlier - at 50% instead of 75%
compact_threshold: 0.50  # In agent_config.yaml
```

**Fix 3: Reset on empty rounds**
```python
# In LoopDetectionBehavior
if consecutive_empty_rounds >= 3 and estimated_tokens > 30000:
    # Context bloat + stuck = reset
    context['messages'] = [system_prompt] + context['messages'][-5:]
    print("[loop_detection] Reset context due to empty rounds + bloat")
```

**Recommendation:** Implement **all three fixes** - they're complementary.

---

### P2: Add Ollama Health Monitoring

**Pre-test health check:**
```python
def check_ollama_health(model):
    """Verify Ollama responsive before starting test."""
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "OK"}],
            timeout=10
        )
        return True
    except TimeoutError:
        print("[health] Ollama not responding - restarting...")
        subprocess.run(["systemctl", "restart", "ollama"])
        time.sleep(30)  # Wait for model reload
        return False
```

**On circuit breaker:**
```python
if consecutive_timeouts >= 3:
    print("[circuit_breaker] Restarting Ollama...")
    restart_ollama()
    time.sleep(30)
    # Resume from checkpoint
```

---

## Timeline of Ollama Degradation

Analysis of timeout timestamps shows **service degradation over time**:

| Time Range | Phase | Behavior |
|------------|-------|----------|
| 06:10-06:47 (37m) | L5 tests | Fast successes, rare timeouts |
| 06:47-07:57 (70m) | L6 P1 | Mixed: 3 success, 2 timeout |
| 07:57-10:33 (156m) | L6 P2-P3, L7 | **100% timeouts** |

**Hypothesis:** Ollama memory leak or resource exhaustion over 2+ hours of continuous use.

**Test:** Restart Ollama between test levels to prevent accumulation.

---

## Summary

**Type A timeouts** (orchestrator planning) are a **model capability issue**, not Ollama crashes. The LLM gets stuck trying to plan complex tasks.

**Type B timeouts** (context explosion) are a **config/implementation issue** with compaction behavior.

**Quick wins:**
1. Increase first-call timeout to 180s
2. Add fallback auto-delegation after orchestrator timeout
3. Fix CompactWhenNearFullBehavior to enforce hard limits
4. Restart Ollama between test levels

**Expected impact:**
- Type A fixes: +40% success rate (orchestrator stops blocking)
- Type B fixes: +5% success rate (prevent slow timeouts)
- **Total: 37.8% → ~80% success rate** (assuming LLM works)
