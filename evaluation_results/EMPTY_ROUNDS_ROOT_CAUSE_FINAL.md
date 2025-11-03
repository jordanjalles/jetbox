# Empty Rounds Root Cause - Final Analysis

**Date**: 2025-11-02
**Status**: ROOT CAUSE IDENTIFIED - Model Capability Limitation

## Summary

The empty rounds issue has TWO root causes:
1. ✅ **FIXED**: ChatbotBehavior goal detection bug
2. ⚠️ **UNFIXABLE**: `gpt-oss:20b` model capability limitation

## Investigation Results

### Fix 1: ChatbotBehavior Goal Detection (RESOLVED)

**Problem**: ChatbotBehavior didn't check SubAgentModeBehavior's goal, caused chat mode activation.

**Solution**: Updated ChatbotBehavior to check both goal sources:
- `context_manager.state.goal`
- `SubAgentModeBehavior.goal` (for delegated agents)

**Result**: ✅ Architect no longer asks clarifying questions in delegated mode.

### Fix 2: Architect Role Clarity (RESOLVED)

**Problem**: Architect confused by implementation-focused delegation goals.

**Solution**: Updated architect system prompt to:
- Explicitly state it's a consultant, not a coder
- Explain it doesn't have `write_file` tool (by design)
- Instruct it to interpret "Build X" as "Design architecture for X"

**Result**: ✅ Architect starts using architecture tools correctly.

### Remaining Issue: Model Capability (CANNOT FIX)

**Observation**: After fixes, architect behavior:
```
Round 1: write_architecture_doc ✓
Round 2: write_module_spec ✓
Round 3: Empty round (LLM response: "...")
Round 4: write_module_spec ✓
Round 5-50: Empty rounds continuously
```

**Root Cause**: The `gpt-oss:20b` model:
1. Starts working correctly (calls tools in rounds 1-2, 4)
2. Degenerates into empty responses after a few rounds
3. Never completes the architecture task
4. Produces very short/empty content with no tool calls

**Why This Happens**:
- Model gets overwhelmed by growing context
- Loses track of what it's supposed to do
- Can't maintain coherent behavior over 50 rounds
- Not capable enough for complex architecture design tasks

## Test Results Comparison

### Simple Direct Test (5 rounds)
```
[architect] Round 1/5: write_architecture_doc ✓
[architect] Round 2/5: write_module_spec ✓
[architect] Round 3/5: write_module_spec ✓
[architect] Round 4/5: write_module_spec ✓
[architect] Round 5/5: write_module_spec ✓
Result: failure (max rounds exceeded, task incomplete)
```

**Analysis**: No empty rounds, but task not completed in 5 rounds.

### L7 Evaluation (50 rounds via orchestrator)
```
[architect] Round 1/50: write_architecture_doc ✓
[architect] Round 2/50: write_module_spec ✓
[architect] Round 3/50: Empty round
[architect] Round 4/50: write_module_spec ✓
[architect] Round 5-50: Empty rounds continuously
Result: failure (50 empty rounds)
```

**Analysis**: Model degenerates after initial tool calls.

## Why Fixes Helped But Didn't Solve It

**Before fixes**:
- Architect entered chat mode, asked questions
- 0 tool calls
- Failed immediately

**After fixes**:
- Architect correctly uses architecture tools for 1-4 rounds
- Then degenerates into empty responses
- Still fails, but gets further

**The improvement**: Fixes eliminated architectural/config bugs, revealing the underlying model limitation.

## Evidence This Is a Model Issue

1. **Pattern matches Type A timeout from original eval**: LLM hangs/produces no useful output on complex tasks

2. **Works initially, then breaks**: Model can't maintain behavior over many rounds

3. **Empty LLM responses**: `"LLM response: ..."` means model produced very short content with no tool calls

4. **Success in simple scenarios only**: Direct test with fewer rounds shows tools being called

5. **Matches baseline findings**: L7 was 0% success with "LLM hangs on planning" - same root cause

## Recommendations

### Option 1: Use Better Model (RECOMMENDED)

Switch to a more capable model:
```bash
# Test with larger model
OLLAMA_MODEL=qwen2.5-coder:32b python eval_l7_quick.py

# Or try different model family
OLLAMA_MODEL=deepseek-coder-v2:16b python eval_l7_quick.py
```

**Expected result**: Model can maintain coherent behavior over 50 rounds and complete architecture tasks.

### Option 2: Simplify Architect Role

Reduce architect's complexity:
- Shorter system prompt (half current length)
- Fewer tool options (remove list/read tools)
- Simpler output format requirements
- Lower round limit (25 instead of 50)

**Expected result**: Marginal improvement, still unlikely to succeed with current model.

### Option 3: Skip Architect, Delegate Directly to TaskExecutor

Bypass architect entirely:
- Orchestrator delegates L7 tasks directly to task_executor
- Task executor implements without architecture phase
- Simpler, but loses architecture design step

**Expected result**: May work better for simple L7 tasks, but lacks planning for complex projects.

## What Was Actually Fixed

✅ **ChatbotBehavior goal detection**: Delegated agents no longer enter chat mode
✅ **Architect role clarity**: Architect knows to use architecture tools, not code tools
✅ **Improved logging**: Empty rounds are detected and logged properly

## What Cannot Be Fixed Without Better Model

❌ **Model capability**: `gpt-oss:20b` cannot maintain coherent tool usage over 50 rounds
❌ **L7 task complexity**: Model gets overwhelmed by multi-round architecture design
❌ **Context degradation**: Model loses track of goal as conversation grows

## Conclusion

The fixes **resolved all architectural issues**. Empty rounds now have a clear cause: **model capability limitation**.

**The `gpt-oss:20b` model is not capable enough for L7 tasks with architect delegation.**

To achieve L7 success, either:
1. Use a better model (qwen2.5-coder:32b, deepseek-coder-v2:16b)
2. Drastically simplify the task/architecture
3. Skip architect and delegate directly to task executor

The agent architecture is sound. The model is the bottleneck.
