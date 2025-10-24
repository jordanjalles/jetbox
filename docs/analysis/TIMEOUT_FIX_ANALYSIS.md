# Timeout Fix - Why Did Regressions Occur?

## The Problem We Tried to Fix

**Original issue**: Agent hung forever when Ollama was slow, causing tests to timeout with "0 rounds" (agent never started).

**Solution implemented**: Added 120-second timeout wrapper around `decompose_goal()` LLM call with fallback to generic task.

## What Actually Happened

### Results Summary

| Test | Before | After | Change |
|------|--------|-------|--------|
| L3-2 (Fix Buggy Code) | 70% | 80% | ✅ +10% |
| L4-1 (TodoList) | 70% | 90% | ✅ +20% |
| L3-3 (Add Feature) | 70% | 60% | ❌ -10% |
| L5-2 (Refactoring) | 70% | 33% | ❌ -37% |

### The Critical Pattern

**ALL timeout failures show "0 rounds"** - meaning:
1. Agent started successfully
2. `decompose_goal()` was called
3. **Something went wrong before Round 1**
4. Test framework timeout (240-360s) killed the process

## Root Cause: The Timeout Made Things Worse for Complex Tasks

### What We Thought Would Happen

```
Ollama slow → decompose_goal() hangs forever → timeout wrapper catches it
→ fallback creates simple task → agent continues working
```

### What Actually Happens

```
Complex task (L5-2) → decompose_goal() needs >120s to think
→ timeout wrapper KILLS the call prematurely
→ fallback returns: [{"description": goal, "subtasks": ["Complete the goal"]}]
→ agent.py tries to use this useless structure
→ agent gets stuck trying to figure out what "Complete the goal" means
→ test timeout (360s) kills process with 0 rounds
```

## The Real Problem: We Added a NEW Failure Mode

### Before Our Fix (Original Behavior)

**When Ollama is slow:**
- decompose_goal() waits forever for Ollama response
- Test times out after 240-360s
- Result: **Timeout, 0 rounds**

**Success rate:** 70% (7/10 passed)

### After Our Fix (Current Behavior)

**For simple tasks (L3-2, L4-1):**
- decompose_goal() completes within 120s ✅
- Agent works normally
- **Success rate improved**: 80-90%

**For complex tasks (L5-2):**
- decompose_goal() needs >120s to properly analyze complex requirement
- Our timeout CUTS IT OFF prematurely
- Fallback creates useless task structure: `["Complete the goal"]`
- Agent can't work with this vague instruction
- Agent hangs trying to understand what to do
- Test times out with 0 rounds
- **Success rate DESTROYED**: 33% (was 70%)

## Why L5-2 Is Hit So Hard

**Task**: "Refactor the entire mathx package to use a unified MathOperation base class"

This requires the LLM to:
1. Understand the existing mathx package structure
2. Design a base class architecture
3. Plan how each module needs to change
4. Break down into specific refactoring steps
5. Include testing and verification steps

**Ollama's gpt-oss:20b model needs >120 seconds for this analysis.**

When we cut it off, we get:
```json
[{"description": "Refactor the entire mathx package to use a unified MathOperation base class",
  "subtasks": ["Complete the goal"]}]
```

The agent literally doesn't know what "Complete the goal" means in concrete terms!

## The Fallback Is Broken

Current fallback (agent.py:949):
```python
return [{"description": goal, "subtasks": ["Complete the goal"]}]
```

This is **worse than useless** for complex tasks. The agent needs:
- Specific, actionable subtasks
- Clear file paths or targets
- Test/verification steps

"Complete the goal" provides NONE of this.

## Solution: Increase Timeout OR Remove It

### Option 1: Longer Timeout
Change 120s → 300s for decompose_goal()
- Pros: Gives complex tasks time to decompose properly
- Cons: Still fails if Ollama actually hangs

### Option 2: Remove Timeout Completely
Go back to no timeout on decompose_goal()
- Pros: Never cuts off legitimate thinking
- Cons: Hangs forever if Ollama is truly dead

### Option 3: Better Fallback (RECOMMENDED)
Keep 120s timeout BUT create intelligent fallback:
```python
# For complex tasks, create basic exploration structure
return [
    {"description": "Understand current code",
     "subtasks": ["Read relevant files", "Identify modules to change"]},
    {"description": "Implement changes",
     "subtasks": ["Make necessary modifications"]},
    {"description": "Verify",
     "subtasks": ["Run tests", "Check quality"]}
]
```

This gives the agent a fighting chance even when decomposition times out.

## Bottom Line

**The timeout wrapper works for simple tasks but BREAKS complex tasks.**

The real issue: **The fallback is too simplistic.**

We need either:
1. **Longer timeout** (300s instead of 120s)
2. **Smarter fallback** that creates workable task structure
3. **Adaptive timeout** based on task complexity
4. **No timeout on decompose_goal()** (accept the original hang risk)

The current 120s timeout + generic fallback is the worst of both worlds for complex tasks.
