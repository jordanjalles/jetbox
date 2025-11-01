# TaskExecutor Idle Loop Debugging Report

**Date:** 2025-10-31
**Issue:** TaskExecutor with SubAgentStrategy gets stuck in idle loop, never calls mark_complete/mark_failed
**Test:** L8 Web Application (Orchestrator → Architect → TaskExecutor integration)

## Problem Statement

During the orchestrator-architect-executor integration test, the TaskExecutor received a delegated task (FastAPI backend implementation) but got stuck:

- Ran for 13+ rounds showing "AGENT STATUS: 💤 idle"
- Made 12 LLM calls but only 12 tool executions (minimal progress)
- Never called `mark_complete` or `mark_failed` to report back to orchestrator
- Context grew to 60K+ tokens without making meaningful progress
- Eventually hit timeout (180s)

## Root Cause Analysis

### 1. Code Flow Traced

**Orchestrator delegation:**
```
orchestrator_main.py:451-466
→ subprocess.run([python, agent.py, task_description])
```

**agent.py creates TaskExecutor:**
```python
# agent.py:78-84
executor = TaskExecutorAgent(
    workspace=workspace,
    goal=args.goal,
    ...
)
# NO context_strategy parameter → uses default
```

**TaskExecutor default strategy:**
```python
# task_executor_agent.py:72-74
# Context strategy (default to SubAgentStrategy for delegated work)
self.context_strategy = context_strategy or SubAgentStrategy()
```

**SubAgentStrategy provides completion tools:**
```python
# context_strategies.py:975-995
- mark_complete(summary)
- mark_failed(reason)
```

### 2. Hypothesis: Why LLM Doesn't Call Tools

After analyzing the SubAgentStrategy implementation (context_strategies.py:746-995), I identified potential issues:

#### Issue #1: Ambiguous Completion Criteria

The SubAgentStrategy instructions (line 944-966) say:
```
IMPORTANT - YOU MUST SIGNAL COMPLETION:
- When work is done and tests pass: call mark_complete(summary="what you accomplished")
- If you cannot complete the task: call mark_failed(reason="why it failed")
- DO NOT just stop - always call one of these tools to report results
```

**Problem:** For a complex delegated task like "Implement FastAPI backend with CRUD endpoints and JWT authentication", the LLM may not know when it's "done" because:
- The task description is high-level
- No clear acceptance criteria
- LLM may keep trying to add more features
- No explicit "you've done enough" signal

#### Issue #2: Default System Prompt Conflict

The TaskExecutor uses a generic system prompt from `config.llm.system_prompt` that says:
```
WORKFLOW:
Your goal is shown at the start of the conversation. Simply complete the work using the available tools.

Work directly on the goal - no need to decompose into subtasks.
When all work is complete and tests pass, call mark_goal_complete().
```

**Problem:** The system prompt mentions `mark_goal_complete()` but SubAgentStrategy provides `mark_complete()` and `mark_failed()`. This creates confusion about which tool to call.

Looking at task_executor_agent.py:147-179, the system prompt is built as:
```python
base_prompt = config.llm.system_prompt
+ strategy_instructions  # SubAgentStrategy instructions
```

So the LLM sees BOTH:
- "When all work is complete and tests pass, call mark_goal_complete()"
- "When work is done and tests pass: call mark_complete(summary=...)"

#### Issue #3: No Progress Milestones

For complex tasks, the LLM needs intermediate progress signals. SubAgentStrategy doesn't provide any mechanism for:
- Reporting partial progress
- Getting feedback that work is acceptable so far
- Understanding what "minimum viable" completion looks like

The LLM may be stuck trying to achieve perfection instead of reporting reasonable progress.

#### Issue #4: Context Growth Without Compaction

SubAgentStrategy appends all messages until hitting 75% of token limit (128K tokens). At 60K tokens:
- Context is not yet triggering compaction (would need ~96K)
- LLM is seeing lots of historical context
- May be confusing previous attempts with current state
- No clear signal about current status vs. history

### 3. Evidence from Timeout Dumps

I examined `.agent_context/timeout_dumps/timeout_inactivity_20251031_213933.json`:

**Observation:** This dump shows a DIFFERENT task (CLI calculator), and the agent timed out after writing just ONE file (calculator.py). It got an inactivity timeout (30.68 seconds) - the LLM hung without responding.

**Implication:** The issue may not just be about calling mark_complete/mark_failed. The LLM might be:
1. Getting stuck generating responses
2. Entering infinite reasoning loops
3. Trying to generate perfect code and never finishing

## Identified Issues Summary

### Confirmed Issues:

1. **Tool naming inconsistency** - System prompt says `mark_goal_complete()` but SubAgentStrategy provides `mark_complete()` and `mark_failed()`

2. **Unclear completion criteria** - No explicit acceptance criteria or "minimum viable" definition for delegated tasks

3. **LLM inactivity timeouts** - LLM sometimes hangs and doesn't generate output (inactivity timeout)

4. **No progress checkpoints** - Complex tasks need intermediate signals, but SubAgentStrategy only has final completion tools

### Potential Contributing Factors:

5. **Context growth** - 60K tokens without compaction may confuse LLM about current state

6. **Model quality** - gpt-oss:20b may struggle with complex reasoning about task completion

7. **Lack of reflection** - No mechanism for LLM to assess "is this good enough?" before reporting completion

## Recommended Fixes

### Immediate Fixes (High Priority):

#### Fix #1: Align Tool Names
**File:** `agent_config.yaml` or base system prompt
**Change:** Update generic system prompt to NOT mention `mark_goal_complete()` when using SubAgentStrategy. Let the strategy instructions handle completion signaling.

**Implementation:**
```python
# task_executor_agent.py: get_system_prompt()
# Don't include completion instructions in base prompt
# Let strategy provide them
```

#### Fix #2: Add Explicit Completion Nudge
**File:** `task_executor_agent.py`
**Change:** After N rounds without calling completion tools, inject a user message forcing status report.

**Implementation:**
```python
# In TaskExecutor.run() loop
if self.state.total_rounds % 5 == 0:  # Every 5 rounds
    if not has_called_completion_tools():
        nudge = {
            "role": "user",
            "content": "⚠️ Progress check: Have you made meaningful progress on the task? If yes, call mark_complete() with what you've accomplished so far. If blocked, call mark_failed()."
        }
        messages.append(nudge)
```

#### Fix #3: Shorter Inactivity Timeout
**File:** `task_executor_agent.py`
**Current:** inactivity_timeout=30s
**Change:** Reduce to 15s and handle timeout more aggressively

**Implementation:**
```python
# task_executor_agent.py:558
inactivity_timeout=15,  # Catch hung LLM faster
```

### Medium Priority Fixes:

#### Fix #4: Earlier Context Compaction
**File:** `context_strategies.py:SubAgentStrategy`
**Change:** Trigger compaction at 50% of max tokens instead of 75%

**Rationale:** Forces earlier summarization, keeps context focused

#### Fix #5: Add Progress Reporting Tool
**File:** `context_strategies.py:SubAgentStrategy`
**Change:** Add `report_progress(status, next_steps)` tool for intermediate updates

**Benefit:** Gives LLM a way to communicate without signaling final completion

### Lower Priority (Architectural):

#### Fix #6: Task Acceptance Criteria
**File:** `orchestrator_main.py`
**Change:** When delegating, explicitly state acceptance criteria

**Example:**
```
Task: Implement FastAPI backend with CRUD endpoints and JWT authentication

Acceptance Criteria:
- At least 3 CRUD endpoints implemented
- JWT middleware present
- Basic tests pass
- No syntax errors

You don't need to be perfect - just meet these criteria and call mark_complete().
```

#### Fix #7: Add Reflection Step
**File:** `task_executor_agent.py`
**Change:** Before allowing mark_complete, ask LLM to self-assess

**Implementation:**
```python
if "mark_complete" in tool_calls:
    # Inject reflection question
    reflection = {
        "role": "user",
        "content": "Before marking complete, confirm: Have you met the core requirements? List what you accomplished."
    }
```

## Next Steps

1. ✅ Document findings (this file)
2. ⬜ Implement Fix #1 (tool name alignment)
3. ⬜ Implement Fix #2 (completion nudge)
4. ⬜ Implement Fix #3 (shorter inactivity timeout)
5. ⬜ Test with L8 task again
6. ⬜ If still failing, implement medium priority fixes

## Test Plan

After implementing fixes:

1. Run L5 test (simple task) - should complete quickly
2. Run L6 test (medium task) - should complete with nudge
3. Run L8 test without architect - should delegate and complete
4. Run L8 test with architect - full integration test

Success criteria:
- TaskExecutor calls mark_complete or mark_failed within 20 rounds
- No idle loops lasting more than 3 rounds
- Context stays under 40K tokens
- Completion rate >80% for L5-L6, >60% for L7-L8

---

**Status:** Analysis complete, ready to implement fixes
**Next:** Start with high-priority fixes (#1-#3)
