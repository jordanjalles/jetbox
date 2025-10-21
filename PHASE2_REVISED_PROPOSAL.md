# Phase 2 Revised: Hierarchical Task Decomposition on MAX_ROUNDS

## Philosophy

Instead of **increasing MAX_ROUNDS** when tasks get complex, we should:
1. **Lower MAX_ROUNDS** (e.g., 16-20 instead of 24) to force faster iteration
2. When hitting the limit, **don't fail** - instead **escalate** and decide:
   - **Break into smaller subtasks** (zoom in)
   - **Reconsider approach** (zoom out to parent task/goal level)

This aligns with the existing hierarchical context: **Goal → Task → Subtask → Action**

## Core Insight

The current system already has hierarchy. When an agent gets stuck:
- At **Action** level → Escalate to **Subtask** (try different approach)
- At **Subtask** level → Escalate to **Task** (break into smaller pieces)
- At **Task** level → Escalate to **Goal** (reconsider overall strategy)

## Proposed Changes

### 1. Lower MAX_ROUNDS and Add Escalation Trigger

**File:** `agent.py`

**Current:**
```python
MAX_ROUNDS = 24
```

**Proposed:**
```python
MAX_ROUNDS = 16  # Lower limit to force faster decisions
MAX_ROUNDS_PER_SUBTASK = 8  # Even tighter limit per subtask
```

### 2. Add Escalation Logic on Round Limit

**File:** `agent.py` (main loop)

**Add before the "Hit MAX_ROUNDS" exit:**

```python
# In main loop, around line 450-460
if round_no >= MAX_ROUNDS:
    log(f"Hit MAX_ROUNDS ({MAX_ROUNDS}). Attempting escalation...")

    # Try to escalate before giving up
    escalation_result = ctx.escalate_on_stuck(
        reason="max_rounds",
        state=probe_state_generic()
    )

    if escalation_result["action"] == "decompose":
        log(f"Escalation: Breaking current task into smaller subtasks")
        # Continue with new subtasks
        round_no = 0  # Reset counter for new approach
        continue

    elif escalation_result["action"] == "zoom_out":
        log(f"Escalation: Reconsidering approach at {escalation_result['level']} level")
        # Move up hierarchy and try different approach
        round_no = 0
        continue

    elif escalation_result["action"] == "give_up":
        # Only give up if escalation also fails
        log("Escalation failed. Giving up.")
        break
```

### 3. Implement Escalation Logic in ContextManager

**File:** `context_manager.py`

**Add new method to ContextManager class:**

```python
def escalate_on_stuck(self, reason: str, state: dict[str, Any]) -> dict[str, str]:
    """
    When stuck (loops, max rounds, repeated failures), escalate up the hierarchy.

    Strategy:
    1. Stuck on Action → Try different subtask approach
    2. Stuck on Subtask → Break into smaller subtasks OR mark blocked and move to next
    3. Stuck on Task → Reconsider task decomposition, try different strategy
    4. Stuck on Goal → Give up (can't escalate higher)

    Returns:
        dict with "action" (decompose|zoom_out|give_up) and "level" (subtask|task|goal)
    """
    if not self.state.goal:
        return {"action": "give_up", "level": "none", "reason": "No active goal"}

    current_task = self._get_current_task()
    if not current_task:
        return {"action": "give_up", "level": "goal", "reason": "No active task"}

    current_subtask = current_task.active_subtask()

    # Determine current stuck level
    if current_subtask:
        # Stuck at subtask level
        action_count = len(current_subtask.actions)

        if action_count < 3:
            # Not enough attempts yet, decompose into smaller steps
            return self._decompose_subtask(current_subtask, current_task, reason)
        else:
            # Too many attempts on this subtask, zoom out to task level
            return self._zoom_out_from_subtask(current_subtask, current_task, reason, state)

    elif current_task.subtasks:
        # Stuck at task level (has subtasks but none active)
        return self._reconsider_task_approach(current_task, reason, state)

    else:
        # Stuck at goal level (no tasks or all blocked)
        return self._reconsider_goal_approach(reason, state)

def _decompose_subtask(self, subtask: Subtask, task: Task, reason: str) -> dict[str, str]:
    """Break current subtask into smaller pieces."""
    # Mark current subtask as "needs_decomposition"
    subtask.status = "blocked"
    subtask.failure_reason = f"Decomposing due to {reason}"

    # The agent will see this in context and should propose smaller subtasks
    # We don't auto-generate them - the LLM should decide how to break it down

    self._save_state()
    return {
        "action": "decompose",
        "level": "subtask",
        "reason": f"Subtask '{subtask.description}' needs smaller steps",
        "blocked_subtask": subtask.description
    }

def _zoom_out_from_subtask(
    self, subtask: Subtask, task: Task, reason: str, state: dict[str, Any]
) -> dict[str, str]:
    """Zoom out from stuck subtask to task level, try different approach."""
    # Mark subtask as failed/blocked
    subtask.status = "blocked"
    subtask.failure_reason = f"Failed: {reason} after {len(subtask.actions)} attempts"

    # Check if there are other pending subtasks we can try
    next_subtask = task.next_pending_subtask()
    if next_subtask:
        # Move to next subtask
        next_subtask.status = "in_progress"
        self._save_state()
        return {
            "action": "zoom_out",
            "level": "task",
            "reason": f"Skipping blocked subtask, trying next approach",
            "next_subtask": next_subtask.description
        }

    # No more subtasks - need to reconsider task approach
    return self._reconsider_task_approach(task, reason, state)

def _reconsider_task_approach(self, task: Task, reason: str, state: dict[str, Any]) -> dict[str, str]:
    """Reconsider task-level strategy when all subtasks blocked."""
    # Look at what we've accomplished so far
    completed_subtasks = [st for st in task.subtasks if st.status == "completed"]
    blocked_subtasks = [st for st in task.subtasks if st.status == "blocked"]

    # If we have some progress, maybe we're actually closer than we think
    if len(completed_subtasks) > len(blocked_subtasks):
        # We've made good progress - try verification
        if state.get("files_exist") and not state.get("recent_errors"):
            # Files exist, no errors - maybe we're done?
            return {
                "action": "zoom_out",
                "level": "task",
                "reason": "Verify if task is actually complete despite blocked subtasks",
                "suggestion": "run_verification"
            }

    # Clear all subtasks and let LLM propose new approach
    task.subtasks = []
    task.status = "in_progress"
    self._save_state()

    return {
        "action": "zoom_out",
        "level": "task",
        "reason": f"Clearing {len(blocked_subtasks)} blocked subtasks, reconsidering task approach",
        "files_exist": state.get("files_exist", []),
        "recent_errors": state.get("recent_errors", [])
    }

def _reconsider_goal_approach(self, reason: str, state: dict[str, Any]) -> dict[str, str]:
    """Final escalation - reconsider goal-level strategy."""
    # At this point, we've exhausted task-level strategies
    # Check if we've actually made progress toward goal

    if not self.state.goal:
        return {"action": "give_up", "level": "goal"}

    completed_tasks = [t for t in self.state.goal.tasks if t.status == "completed"]

    if completed_tasks:
        # We've completed some tasks - maybe goal is partially achieved?
        return {
            "action": "zoom_out",
            "level": "goal",
            "reason": f"Reconsidering goal strategy. {len(completed_tasks)} tasks completed.",
            "suggestion": "verify_partial_completion_or_new_strategy"
        }

    # No progress at all - give up
    return {
        "action": "give_up",
        "level": "goal",
        "reason": f"No progress after escalation. {reason}"
    }
```

### 4. Add Escalation Context to LLM Prompt

**File:** `agent.py` (build_context function)

**Add escalation info to context when it happens:**

```python
def build_context(ctx: ContextManager, user_goal: str, escalation: dict | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ... existing context building ...

    # If we're in escalation mode, add special guidance
    if escalation and escalation.get("action") != "give_up":
        escalation_prompt = generate_escalation_prompt(escalation)
        messages.append({
            "role": "system",
            "content": escalation_prompt
        })

    return messages

def generate_escalation_prompt(escalation: dict) -> str:
    """Generate LLM prompt for escalation scenarios."""

    if escalation["action"] == "decompose":
        return f"""
ESCALATION: Task Decomposition Required

The current subtask is stuck: {escalation.get('blocked_subtask', 'unknown')}
Reason: {escalation['reason']}

Please:
1. Break this subtask into 2-4 smaller, more concrete steps
2. Each step should be achievable in 1-3 actions
3. Propose the smaller steps using mark_subtask_complete/start pattern

Example:
- Current stuck subtask: "Create package with tests"
- Break into:
  1. "Create package __init__.py"
  2. "Create core module file"
  3. "Create tests directory and test file"
  4. "Run pytest to verify"
"""

    elif escalation["action"] == "zoom_out":
        level = escalation.get("level", "task")

        if level == "subtask":
            return f"""
ESCALATION: Moving to Next Subtask

Previous subtask blocked: {escalation.get('reason', 'unknown')}
Moving to: {escalation.get('next_subtask', 'next available subtask')}

Try a different approach for this subtask.
"""

        elif level == "task":
            files_exist = escalation.get("files_exist", [])
            errors = escalation.get("recent_errors", [])

            if escalation.get("suggestion") == "run_verification":
                return f"""
ESCALATION: Verify Task Completion

Some subtasks blocked, but progress made:
- Files created: {', '.join(files_exist) if files_exist else 'none'}
- Recent errors: {', '.join(errors) if errors else 'none'}

Before reconsidering approach:
1. Run verification (pytest, ruff, or manual checks)
2. If verification passes, mark task complete
3. If verification fails, identify the specific gap and create ONE focused subtask to fix it
"""

            else:
                return f"""
ESCALATION: Reconsidering Task Approach

Previous approach had {len(escalation.get('files_exist', []))} files but got stuck.
Errors: {escalation.get('recent_errors', [])}

Please:
1. Review what's been accomplished so far
2. Identify the simplest path forward
3. Propose a NEW set of 2-3 subtasks with a different strategy
4. Avoid repeating the same actions that failed before
"""

        elif level == "goal":
            return f"""
ESCALATION: Reconsidering Goal Strategy

{escalation['reason']}

Please:
1. Review the original goal
2. Check if partial progress is acceptable
3. If not, propose a completely different high-level approach
4. Consider simplifying the goal or breaking it into stages
"""

    return ""
```

### 5. Track Subtask-Level Round Counts

**File:** `agent.py`

**Add tracking for rounds per subtask:**

```python
# At top of main loop
subtask_rounds = 0  # Track rounds for current subtask

# In loop:
current_subtask_sig = None
if ctx.state.goal:
    task = ctx._get_current_task()
    if task:
        subtask = task.active_subtask()
        if subtask:
            current_subtask_sig = subtask.signature()

# If subtask changed, reset counter
if current_subtask_sig != last_subtask_sig:
    subtask_rounds = 0
    last_subtask_sig = current_subtask_sig
else:
    subtask_rounds += 1

# Check subtask round limit
if subtask_rounds >= MAX_ROUNDS_PER_SUBTASK:
    log(f"Subtask hit {MAX_ROUNDS_PER_SUBTASK} rounds. Escalating...")
    escalation_result = ctx.escalate_on_stuck(
        reason="subtask_max_rounds",
        state=probe_state_generic()
    )
    # ... handle escalation ...
    subtask_rounds = 0
```

## Benefits of This Approach

1. **Forces better task decomposition** - Agent learns to break things smaller
2. **Prevents infinite loops** - Automatic escalation when stuck
3. **Uses existing hierarchy** - Leverages Goal→Task→Subtask structure
4. **More efficient** - Lower round limit means faster iterations
5. **Self-correcting** - Agent can zoom out and try different approaches
6. **Matches cognitive problem-solving** - How humans tackle stuck problems

## Example Escalation Flow

```
Round 1-8: Working on "Create mathx package with tests"
  → Creating files, writing tests

Round 9-16: Stuck - pytest failing with import errors
  → Trying different fixes, same error

Round 16: Hit MAX_ROUNDS
  → Escalate: Decompose subtask

Round 17-20: New approach - Smaller subtasks:
  1. ✓ Create mathx/__init__.py
  2. ✓ Create mathx/basic.py
  3. → Fix PYTHONPATH for tests (current)

Round 21-24: Still stuck on PYTHONPATH
  → Escalate: Zoom out to task level

Round 25-28: Reconsidering task approach
  → Verify what exists
  → Files are good, just import issue
  → Create one focused subtask: "Add sys.path.insert to test file"

Round 29: ✓ Task complete
```

## Integration with Phase 1

These changes work together:

**Phase 1** fixes infrastructure (PYTHONPATH, cleanup)
**Phase 2** adds intelligence when infrastructure isn't enough

The escalation system will kick in less often once Phase 1 is implemented, but provides a safety net for truly complex tasks.

## Testing the Escalation System

Add a new test case that forces escalation:

```python
# In run_stress_tests.py
{
    "id": "E1",
    "level": 2,
    "name": "Escalation Test: Deliberately Hard Task",
    "task": "Create a package with 5 modules, each with tests, all must pass pytest and ruff",
    "expected_files": [
        "pkg/__init__.py",
        "pkg/mod1.py", "pkg/mod2.py", "pkg/mod3.py", "pkg/mod4.py", "pkg/mod5.py",
        "tests/test_mod1.py", "tests/test_mod2.py", "tests/test_mod3.py",
        "tests/test_mod4.py", "tests/test_mod5.py"
    ],
    "timeout": 120,
    "verify_cmd": ["pytest", "-q"],
    "notes": "Should trigger escalation and decomposition"
}
```

Expected behavior:
1. Agent starts creating all files
2. Hits MAX_ROUNDS (16)
3. Escalates → Decomposes into "Create module 1", "Create module 2", etc.
4. Completes decomposed subtasks
5. Success
