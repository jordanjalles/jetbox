# L5 Task Timeout Analysis & Fix Proposals

## Root Cause Analysis

### The Timeout Pattern

**L5 Task Flow** (15-minute total timeout):
1. **Orchestrator round 1** (~30s): Calls `consult_architect`
2. **Architect runs** (~5-7 min): Creates 5-6 detailed architecture docs
3. **Orchestrator round 2** (~30s): Calls `delegate_to_executor`
4. **task_executor runs** (~7-8 min remaining):
   - Round 1-2: Lists directories (~30s)
   - Round 3-7: Reads architecture files, ONE AT A TIME (~5-7 min)
   - **TIMEOUT before writing any code**

### The Time-Box Bug

**Configuration**:
```yaml
# config/agents/task_executor.yaml
- type: TimeBoxBehavior
  params:
    total_budget_minutes: 60  # ← BUG: Should be 15!
    default_nudges: [25, 50, 75]
```

**Actual nudge times**:
- 25% of 60 min = **15 minutes** ← Nudge fires EXACTLY when timeout hits!
- 50% of 60 min = 30 minutes (never reached)
- 75% of 60 min = 45 minutes (never reached)

**Result**: task_executor has ZERO time pressure before timeout.

### Evidence from Snapshots

**task_executor behavior** (blog_system):
- Round 1: `list_dir(".")` - checking workspace
- Round 2: `list_dir("architecture")` - checking docs
- Round 3: `read_file("blog-system-architecture.md")` - reading main doc
- Round 4: `read_file("modules/post-model.md")` - reading module doc
- Round 5: `read_file("modules/comment-model.md")` - reading another module
- Round 6: `read_file("modules/json-persistence.md")` - still reading
- Round 7: `read_file("modules/blog-manager.md")` - still reading
- **TIMEOUT** - never called `write_file` once

## Comprehensive Fix Proposals

### 🔴 CRITICAL: Immediate Fixes (Stop the bleeding)

#### Fix 1A: Correct Default Time Budget
**Problem**: 60-minute budget is 4x too long for delegation context
**Solution**: Change default to match typical delegation timeout

```yaml
# config/agents/task_executor.yaml
- type: TimeBoxBehavior
  params:
    total_budget_minutes: 15  # Changed from 60
    default_nudges: [25, 50, 75]
```

**Nudges become**:
- 25% of 15 min = **3.75 minutes** ← Early warning!
- 50% of 15 min = **7.5 minutes** ← Halfway point pressure
- 75% of 15 min = **11.25 minutes** ← Final warning

**Impact**: ✅ Immediate, ✅ Zero code changes, ⚠️ Still assumes 15-min timeout

---

#### Fix 1B: More Aggressive Nudge Schedule
**Problem**: Even with correct budget, 3 nudges might not be enough
**Solution**: Add more frequent check-ins

```yaml
- type: TimeBoxBehavior
  params:
    total_budget_minutes: 15
    default_nudges: [20, 40, 60, 80]  # Changed from [25, 50, 75]
```

**Nudges become**:
- 20% = 3 minutes - "Getting started, verify workspace"
- 40% = 6 minutes - "Time to start implementing"
- 60% = 9 minutes - "Midpoint, should have files created"
- 80% = 12 minutes - "Final push, wrap up and test"

**Impact**: ✅ Better pacing, ✅ Simple change, ⚠️ May add noise

---

### 🟡 HIGH PRIORITY: Behavioral Fixes (Change agent behavior)

#### Fix 2: task_executor System Prompt Enhancement
**Problem**: "Verify first" causes over-reading when architecture exists
**Solution**: Add architecture-aware guidance to system prompt

```python
# In task_executor system prompt, add section:

## Working with Architecture

If architecture documentation exists in the workspace:
1. **Read strategically, not exhaustively**:
   - Read the MAIN architecture doc only (e.g., `architecture/*.md` in root)
   - Start implementing immediately from that design
   - Refer back to module docs ONLY when you need specific details

2. **Prefer action over analysis**:
   - After reading 1-2 docs, START WRITING CODE
   - Don't read all module docs upfront - implement and refer as needed
   - Architecture docs are REFERENCE material, not checklist items

3. **Time-aware reading**:
   - If you've spent >3 rounds just reading, START IMPLEMENTING
   - Code can always be refined; perfect understanding blocks progress
```

**Impact**: ✅ Addresses root cause, ✅ No code changes, ⚠️ Relies on prompt following

---

#### Fix 3: Reading Loop Detection
**Problem**: Agent doesn't realize it's stuck in read-only mode
**Solution**: Add loop detection specifically for read-heavy patterns

```python
# In LoopDetectionBehavior or new ReadingLoopBehavior:

def detect_reading_loop(self, recent_tools: list[str], round_num: int) -> str | None:
    """Detect when agent is stuck reading files instead of implementing."""

    read_tools = ['read_file', 'list_dir', 'run_bash cat', 'run_bash head']
    write_tools = ['write_file', 'run_bash', 'mark_complete']

    # Count read vs write tools in last N rounds
    recent_reads = sum(1 for tool in recent_tools[-5:] if any(r in tool for r in read_tools))
    recent_writes = sum(1 for tool in recent_tools[-5:] if any(w in tool for w in write_tools))

    # If 5+ consecutive rounds of mostly reading, nudge
    if recent_reads >= 4 and recent_writes == 0 and round_num >= 5:
        return (
            "⚠️  READING LOOP DETECTED\n"
            "You've spent 5 rounds reading files without writing any code.\n"
            "Architecture docs are for reference - you don't need to read them all.\n"
            "START IMPLEMENTING NOW. You can refer back to docs as needed."
        )

    return None
```

**Impact**: ✅ Directly addresses pattern, ✅ Reusable, ⚠️ Requires behavior code change

---

### 🟢 MEDIUM PRIORITY: Infrastructure Fixes (Proper solution)

#### Fix 4: Dynamic Time Budget from Subprocess Timeout
**Problem**: task_executor doesn't know its actual subprocess timeout
**Solution**: Pass timeout through delegation call

**Implementation**:

```python
# In behaviors/delegation.py, modify _generic_subprocess_delegation:

def _generic_subprocess_delegation(...):
    # Calculate remaining time budget for subprocess
    if hasattr(calling_agent, '_start_time'):
        elapsed = time.time() - calling_agent._start_time
        budget_minutes = calling_agent.config.get('timeout_minutes', 60)
        remaining_minutes = max(5, budget_minutes - (elapsed / 60))
    else:
        remaining_minutes = 15  # Conservative default

    # Pass as environment variable
    env = os.environ.copy()
    env['JETBOX_TIMEOUT_MINUTES'] = str(int(remaining_minutes))

    subprocess.run([...], env=env)
```

```python
# In behaviors/time_box.py __init__:

def __init__(self, total_budget_minutes: int | None = None, ...):
    # Check for dynamic timeout from environment
    env_timeout = os.getenv('JETBOX_TIMEOUT_MINUTES')
    if env_timeout:
        self.budget_minutes = int(env_timeout)
    else:
        self.budget_minutes = total_budget_minutes
```

**Impact**: ✅ Correct for ANY timeout, ✅ Works with delegation, ⚠️ Requires infra changes

---

### 🔵 LOW PRIORITY: Optimizations (Reduce time spent)

#### Fix 5: Architect Output Limits
**Problem**: Architect creates 5-6 detailed docs that all need reading
**Solution**: Limit architect to single comprehensive doc

```yaml
# In architect system prompt:
- Create ONE main architecture.md file with all design decisions
- Don't create separate module files unless explicitly required
- Keep architecture concise: 200-300 lines maximum
```

**Impact**: ✅ Less reading required, ⚠️ May reduce design clarity, ⚠️ Doesn't fix time awareness

---

## Recommended Implementation Plan

### Phase 1: Emergency Patch (< 5 minutes)
1. ✅ **Fix 1A**: Change task_executor budget: 60 → 15 minutes
2. ✅ **Fix 1B**: Change nudges: [25, 50, 75] → [20, 40, 60, 80]

**Expected improvement**: L5 success rate 0% → 20-30%

---

### Phase 2: Behavioral Improvements (< 30 minutes)
3. ✅ **Fix 2**: Add architecture-aware guidance to task_executor prompt
4. ✅ **Fix 3**: Add reading loop detection behavior

**Expected improvement**: L5 success rate 20-30% → 40-50%

---

### Phase 3: Proper Fix (< 2 hours)
5. ✅ **Fix 4**: Dynamic timeout passing through delegation
6. ✅ **Fix 5**: Architect output optimization

**Expected improvement**: L5 success rate 40-50% → 60-70%

---

## Testing Plan

After each phase, run:
```bash
python tests/orchestrator_l3_l7_eval.py
# Focus on L5 tasks: blog_system, todo_app, inventory_system
```

**Success metrics**:
- Phase 1: At least 1/5 L5 tasks pass (currently 0/2)
- Phase 2: At least 2/5 L5 tasks pass
- Phase 3: At least 3/5 L5 tasks pass (60% target)

---

## Why This Matters

**Current state**:
- L3 tasks: 83% success (simple, direct implementation)
- L5 tasks: 0% success (orchestrator workflow hits timeout)

**The gap**: Not capability, but **pacing and time awareness**.

The LLM CAN implement these tasks, but it:
1. Doesn't know it's running out of time (wrong budget)
2. Over-prepares by reading all docs (no reading loop detection)
3. Never transitions from reading to writing (no phase awareness)

**With these fixes**: L5 tasks become tractable. The orchestrator workflow (architect + task_executor) can actually deliver results within realistic timeouts.
