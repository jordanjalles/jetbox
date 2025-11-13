# L3-L7 Orchestrator Evaluation - Failure Analysis

**Evaluation Date:** 2025-11-13
**Context Fixes Applied:** Removed duplicate GOAL, simplified mode explanations (-2500 chars)
**Tasks Completed:** 10/26 (stopped for analysis after exceeding failure threshold)

## Summary Statistics

- **Successes:** 3/10 (30%)
- **True Failures:** 6/10 (60%) - Agent didn't complete work correctly
- **Validation Bugs:** 1/10 (10%) - Test command syntax issues

## Key Finding: Context Fixes WORK

✅ **No timeouts** (was 100% timeout rate before fixes)
✅ **No crashes** (all agents exit code 0)
✅ **Some successes** (binary_search, csv_processor, async_downloader)

The ~2500 char context pollution fix successfully eliminated the timeout plague.

## Failure Patterns

### Pattern 1: Premature Completion (3 failures)

Agents exit quickly (14-195s) with empty workspace:

| Task | Duration | Files Created | Agent Behavior |
|------|----------|---------------|----------------|
| linked_list | 14.0s | 0 | Exited immediately |
| json_parser | 20.8s | 0 | Exited immediately |
| bubble_sort | 195.0s | 0 | Ran 3 min then exited |

**Root Cause:** Agents calling mark_complete() without creating required files.

**Hypothesis:** Completion nudging too aggressive OR agents misunderstanding goal.

### Pattern 2: Wrong Implementation (3 failures)

Agents work extensively but produce broken/incomplete code:

| Task | Duration | Issue | Details |
|------|----------|-------|---------|
| rest_api_mock | 541.7s | Wrong filename | Created app.py instead of api.py |
| sqlite_manager | 258.6s | Buggy code | AttributeError in create_table() |
| test_framework_basic | 335.0s | Missing method | TestRunner lacks run() method |

**Root Cause:** Agents not verifying requirements OR not testing their implementations.

### Pattern 3: Validation Bug (1 failure)

Test command has syntax error:

| Task | Duration | Issue |
|------|----------|-------|
| cache_decorator | 219.2s | Multiline decorator in `python -c` |

**Note:** Not an agent failure - test command bug.

## Success Pattern

Tasks that succeeded had:
- 160-270s duration (reasonable work time)
- Multiple files created (code + tests)
- Tests executed (__pycache__ present)
- Proper verification

**Examples:**
- binary_search: 164s, created search.py + tests
- csv_processor: 173s, created csv_utils.py + 2 test files
- async_downloader: 274s, created downloader.py

## Deep Dive: Premature Completion

### Case: linked_list (14.0s)

```
Workspace: /tmp/orch_L3_linked_list_2bui882a
Expected: linked_list.py with LinkedList class
Actual: Empty workspace (only .agent_context/ dir)
Notes file: Not created
```

Agent exited successfully but did nothing. No context dumps available (agent.py output not captured).

### Case: rest_api_mock (541.7s)

```
Workspace: /tmp/orch_L4_rest_api_mock_o7n4ozu9
Expected: api.py
Actual: app.py, test_app.py, README.md, requirements.txt (6 files)
```

Agent did substantial work (9 min) but used wrong filename. Shows agent understood task but not filename requirement.

## Recommended Next Steps

1. **Investigate premature completion:**
   - Check orchestrator delegation context
   - Verify mark_complete tool description clarity
   - Review completion nudging logic

2. **Check goal propagation:**
   - Are sub-agents receiving full requirements?
   - Is delegation losing details (e.g., filename api.py)?

3. **Add verification requirements:**
   - System prompt: "Verify files exist before mark_complete"
   - Add list_dir as completion check

4. **Context inspection:**
   - Dump orchestrator context when delegating
   - Check if sub-agent receives complete goal

## Test Environment Notes

- Model: qwen3-coder:30b (default)
- Team: orchestrator → architect + task_executor
- Timeouts: L3=8min, L4=10min (agents completing faster than timeout)
- Workspace: /tmp/orch_* (isolated per task)
