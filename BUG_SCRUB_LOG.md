# Deep Bug Scrub Log

Started: 2025-11-10

## Testing Strategy

1. **Phase 1**: Simple CLI tests (--help, --list-teams, basic startup)
2. **Phase 2**: Solo agent - trivial tasks (single file, hello world)
3. **Phase 3**: Solo agent - simple tasks (calculator, basic tests)
4. **Phase 4**: Solo agent - moderate tasks (multi-file, tests, linting)
5. **Phase 5**: Orchestrator - simple delegation
6. **Phase 6**: Orchestrator - complex projects (architecture + implementation)
7. **Phase 7**: Test script validation (existing test suite)
8. **Phase 8**: Edge cases and error handling

## Bugs Found and Fixed

---

---

## Phase 3: Solo Agent - Simple Tasks (Calculator with Tests)

### Test 3.1: Calculator without --once (full run loop)
❌ FAIL - Agent never calls mark_complete, runs out of rounds

**Bug**: Goal never set, so mark_complete tool never available
**Root cause**:
- Code calls trigger_behavior_event("onGoalSet") legacy event
- But behaviors define on_goal_start() method, not onGoalSet()
- Event name mismatch means no behavior responds
- agent.goal never gets set
- mark_complete tools only added when agent.goal exists (tool_dispatch.py:326)
- Agent can't mark completion even when work is done

**Where it happened**:
1. base_agent.py:1030 - run_agent() for non-chat mode
2. agents/task_executor_agent.py:58 - __init__ when goal provided
3. agents/architect_agent.py:50 - __init__ when goal provided
4. agent_registry.py:228 - delegation system

**Fix applied**: ✅ Replace trigger_behavior_event("onGoalSet") with set_goal()
- Renamed _handle_goal_set() to set_goal() (public API)
- Updated all 4 call sites to use set_goal() directly
- Goal now properly set, mark_complete tool available
- Agent successfully completes tasks!

### Test 3.2: Calculator after fix
✅ PASS - Agent creates calculator, tests it, and marks complete in 6 rounds!

**Command**: `python agent.py --team solo "Create calculator with add and multiply functions"`
**Result**:
- calculator.py created with add() and multiply() functions
- test_calculator.py created with comprehensive tests
- Tests run and pass
- Agent calls mark_complete(summary=...)
- Task completes successfully

**This was a CRITICAL bug** - without it, agents could never mark completion!

---

## Phase 2: Solo Agent - Trivial Tasks

### Test 2.1: Create single file
❌ FAIL - Agent reads from root directory instead of workspace, creates nothing

**Bug**: workspace_manager never initialized, file tools use root directory
**Root cause**:
- main() creates isolated workspace at /workspace/.agent_workspaces/goal-slug/
- Agent is initialized with workspace=that path
- But agent.workspace_manager is never initialized
- File tools check `getattr(agent, 'workspace_manager', None)`
- When None, they use Path(".") which is /workspace/ (root directory)
- Agent spends all rounds reading root files (tests/, src/) instead of working in its workspace

**Fix applied**: ✅ Initialize workspace_manager in BaseAgent.__init__ (commit 74950f1)

### Test 2.2: Tool dispatch type mismatch with --once
❌ FAIL - Crash with AttributeError: 'str' object has no attribute 'get'

**Bug**: Tool behaviors return strings, but dispatcher expects dicts
**Root cause**:
- File tool behaviors (write_file, read_file, etc.) return strings
- ToolDispatcher._dispatch_to_behavior expects dict[str, Any] return
- execute_task() in base_agent.py tries to call result.get('success') on string
- Crashes with AttributeError

**Fix applied**: ✅ Normalize tool results in ToolDispatcher (handles str/dict/list)

### Test 2.3: Simple file creation with --once
✅ PASS - File created correctly in isolated workspace

**Command**: `python agent.py --team solo --once "Create hello.txt with Hello World"`
**Result**: hello.txt created with correct content in workspace
**All systems working**: workspace_manager, tool dispatch, --once mode

---

## Phase 1: Simple CLI Tests

### Test 1.1: --help flag
✅ PASS - Help text displays correctly

### Test 1.2: --list-teams flag
✅ PASS - All 5 teams listed correctly (chatbot, default, eval_with_inspection, meta, solo)

### Test 1.3: Chatbot team with --once flag
✅ PASS (after fixes)

**Bugs found and fixed:**
1. **Import error in agent_lifecycle.py**: `from llm_utils import` should be `from src.llm_utils import`
   - Fixed in src/agent_lifecycle.py:134
2. **ChatbotBehavior excluded in --once mode**: agent.py was excluding ChatbotBehavior for any initial_message
   - Fixed in agent.py:306 to not exclude when exit_after_initial=True
   - Now chatbot team works correctly with --once flag
