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
