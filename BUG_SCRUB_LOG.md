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
