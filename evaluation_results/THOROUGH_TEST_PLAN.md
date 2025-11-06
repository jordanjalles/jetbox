# Thorough Test Plan - Lifecycle API Migration Verification

**Purpose**: Comprehensive validation of lifecycle API migration to ensure no regressions and verify all behaviors work correctly with the new event-driven architecture.

**Status**: READY TO EXECUTE (awaiting approval)

---

## Test Categories

### 1. L1-L6 Baseline Tests (FIXED)
**Status**: Test harness updated with PYTHONPATH fix
**Command**: `python test_lifecycle_api_l1_l6.py`
**Expected Results**:
- L1 (Simple file): PASS
- L2 (Calculator + tests): PASS
- L3 (mathx package): PASS (now fixed)
- L4 (Calculator package + linting): PASS (now fixed)
- L5 (Validator package): PASS (now fixed)
- L6 (Event bus): PASS

**What This Tests**:
- Basic file operations (WriteFileToolsBehavior)
- Directory operations (DirectoryToolsBehavior)
- Command execution (CommandToolsBehavior)
- Package creation in isolated workspaces
- Test execution (pytest integration)
- Linting integration (ruff)
- Context compaction (CompactWhenNearFullBehavior)
- Loop detection (LoopDetectionBehavior)
- Workspace task notes (WorkspaceTaskNotesBehavior)

**Pass Criteria**: 6/6 tests pass (100%)

---

### 2. Lifecycle Event Verification Tests
**Status**: NOT IMPLEMENTED YET
**Location**: Create `tests/test_lifecycle_events.py`

**Test Cases**:

#### 2.1 Event Sequence Verification
- Verify `on_goal_start()` called exactly once at goal start
- Verify `on_initial_context()` called exactly once before first round
- Verify `on_round_start()` called every round
- Verify `on_llm_response()` called after LLM responds
- Verify `on_tool_call()` called for each tool execution
- Verify `on_round_end()` called after all tools in round
- Verify `on_goal_complete()` or `on_timeout()` called at end

**Implementation Strategy**:
```python
class EventTrackingBehavior(AgentBehavior):
    """Behavior that logs all lifecycle events for verification."""
    def __init__(self):
        self.events = []

    def on_goal_start(self, agent, goal):
        self.events.append(('on_goal_start', {'goal': goal}))

    # ... track all events
```

**Pass Criteria**: All events fire in correct order

---

#### 2.2 Agent State Access Verification
**Test**: Verify behaviors can access agent state via first parameter

**Test Cases**:
- `agent.workspace` accessible in all lifecycle methods
- `agent.state` accessible (goal, messages, etc.)
- `agent.context_manager` accessible
- `agent.workspace_manager` accessible

**Pass Criteria**: All agent properties accessible

---

#### 2.3 Context Injection Verification
**Test**: Verify `on_initial_context()` vs `on_round_start()` timing

**Test Cases**:
- Static content injected by `on_initial_context()` appears once
- Dynamic content injected by `on_round_start()` appears every round
- Verify context efficiency (no redundant injections)

**Pass Criteria**: Static content injected once, dynamic content every round

---

### 3. Behavior-Specific Tests

#### 3.1 CompactWhenNearFullBehavior
**Status**: Unit test exists at `tests/test_compact_when_near_full_behavior.py`
**Command**: `pytest tests/test_compact_when_near_full_behavior.py -v`

**What This Tests**:
- Token estimation
- Compaction trigger at 75% threshold
- Context summarization
- Compaction state tracking

**Pass Criteria**: All unit tests pass

---

#### 3.2 ChatbotBehavior
**Status**: Simplified, needs new tests
**Location**: Create `tests/test_chatbot_behavior.py`

**Test Cases**:
- Chat mode activates when no goal provided
- `set_goal` tool available in chat mode
- `clarify_with_user` tool available in chat mode
- Chat instructions inject ONCE (not every round)
- Chat instructions append to messages (not inject after system)
- Transition to execution mode when `set_goal` called
- No chat loops (agent handles I/O)

**Pass Criteria**: All test cases pass

---

#### 3.3 LoopDetectionBehavior
**Status**: Needs integration test
**Location**: Create `tests/test_loop_detection_behavior.py`

**Test Cases**:
- Detect repeated identical tool calls
- Inject warning after threshold exceeded
- Reset counter on different tool call
- Handle max_repeats configuration

**Pass Criteria**: Loop detection works correctly

---

#### 3.4 WorkspaceTaskNotesBehavior
**Status**: Needs integration test
**Location**: Create `tests/test_workspace_task_notes_behavior.py`

**Test Cases**:
- Load existing notes on agent startup
- Auto-summarize completed tasks
- Auto-summarize successful goals
- Persist notes to workspace
- Handle missing notes file gracefully

**Pass Criteria**: Notes load, summarize, and persist correctly

---

### 4. Integration Tests

#### 4.1 Multi-Behavior Composition
**Status**: NOT IMPLEMENTED
**Location**: Create `tests/test_behavior_composition.py`

**Test Cases**:
- All behaviors work together without conflicts
- No behavior dependencies on other behaviors
- Event handlers don't interfere with each other
- Context modifications compose correctly

**Pass Criteria**: Behaviors compose cleanly

---

#### 4.2 Config-Driven Behavior Loading
**Status**: Implicitly tested by L1-L6
**Location**: Create `tests/test_config_behavior_loading.py`

**Test Cases**:
- Load behaviors from YAML config
- Apply global defaults
- Apply per-agent overrides
- Exclude behaviors via config
- Handle missing config gracefully

**Pass Criteria**: Config loading works correctly

---

### 5. Regression Tests

#### 5.1 Workspace Isolation
**Status**: Needs verification
**Command**: Run multiple agents concurrently

**Test Cases**:
- Each agent gets isolated workspace
- No workspace conflicts
- File operations stay within workspace
- Workspace cleanup doesn't affect other workspaces

**Pass Criteria**: Full workspace isolation

---

#### 5.2 Performance Comparison
**Status**: Baseline exists from previous runs
**Command**: Compare old vs new lifecycle API timings

**Metrics**:
- Average time per task (should be similar)
- Token usage (should be lower with on_initial_context)
- Success rate (should be same or better)

**Pass Criteria**: No performance regression

---

## Execution Order

**Phase 1: Quick Validation** (5 minutes)
1. Run L1-L6 baseline tests
2. Verify 6/6 pass

**Phase 2: Unit Tests** (10 minutes)
1. Run CompactWhenNearFullBehavior unit tests
2. Create and run ChatbotBehavior tests
3. Create and run LoopDetectionBehavior tests
4. Create and run WorkspaceTaskNotesBehavior tests

**Phase 3: Integration Tests** (15 minutes)
1. Create lifecycle event tracking test
2. Create behavior composition test
3. Create config loading test

**Phase 4: Regression Tests** (10 minutes)
1. Run workspace isolation test
2. Compare performance metrics

**Total Estimated Time**: ~40 minutes

---

## Success Criteria

### Must Pass (Critical)
- ✅ L1-L6 baseline tests: 6/6 pass
- ✅ CompactWhenNearFullBehavior unit tests: all pass
- ✅ Lifecycle event sequence: correct order
- ✅ Agent state access: all properties accessible
- ✅ No regressions in workspace isolation

### Should Pass (Important)
- ChatbotBehavior tests: all pass
- LoopDetectionBehavior tests: all pass
- WorkspaceTaskNotesBehavior tests: all pass
- Behavior composition: no conflicts

### Nice to Have (Optional)
- Performance comparison: similar or better
- Config loading: handles edge cases

---

## Current Status

**Completed**:
- ✅ Lifecycle API migration
- ✅ Removed deprecated code
- ✅ Simplified ChatbotBehavior
- ✅ Fixed test harness PYTHONPATH issue
- ✅ Test plan prepared

**Ready to Execute**:
- Test harness updated and ready: `test_lifecycle_api_l1_l6.py`
- PYTHONPATH fix verified: tests pass on L3 workspace

**Awaiting Approval**:
- User instruction: "get ready to do a thorough test but do not start"
- Awaiting go-ahead to execute test plan

---

## Commands to Execute (When Approved)

### Automated Test Runner
```bash
# Run all phases
python run_thorough_tests.py

# Run specific phase only
python run_thorough_tests.py --phase 1  # L1-L6 baseline
python run_thorough_tests.py --phase 2  # Unit tests
python run_thorough_tests.py --phase 3  # Integration tests
python run_thorough_tests.py --phase 4  # Regression tests
```

### Individual Test Commands
```bash
# Phase 1: Quick Validation
python test_lifecycle_api_l1_l6.py

# Phase 2: Unit Tests (after creating tests)
pytest tests/test_compact_when_near_full_behavior.py -v
pytest tests/test_chatbot_behavior.py -v
pytest tests/test_loop_detection_behavior.py -v
pytest tests/test_workspace_task_notes_behavior.py -v

# Phase 3: Integration Tests (after creating tests)
pytest tests/test_lifecycle_events.py -v
pytest tests/test_behavior_composition.py -v
pytest tests/test_config_behavior_loading.py -v

# Phase 4: Regression Tests
# (Manual verification of workspace isolation)
# (Compare metrics with baseline)
```

---

## Expected Results

After full test execution:
- All L1-L6 tests pass (6/6)
- All unit tests pass
- All integration tests pass
- No regressions detected
- Lifecycle API validated as working correctly

**Confidence Level**: HIGH - Based on:
- Previous L1-L6 results: 3/6 pass (before PYTHONPATH fix)
- PYTHONPATH fix verified: L3 workspace tests now pass
- Lifecycle API migration completed cleanly
- No deprecated code remaining
