# Orchestrator-Architect-TaskExecutor Integration Test Plan

**Version**: 3.0
**Date**: 2025-10-31
**Status**: Ready for execution

## Overview

This test plan validates the complete orchestrator → architect → task executor workflow with the new composable enhancement architecture. The evaluation suite tests L5-L8 tasks to ensure the multi-agent system correctly handles task decomposition, delegation, and completion.

## Architecture Under Test

### Multi-Agent Workflow
```
User Goal
    ↓
Orchestrator (task decomposition & delegation)
    ↓
Architect (design & planning for L8 projects)
    ↓
TaskExecutor (code implementation)
    ↓
Result (files, tests, working code)
```

### Key Components Validated

1. **Orchestrator Agent**
   - Task decomposition into subtasks
   - Delegation to architect (L8 tasks) or task executor (L5-L7 tasks)
   - Task status tracking and completion
   - No max_rounds limit (wall-clock timeout only)

2. **Architect Agent** (L8 only)
   - Architecture design for complex systems
   - Component planning and specification
   - Design document generation

3. **TaskExecutor Agent**
   - Code implementation
   - File creation and modification
   - Test execution and validation
   - Linting and code quality

4. **Composable Enhancement System**
   - AppendUntilFullStrategy (base context strategy)
   - TaskManagementEnhancement (task context injection)
   - JetboxNotesEnhancement (persistent summaries)

## Test Levels

### L5: Simple Utilities (3 tasks)
**Complexity**: Single-file utilities with basic functionality
**Timeout**: 5 minutes
**Workflow**: Orchestrator → TaskExecutor (direct)

**Tasks**:
1. JSON/CSV Converter
2. Data Validator (email, phone, URL)
3. CLI Calculator with history

**Success Criteria**:
- All required functions/classes exist
- Tests pass (if included)
- Ruff checks pass
- Files created in workspace

**Expected Behavior**:
- Orchestrator creates 1-3 subtasks
- TaskExecutor implements directly
- Minimal delegation overhead

### L6: Multi-File Modules (3 tasks)
**Complexity**: Multiple files with dependencies
**Timeout**: 5 minutes
**Workflow**: Orchestrator → TaskExecutor (with decomposition)

**Tasks**:
1. REST API Client with auth and rate limiting
2. Data Processing Pipeline (multi-stage)
3. Config Manager with validation

**Success Criteria**:
- Multiple files created with correct structure
- Module imports work correctly
- Tests pass with good coverage
- Code quality maintained

**Expected Behavior**:
- Orchestrator decomposes into 3-6 subtasks
- TaskExecutor implements each component
- Task dependencies managed correctly

### L7: Complete Packages (3 tasks)
**Complexity**: Full Python packages with setup.py, docs, tests
**Timeout**: 5 minutes
**Workflow**: Orchestrator → TaskExecutor (structured project)

**Tasks**:
1. Python Package with setup.py and docs
2. Multi-Module Library (4+ modules)
3. CLI Tool with config and integration tests

**Success Criteria**:
- Package structure complete (setup.py, README, tests/)
- All modules and dependencies defined
- Tests pass with integration coverage
- Package installable with pip

**Expected Behavior**:
- Orchestrator creates 4-8 subtasks
- TaskExecutor builds package incrementally
- File structure follows conventions
- Jetbox notes capture implementation details

### L8: Full Systems (3 tasks × 2 modes = 6 tests)
**Complexity**: Multi-service systems with Docker, databases, APIs
**Timeout**: 10 minutes
**Workflow**: Orchestrator → **Architect** → TaskExecutor

**Tasks** (tested with AND without architect):
1. Microservices (user service + product service + Docker)
2. Web Application (React frontend + Flask backend + PostgreSQL)
3. Distributed System (API + worker + queue + storage)

**Success Criteria**:
- All components/services present
- Architecture matches requirements
- Docker/deployment config correct
- Integration tests pass (if included)
- Services can communicate

**Expected Behavior**:
- Orchestrator invokes architect for design
- Architect produces component specifications
- TaskExecutor implements each component
- Task management tracks dependencies
- Jetbox notes preserve design decisions

## Critical Test Scenarios

### 1. Timeout Handling
**Scenario**: Task exceeds wall-clock timeout
**Expected**:
- Process killed after timeout (5min for L5-L7, 10min for L8)
- Jetbox notes summary created (if enabled)
- Context dump saved to `.agent_context/timeout_dumps/`
- Graceful failure reported

**Validation**:
```bash
# Check timeout behavior
python run_project_evaluation.py --level L5
# Verify timeout at 300 seconds per task
```

### 2. Task Decomposition
**Scenario**: Orchestrator decomposes complex goal
**Expected**:
- Logical subtask breakdown
- Clear task dependencies
- Reasonable number of subtasks (2-8)
- Each subtask actionable by TaskExecutor

**Validation**:
- Review orchestrator logs for task creation
- Check task management enhancement context injection
- Verify subtask completion tracking

### 3. Architect Integration (L8)
**Scenario**: Complex system requires architecture design
**Expected**:
- Orchestrator delegates to architect first
- Architect produces design document
- Design passed to TaskExecutor
- TaskExecutor follows architecture

**Validation**:
```bash
# Run L8 with architect
python run_project_evaluation.py --level L8
# Check for architecture documents in workspace
# Verify component structure matches design
```

### 4. Context Strategy Composition
**Scenario**: Multiple enhancements active simultaneously
**Expected**:
- AppendUntilFullStrategy base working
- TaskManagement context injection present
- JetboxNotes context injection present
- No conflicts or duplication

**Validation**:
- Check LLM context for enhancement injections
- Verify each enhancement adds expected content
- Confirm no context overflow or truncation

### 5. Task Status Tracking
**Scenario**: Multi-task project with dependencies
**Expected**:
- Status display shows task hierarchy
- Progress bars update correctly
- Completed tasks marked
- Next pending task identified

**Validation**:
- Monitor status display during execution
- Check `.agent_context/state.json` for task status
- Verify performance stats tracking

## Execution Plan

### Phase 1: Validation (10 minutes)
```bash
# Verify test infrastructure
python test_eval_suite_quick.py

# Expected: All validation checks pass
```

### Phase 2: L5 Tests (15 minutes)
```bash
# Run simple utility tests
python run_project_evaluation.py --level L5

# Expected results:
# - 3/3 tasks complete
# - All within 5-minute timeout
# - Files created correctly
# - Tests pass
```

**Success Metrics**:
- Pass rate: ≥80% (2/3 tasks)
- Average duration: <180s per task
- All files created in workspace
- No context overflow errors

### Phase 3: L6 Tests (15 minutes)
```bash
# Run multi-file module tests
python run_project_evaluation.py --level L6

# Expected results:
# - 3/3 tasks complete
# - Correct file structure
# - Module imports work
# - Tests pass
```

**Success Metrics**:
- Pass rate: ≥66% (2/3 tasks)
- Average duration: <240s per task
- Multi-file structure correct
- Dependencies resolved

### Phase 4: L7 Tests (15 minutes)
```bash
# Run complete package tests
python run_project_evaluation.py --level L7

# Expected results:
# - 3/3 tasks complete
# - Package structure complete
# - Setup.py correct
# - Installable
```

**Success Metrics**:
- Pass rate: ≥66% (2/3 tasks)
- Average duration: <270s per task
- Package installable
- All components present

### Phase 5: L8 Tests (30 minutes)
```bash
# Run full system tests (with and without architect)
python run_project_evaluation.py --level L8

# Expected results:
# - 6/6 test cases complete (3 tasks × 2 modes)
# - Architect produces designs (architect mode)
# - All components created
# - Services can start
```

**Success Metrics**:
- Pass rate: ≥50% (3/6 test cases)
- Average duration: <480s per task
- Architecture documents present (architect mode)
- All services/components exist
- Docker configs correct

### Phase 6: Full Suite (60 minutes)
```bash
# Run complete evaluation suite
python run_project_evaluation.py

# Expected results:
# - 15 total test cases
# - Comprehensive report generated
# - Performance statistics collected
```

**Success Metrics**:
- Overall pass rate: ≥66% (10/15 tests)
- No crashes or hangs
- All timeouts respected
- Summary report generated

## Validation Criteria

### Per-Task Validation

**Automated Checks** (in `validate_task_result()`):
- Required classes/functions exist in code
- File structure matches specification
- Tests pass (if included)
- Ruff checks pass (Python)
- No syntax errors

**Manual Review**:
- Code quality and readability
- Architecture matches requirements
- Component interaction works
- Error handling appropriate

### Suite-Level Validation

**Quantitative**:
- Pass rate by level (L5, L6, L7, L8)
- Average duration by level
- Timeout rate
- Error categories

**Qualitative**:
- Task decomposition quality
- Delegation patterns
- Context strategy effectiveness
- Enhancement integration

## Known Issues and Considerations

### 1. Orchestrator Max Rounds Removed
**Issue**: Previous 10-round limit caused premature completion
**Fix**: Removed max_rounds, now uses wall-clock timeout exclusively
**Impact**: Orchestrator can fully complete complex tasks

### 2. Wall-Clock Timeout Priority
**Behavior**: Timeout kills entire subprocess (orchestrator + delegates)
**Consideration**: Long-running L8 tasks may hit 10-minute limit
**Mitigation**: Monitor timeout rate, adjust if needed

### 3. Context Size Management
**Config**: max_tokens = 8000 in agent_config.yaml
**Strategy**: AppendUntilFullStrategy with enhancements
**Risk**: Complex L8 tasks may approach context limit
**Mitigation**: Monitor context size in logs

### 4. Jetbox Notes Integration
**Feature**: Auto-summarization on task completion
**Benefit**: Context continuity for long tasks
**Validation**: Check `jetboxnotes.md` in workspace after runs

### 5. Task Management Enhancement
**Feature**: Context injection showing task status
**Benefit**: Agent sees task hierarchy and dependencies
**Validation**: Review LLM context for task info injection

## Output and Reporting

### Real-Time Monitoring
- Console output shows test progress
- Status display shows task hierarchy
- Performance stats update per round

### Results Files
- `evaluation_results/project_eval_results.jsonl` - Raw results (one JSON per line)
- `evaluation_results/PROJECT_EVAL_SUMMARY.md` - Generated markdown report
- `.agent_workspace/{task-id}/` - Individual workspace per task
- `.agent_context/state.json` - Final state per agent run

### Report Analysis
```bash
# View summary report
cat evaluation_results/PROJECT_EVAL_SUMMARY.md

# Analyze results programmatically
python -c "
import json
results = [json.loads(line) for line in open('evaluation_results/project_eval_results.jsonl')]
pass_rate = sum(1 for r in results if r['validation_result']['passed']) / len(results)
print(f'Pass rate: {pass_rate:.1%}')
"
```

## Success Definition

### Minimum Viable Success
- ≥50% pass rate across all levels (8/15 tests)
- All L5 tests pass (3/3)
- No crashes or infinite loops
- Timeouts respected

### Target Success
- ≥66% pass rate across all levels (10/15 tests)
- ≥80% pass rate on L5-L6 (5/6 tests)
- ≥50% pass rate on L7-L8 (5/9 tests)
- Architect improves L8 results vs no-architect mode

### Stretch Goals
- ≥80% pass rate across all levels (12/15 tests)
- All L5-L6 tests pass (6/6)
- ≥66% pass rate on L7-L8 (6/9 tests)
- Average task duration <60% of timeout

## Next Steps After Evaluation

### If Success Rate ≥66%
1. Document successful patterns
2. Create example workflows
3. Run extended evaluation (L9-L10 if available)
4. Performance optimization

### If Success Rate <66%
1. Analyze failure patterns by level
2. Review timeout adequacy (may need increase)
3. Check context strategy effectiveness
4. Improve task decomposition prompts
5. Enhance architect agent guidance

### Continuous Improvement
1. Add more test cases for weak areas
2. Refine validation criteria
3. Optimize context strategies
4. Improve agent prompts based on common failures
5. Add regression tests for fixed issues

## Running the Evaluation

### Quick Validation (2 minutes)
```bash
python test_eval_suite_quick.py
```

### Single Test (5-10 minutes)
```bash
pytest tests/test_project_evaluation.py::test_l5_task[L5_cli_calculator] -v
```

### Level-Specific (15-30 minutes)
```bash
python run_project_evaluation.py --level L5  # or L6, L7, L8
```

### Full Suite (60 minutes)
```bash
python run_project_evaluation.py
```

### Results Review
```bash
# View summary
cat evaluation_results/PROJECT_EVAL_SUMMARY.md

# Check specific workspace
ls .agent_workspace/L5_cli_calculator/

# Review jetbox notes
cat .agent_workspace/L5_cli_calculator/jetboxnotes.md
```

## Conclusion

This test plan provides comprehensive validation of the orchestrator-architect-executor integration with the new composable enhancement architecture. The evaluation suite tests realistic coding tasks across complexity levels, validating task decomposition, delegation, context management, and code generation capabilities.

**Ready to execute**: All infrastructure in place after max_rounds fix
**Expected duration**: 60 minutes for full suite
**Success threshold**: ≥66% pass rate (10/15 tests)

Execute Phase 1 (validation) first, then proceed through phases 2-6 based on results.
