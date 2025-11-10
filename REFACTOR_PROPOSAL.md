# Code Maintainability Refactor Proposal

## Executive Summary

The Jetbox codebase has grown organically and now has several maintainability issues:
- **base_agent.py**: 2,745 lines, 71 methods (should be <500 lines)
- **behaviors/**: 29 files in flat structure (needs categorization)
- **tests/**: 163 files in flat directory (needs organization)
- **archive/**: 949MB (includes 946MB of unrelated checkpoint data)
- **Root directory**: 13 Python files (should be <5 core files)

This proposal restructures the codebase for better maintainability, discoverability, and scalability.

---

## Problem Analysis

### 1. base_agent.py is Too Large (2,745 lines)

**Current structure:**
```
base_agent.py (2,745 lines, 71 methods)
├── Abstract methods (line 208)
├── Shared functionality (line 433)
├── Phase 1: Subsystem helpers (line 829)
├── Phase 4: Behavior system (line 877)
├── CLI entry points (line 1802)
└── Generic run() method (line 2264)
```

**Issues:**
- Single file with multiple responsibilities
- Hard to navigate and understand
- Merge conflicts likely in collaborative work
- Testing becomes complex

**Evidence:**
- Industry standard: Classes should be <500 lines
- Current: 2,745 lines (5.5x too large)
- 71 methods (should be <20 per class)

### 2. Flat behaviors/ Directory (29 files)

**Current structure:**
```
behaviors/
├── CalculatorBehavior.py
├── DockerBehavior.py
├── architect_tools.py
├── chatbot.py
├── command_tools.py
├── compact_when_near_full.py
├── create_agent.py
├── delegation.py
├── loop_detection.py
├── ... (20 more files)
```

**Issues:**
- No categorization (hard to find related behaviors)
- Experimental behaviors mixed with production
- Tool behaviors mixed with utility behaviors
- Difficult to onboard new contributors

**Clear categories exist:**
- **Tools**: write_file, read_file, directory, command, server, architect (6 files)
- **Context**: compact_when_near_full, context_inspector (2 files)
- **Management**: workspace, server, task, delegation (4 files)
- **Meta**: create_agent, create_behavior (2 files)
- **Validation**: sandbox_test, validation, test_cli_injector (3 files)
- **Utils**: chatbot, status_display, loop_detection, workspace_task_notes (4 files)
- **Experimental**: Calculator, Docker, Environment, HttpRequest, JsonTools, SearchTools (6 files)

### 3. Flat tests/ Directory (163 files)

**Current structure:**
```
tests/
├── test_CalculatorBehavior.py
├── test_agent_behavior.py
├── test_behavior_composability.py
├── ... (130 more test files)
├── evaluation_suite.py
├── comprehensive_eval.py
├── ... (20+ evaluation scripts)
├── fixtures/ (subdirectory)
├── evaluation_scripts/ (subdirectory - only 8 files)
└── ... (logs, json files, etc.)
```

**Issues:**
- 163 files in flat structure
- Unit tests mixed with integration tests
- Evaluation scripts mixed with tests
- Non-test files (logs, json) cluttering directory
- evaluation_scripts/ subdirectory exists but underutilized

**Categories needed:**
- Unit tests (test individual components)
- Integration tests (test component interactions)
- Evaluation/benchmark scripts
- Fixtures (already exists)

### 4. Root Directory Clutter (13 Python files)

**Current root files:**
```
├── agent.py ✓ (CLI entry - keep)
├── base_agent.py ✓ (Core - keep but refactor)
├── agent_config.py ✓ (Config - keep)
├── llm_utils.py ✓ (LLM - keep)
├── workspace_manager.py ✓ (Workspace - keep)
├── agent_registry.py ⚠️ (Move to src/)
├── server_manager.py ⚠️ (Move to src/)
├── ollama_manager.py ⚠️ (Move to src/)
├── completion_detector.py ⚠️ (Move to src/)
├── chat_with_simple_chatbot.py ❌ (Demo - move to examples/)
├── architect_agent.py ❌ (Delete - use YAML config)
├── orchestrator_agent.py ❌ (Delete - use YAML config)
└── task_executor_agent.py ❌ (Delete - use YAML config)
```

**Issues:**
- Too many files in root (13 vs ideal 5-7)
- Specific agent files violate "config-driven" principle
- Utilities mixed with core files
- Demo scripts in root

### 5. archive/ is Bloated (949MB)

**Current size:**
```
949MB   archive/
├── 946MB   misc_dirs/hrm-jepa/checkpoints
├── 1.6MB   old_evals/
├── 696KB   old_evaluations/
└── 472KB   old_reports/
```

**Issues:**
- **946MB of unrelated checkpoint data** (hrm-jepa - not related to Jetbox!)
- Should not be in repository
- Bloats clone size
- Makes git operations slow

### 6. evaluation_results/ Needs Organization (20MB)

**Current structure:**
- 4.5MB of markdown files in flat root
- Many context_analysis_*/ subdirectories
- No chronological organization
- Hard to find specific evaluation runs

---

## Proposed Structure

### Phase 1: Split base_agent.py

**New structure:**
```
src/
├── base_agent.py                 # Core orchestration (~500 lines)
│   ├── __init__()
│   ├── set_goal()
│   ├── get_tools()
│   ├── build_context()
│   └── call_llm()
│
├── agent_lifecycle.py            # Run loop and setup (~400 lines)
│   ├── run()
│   ├── _setup_run()
│   ├── _main_loop()
│   └── _teardown()
│
├── behavior_loader.py            # Behavior loading system (~600 lines)
│   ├── load_behaviors_from_config()
│   ├── _import_behavior_class()
│   ├── _validate_system_prompt()
│   └── _auto_add_delegation_behavior()
│
├── tool_dispatch.py              # Tool registry and dispatch (~300 lines)
│   ├── dispatch_tool()
│   ├── dispatch_tool_to_behavior()
│   ├── _validate_tool_parameters()
│   └── generate_tool_documentation()
│
├── agent_state.py                # State management (~200 lines)
│   ├── AgentState (dataclass)
│   ├── persist_state()
│   ├── load_state()
│   └── state serialization helpers
│
└── agent_events.py               # Event system (~200 lines)
    ├── trigger_behavior_event()
    ├── _trigger_on_goal_start()
    ├── _trigger_initial_context_setup()
    └── _trigger_on_round_start()
```

**Migration strategy:**
1. Create src/ directory
2. Extract each module with tests
3. Update imports in base_agent.py
4. Gradually move functionality
5. Keep backward compatibility during transition

**Benefits:**
- Each file has single responsibility
- Easier to test individual components
- Better code navigation
- Reduced merge conflicts
- Easier to onboard new contributors

### Phase 2: Organize behaviors/

**New structure:**
```
behaviors/
├── __init__.py
├── base.py                       # Base behavior class
│
├── tools/                        # Tool-providing behaviors
│   ├── __init__.py
│   ├── write_file_tools.py
│   ├── read_file_tools.py
│   ├── directory_tools.py
│   ├── command_tools.py
│   ├── server_tools.py
│   └── architect_tools.py
│
├── context/                      # Context management
│   ├── __init__.py
│   ├── compact_when_near_full.py
│   └── context_inspector.py
│
├── management/                   # System management
│   ├── __init__.py
│   ├── workspace_management.py
│   ├── server_management.py
│   ├── task_management.py
│   └── delegation.py
│
├── meta/                         # Meta-programming
│   ├── __init__.py
│   ├── create_agent.py
│   └── create_behavior.py
│
├── validation/                   # Testing/validation
│   ├── __init__.py
│   ├── sandbox_test.py
│   ├── validation.py
│   └── test_cli_injector.py
│
├── utils/                        # Utility behaviors
│   ├── __init__.py
│   ├── chatbot.py
│   ├── status_display.py
│   ├── loop_detection.py
│   └── workspace_task_notes.py
│
└── experimental/                 # Experimental/example behaviors
    ├── __init__.py
    ├── CalculatorBehavior.py
    ├── DockerBehavior.py
    ├── EnvironmentBehavior.py
    ├── HttpRequestBehavior.py
    ├── JsonToolsBehavior.py
    └── SearchToolsBehavior.py
```

**Migration strategy:**
1. Create subdirectories
2. Move files with git mv (preserves history)
3. Update imports in behaviors/__init__.py
4. Update import paths in tests
5. Verify all tests pass

**Benefits:**
- Clear categorization
- Easy to find related behaviors
- Experimental code clearly separated
- Better for new contributors

### Phase 3: Organize tests/

**New structure:**
```
tests/
├── __init__.py
├── conftest.py                   # Pytest configuration
├── pytest.ini                    # Pytest settings
│
├── unit/                         # Unit tests (one component)
│   ├── __init__.py
│   ├── test_base_agent.py
│   ├── test_agent_config.py
│   ├── test_llm_utils.py
│   ├── behaviors/
│   │   ├── test_write_file_tools.py
│   │   ├── test_command_tools.py
│   │   └── ...
│   └── ...
│
├── integration/                  # Integration tests (multiple components)
│   ├── __init__.py
│   ├── test_agent_behavior_integration.py
│   ├── test_behavior_composability.py
│   ├── test_workspace_isolation.py
│   └── ...
│
├── evaluation/                   # Evaluation/benchmark scripts
│   ├── __init__.py
│   ├── evaluation_suite.py
│   ├── evaluation_suite_extended.py
│   ├── flexible_validation.py
│   ├── scripts/
│   │   ├── run_l5_l7_x5_eval.py
│   │   ├── rerun_l5_l7_eval.py
│   │   ├── comprehensive_eval.py
│   │   └── ...
│   └── validators/
│       └── ... (validation helpers)
│
└── fixtures/                     # Test fixtures
    ├── __init__.py
    └── ... (existing fixtures)
```

**Migration strategy:**
1. Create subdirectories
2. Categorize existing tests:
   - test_*.py → unit/ or integration/
   - *eval*.py, *comprehensive*.py → evaluation/
3. Move files with git mv
4. Update pytest.ini test discovery paths
5. Clean up non-test files (logs, json)

**Benefits:**
- Clear test organization
- Faster test discovery
- Run specific test categories
- Easier to maintain

### Phase 4: Reorganize Root Directory

**New structure:**
```
jetbox/
├── agent.py                      # CLI entry point ✓
├── pyproject.toml                # Project config ✓
├── README.md                     # Documentation ✓
├── CLAUDE.md                     # AI assistant guide ✓
├── JetboxArchitecture.md         # Architecture docs ✓
├── jetbox_commands_whitelist     # Command whitelist ✓
│
├── src/                          # Core library (NEW)
│   ├── __init__.py
│   ├── base_agent.py            # Refactored core
│   ├── agent_lifecycle.py
│   ├── behavior_loader.py
│   ├── tool_dispatch.py
│   ├── agent_state.py
│   ├── agent_events.py
│   ├── agent_config.py          # Moved from root
│   ├── llm_utils.py             # Moved from root
│   ├── workspace_manager.py     # Moved from root
│   ├── agent_registry.py        # Moved from root
│   ├── server_manager.py        # Moved from root
│   ├── ollama_manager.py        # Moved from root
│   └── completion_detector.py   # Moved from root
│
├── behaviors/                    # Organized behaviors ✓ (see Phase 2)
├── config/                       # Configuration files ✓
├── tests/                        # Organized tests ✓ (see Phase 3)
│
├── examples/                     # Example scripts (NEW)
│   ├── chat_with_simple_chatbot.py
│   └── ... (other demos)
│
├── evaluation_results/           # Eval results (reorganize)
│   ├── 2024-11/
│   ├── 2024-10/
│   └── latest/
│
├── debug_scripts/                # Debug utilities ✓
├── tools/                        # Dev tools ✓
├── docs/                         # Documentation ✓
└── archive/                      # Cleaned archive
```

**Files to delete:**
- architect_agent.py ❌
- orchestrator_agent.py ❌
- task_executor_agent.py ❌

These violate the "config-driven" principle and duplicate YAML configs.

**Migration strategy:**
1. Create src/ directory
2. Move core files to src/ with git mv
3. Create examples/ directory
4. Move demo scripts to examples/
5. Delete specific agent files (ensure YAML configs work)
6. Update all import paths
7. Update documentation

**Benefits:**
- Clear separation of library vs entry point
- Utilities properly organized
- Examples separated from core code
- Follows Python best practices

### Phase 5: Clean archive/ (949MB → <10MB)

**Actions:**
1. **Delete archive/misc_dirs/hrm-jepa/** (946MB)
   - This is unrelated checkpoint data
   - Not part of Jetbox
   - Bloats repository

2. **Compress old_evals/** (1.6MB)
   - Create archive/old_evals.tar.gz
   - Delete uncompressed files

3. **Review old_evaluations/** and old_reports/
   - Keep significant findings
   - Move to evaluation_results/archive/
   - Delete redundant files

**Expected result:**
- 949MB → <10MB
- Faster git operations
- Smaller clone size

### Phase 6: Organize evaluation_results/

**Current:** 20MB, flat structure with many .md files

**New structure:**
```
evaluation_results/
├── 2024-11/
│   ├── POST_FIX_SUMMARY.md
│   ├── l4_l7_eval_TRUE_POST_FIX.log
│   └── context_analysis_20241108_*/
│
├── 2024-10/
│   └── ... (older results)
│
├── latest/                       # Symlink to most recent
│
└── archive/
    └── ... (very old results)
```

**Benefits:**
- Chronological organization
- Easy to find recent results
- Clear separation of old data

---

## Migration Plan

### Timeline

**Week 1: Planning & Preparation**
- Review proposal with team
- Identify breaking changes
- Plan rollout strategy
- Create migration branch

**Week 2-3: Phase 1 (Split base_agent.py)**
- Create src/ directory structure
- Extract agent_state.py (with tests)
- Extract tool_dispatch.py (with tests)
- Extract agent_events.py (with tests)
- Extract behavior_loader.py (with tests)
- Extract agent_lifecycle.py (with tests)
- Update base_agent.py to use new modules
- Verify all tests pass

**Week 4: Phase 2 (Organize behaviors/)**
- Create behavior subdirectories
- Move files with git mv
- Update imports
- Verify all tests pass

**Week 5: Phase 3 (Organize tests/)**
- Create test subdirectories
- Categorize and move tests
- Update pytest configuration
- Verify all tests pass

**Week 6: Phase 4 (Root directory)**
- Move core files to src/
- Create examples/ directory
- Delete deprecated agent files
- Update imports across codebase
- Update documentation

**Week 7: Phase 5 & 6 (Cleanup)**
- Clean archive/ directory
- Organize evaluation_results/
- Final testing
- Documentation updates

### Rollback Strategy

Each phase is independent and can be rolled back:
- Work on feature branch
- Merge after each phase passes tests
- Tag each phase for easy rollback
- Maintain backward compatibility during transition

---

## Impact Analysis

### Breaking Changes

**Phase 1 (Split base_agent.py):**
- None (internal refactoring, API unchanged)

**Phase 2 (Organize behaviors/):**
- Import paths change: `from behaviors.write_file_tools` → `from behaviors.tools.write_file_tools`
- Mitigated by: Update behaviors/__init__.py to re-export for backward compatibility

**Phase 3 (Organize tests/):**
- None (tests are internal)

**Phase 4 (Root directory):**
- Import paths change: `from agent_config` → `from src.agent_config`
- Mitigated by: Keep compatibility imports in root or update all imports

**Phase 5 & 6 (Cleanup):**
- None (removes unused files)

### Benefits Summary

**Maintainability:**
- Single-responsibility modules (easier to understand)
- Clear separation of concerns
- Reduced file size (easier to navigate)

**Discoverability:**
- Logical directory structure
- Related files grouped together
- Clear naming conventions

**Scalability:**
- Easy to add new behaviors (clear categories)
- Easy to add new tests (organized structure)
- Room for growth without clutter

**Collaboration:**
- Reduced merge conflicts (smaller files)
- Clear contribution guidelines (organized structure)
- Easier onboarding (intuitive organization)

**Performance:**
- Smaller archive (faster clone/pull)
- Organized tests (faster test discovery)
- Cleaner git history (organized commits)

---

## Alternatives Considered

### Alternative 1: Keep Current Structure
**Pros:** No migration effort
**Cons:** Maintainability continues to degrade, harder to onboard contributors

### Alternative 2: Gradual Refactoring
**Pros:** Less disruptive
**Cons:** Takes longer, inconsistent structure during transition

### Alternative 3: Complete Rewrite
**Pros:** Clean slate
**Cons:** Too risky, loses git history, huge effort

**Chosen approach:** Structured migration (this proposal)
- Balances disruption vs benefit
- Preserves git history
- Can be done incrementally
- Each phase adds value independently

---

## Success Criteria

1. **File size:** base_agent.py < 600 lines
2. **Directory depth:** behaviors/ has 2-3 level hierarchy
3. **Test organization:** tests/ has clear unit/integration/evaluation split
4. **Repository size:** archive/ < 10MB
5. **All tests pass:** No broken functionality
6. **Documentation:** Updated architecture docs
7. **No regressions:** Evaluation benchmarks maintain performance

---

## Recommendation

**Proceed with phased migration:**
1. Start with Phase 1 (base_agent.py split) - Highest impact
2. Follow with Phase 2 (behaviors organization) - High value, low risk
3. Complete remaining phases based on priority

**Key success factors:**
- Maintain backward compatibility during transition
- Comprehensive testing after each phase
- Clear communication with team
- Preserve git history with git mv
- Update documentation continuously

---

## Appendix: File Size Statistics

```
Top 10 Longest Files:
2,745 lines  base_agent.py ❌
1,294 lines  tools/analyze_context.py ⚠️
1,073 lines  behaviors/delegation.py ⚠️
  820 lines  behaviors/workspace_task_notes.py ⚠️
  795 lines  behaviors/create_behavior.py ⚠️
  754 lines  tests/test_agent_validator.py ⚠️
  728 lines  tests/test_project_evaluation.py ✓
  667 lines  tests/test_validation_behavior.py ✓
  656 lines  tests/test_create_behavior.py ✓
  629 lines  tests/flexible_validation.py ✓

Directory Sizes:
  949MB  archive/ ❌
   41MB  docs/ ⚠️
   20MB  evaluation_results/ ⚠️
  2.8MB  tests/ ⚠️
  904KB  behaviors/ ✓
   92KB  debug_scripts/ ✓
   76KB  config/ ✓
```

Legend:
- ❌ Urgent attention needed
- ⚠️ Should be improved
- ✓ Acceptable

---

**End of Proposal**
