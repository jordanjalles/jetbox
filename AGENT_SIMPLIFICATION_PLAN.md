# Agent Architecture Simplification Plan

## Current State Analysis

### File Inventory
- **agent.py** (2106 lines) - Monolithic standalone agent with full task execution
- **base_agent.py** (240 lines) - Abstract base class for multi-agent system
- **orchestrator_agent.py** (414 lines) - User-facing conversational agent
- **task_executor_agent.py** (247 lines) - Coding task executor (thin wrapper)
- **agent_registry.py** (238 lines) - Agent lifecycle and delegation manager
- **Total: 3,245 lines** across 5 files

### Key Redundancies

#### 1. **Dual Architecture Problem**
- `agent.py` = Standalone monolithic agent (can run tasks directly)
- `base_agent.py` + subclasses = Multi-agent system
- **These are COMPLETELY SEPARATE code paths for same functionality**

#### 2. **Tool Definitions**
- `agent.py` has all tools (write_file, read_file, list_dir, run_cmd, etc.) as standalone functions
- `task_executor_agent.py` references agent.py tools via dispatch
- **Tools are defined once but accessed through convoluted dispatch chain**

#### 3. **Context Building**
- `agent.py` has `build_context()` function (~100 lines)
- `task_executor_agent.py` has `build_context()` method (~60 lines)
- **Similar but different implementations of same logic**

#### 4. **Execution Loops**
- `agent.py` has massive `main()` function (~500 lines) with full execution loop
- `orchestrator_main.py` has separate loop for orchestrator
- **Two different ways to run agents**

#### 5. **Imports Circular Reference Risk**
- `base_agent.py` imports from `agent.py` (chat_with_inactivity_timeout, dispatch)
- Creates tight coupling between old and new architecture

## Problems This Causes

1. **Confusing for users**: Which file do I run? agent.py or orchestrator_main.py?
2. **Hard to maintain**: Changes need to happen in multiple places
3. **Code duplication**: Same logic implemented differently in different files
4. **Testing complexity**: Need to test two separate code paths
5. **Feature drift**: New features might only work in one path
6. **Workspace confusion**: agent.py uses WorkspaceManager, task_executor doesn't manage workspace itself

## Simplification Goals

1. **Single source of truth** for each component
2. **Clear separation** between agent behavior and tool implementations
3. **Unified execution model** (one way to run agents)
4. **Easy to extend** (add new agent types without duplicating code)
5. **Backward compatible** (existing workflows still work)

## Proposed Simplification Plan

### Phase 1: Extract Tools into Shared Module ✅ QUICK WIN

**Create `tools.py`** with all tool implementations:
- `write_file()`
- `read_file()`
- `list_dir()`
- `run_cmd()`
- `grep_file()`
- All helper functions

**Benefits:**
- Removes ~500 lines from agent.py
- Single definition of each tool
- Both agent.py and task_executor can import same tools
- Easy to test tools in isolation

**Files affected:** agent.py, task_executor_agent.py, base_agent.py

### Phase 2: Extract LLM Utilities into Shared Module ✅ QUICK WIN

**Create `llm_utils.py`** with:
- `chat_with_inactivity_timeout()`
- `estimate_tokens()` (if needed)
- Any other LLM wrapper functions

**Benefits:**
- Removes ~100 lines from agent.py
- Breaks circular dependency (base_agent.py importing from agent.py)
- Centralized LLM interaction logic

**Files affected:** agent.py, base_agent.py

### Phase 3: Consolidate Context Building 🔧 MODERATE

**Create `context_strategies.py`** with:
- `build_hierarchical_context()` - for TaskExecutor
- `build_append_context()` - for Orchestrator
- Common context manipulation utilities

**Benefits:**
- Single implementation of each strategy
- agent.py can use build_hierarchical_context()
- task_executor_agent.py uses same function
- Easier to add new strategies

**Files affected:** agent.py, task_executor_agent.py, orchestrator_agent.py

### Phase 4: Migrate agent.py to Use BaseAgent 🔧 MODERATE

**Refactor agent.py into `standalone_agent.py`:**
```python
from base_agent import BaseAgent
from tools import write_file, read_file, run_cmd, list_dir
from context_strategies import build_hierarchical_context

class StandaloneAgent(BaseAgent):
    """Standalone agent for direct task execution (backward compatibility)."""

    def get_tools(self):
        return [write_file, read_file, run_cmd, list_dir, ...]

    def build_context(self):
        return build_hierarchical_context(self.context_manager, self.state.messages)

    # ... minimal implementation
```

**Keep `agent.py` as compatibility wrapper:**
```python
# agent.py - backward compatibility entry point
from standalone_agent import StandaloneAgent

def main():
    agent = StandaloneAgent(...)
    agent.run()

if __name__ == "__main__":
    main()
```

**Benefits:**
- Reduces agent.py from 2106 lines to ~50 lines
- Unifies execution model
- StandaloneAgent is just another agent type
- Can still run `python agent.py` for backward compatibility

**Files affected:** agent.py → standalone_agent.py + agent.py wrapper

### Phase 5: Unified Execution Entry Point ⚡ OPTIONAL

**Create `run_agent.py`:**
```bash
# Run standalone agent (old behavior)
python run_agent.py --agent standalone "Create calculator"

# Run orchestrator (multi-agent mode)
python run_agent.py --agent orchestrator

# Run task executor directly (for testing)
python run_agent.py --agent task_executor "Write tests"
```

**Benefits:**
- Single entry point for all agent types
- Consistent CLI interface
- Easy to switch between modes

**Files affected:** New file run_agent.py, orchestrator_main.py becomes optional wrapper

### Phase 6: Consolidate Execution Logic 🔥 COMPLEX

**Extract common execution loop into `agent_executor.py`:**
- Round management
- Status display updates
- Tool call dispatch
- Escalation logic
- Completion detection

**Each agent just defines:**
- Tools
- System prompt
- Context strategy
- Custom behavior hooks (optional)

**Benefits:**
- Massive reduction in code duplication
- Execution improvements benefit all agents
- Testing is much simpler
- New agent types are trivial to add

**Files affected:** agent.py, orchestrator_main.py, task_executor_agent.py

## Recommended Implementation Order

### 🎯 Immediate (Phase 1-2): Extract Shared Utilities
**Effort:** 2-3 hours
**Risk:** Low
**Impact:** High - breaks circular dependencies, reduces duplication

1. Create `tools.py` with all tool implementations
2. Create `llm_utils.py` with LLM wrapper functions
3. Update imports in all files
4. Run tests to verify no breakage

### 🎯 Short-term (Phase 3-4): Unify Agent Model
**Effort:** 4-6 hours
**Risk:** Medium
**Impact:** Very High - single architecture for all agents

1. Create `context_strategies.py`
2. Refactor agent.py into StandaloneAgent + wrapper
3. Update task_executor to use shared context strategies
4. Extensive testing of both execution modes

### 🎯 Medium-term (Phase 5-6): Full Consolidation
**Effort:** 8-12 hours
**Risk:** Medium-High
**Impact:** Extreme - maintainable, extensible architecture

1. Extract common execution loop
2. Create unified entry point
3. Comprehensive testing and documentation
4. Migrate existing workflows

## File Size Projections (After Full Simplification)

### New Structure:
```
tools.py                    ~500 lines (extracted from agent.py)
llm_utils.py               ~150 lines (extracted from agent.py)
context_strategies.py      ~200 lines (consolidated from 3 files)
agent_executor.py          ~400 lines (extracted common execution)
base_agent.py              ~150 lines (simplified, no circular deps)
standalone_agent.py        ~100 lines (thin wrapper, uses shared code)
task_executor_agent.py     ~100 lines (thin wrapper, uses shared code)
orchestrator_agent.py      ~200 lines (thin wrapper, uses shared code)
agent_registry.py          ~200 lines (mostly unchanged)
agent.py                   ~50 lines (compatibility wrapper)
run_agent.py               ~100 lines (unified entry point)
-----------------------------------------------------------
TOTAL:                     ~2,150 lines (vs 3,245 current)
```

**Reduction:** ~1,100 lines (34% reduction)
**More importantly:** Clear separation of concerns, single source of truth for each component

## Migration Path for Users

### Old way (still works):
```bash
python agent.py "Create a calculator package"
```

### New way:
```bash
python run_agent.py --agent standalone "Create a calculator package"
# Or just:
python agent.py "Create a calculator package"  # wrapper calls StandaloneAgent
```

### Multi-agent mode:
```bash
python run_agent.py --agent orchestrator
# Or:
python orchestrator_main.py  # wrapper calls run_agent
```

## Testing Strategy

For each phase:
1. **Unit tests** for extracted modules
2. **Integration tests** for agent execution
3. **Regression tests** for existing workflows
4. **Smoke tests** for all entry points

Key test scenarios:
- Run agent.py standalone → should work as before
- Run orchestrator → should delegate to task executor
- Task executor can use all tools
- Context building works for both strategies
- Workspace isolation works correctly

## Success Criteria

✅ All existing tests pass
✅ Both agent.py and orchestrator_main.py work
✅ No circular dependencies
✅ Each tool/utility defined in one place
✅ New agent types can be added in <100 lines
✅ Code duplication reduced by >30%
✅ Documentation updated to reflect new structure

## Risks and Mitigations

**Risk:** Breaking existing workflows
**Mitigation:** Keep agent.py as compatibility wrapper, extensive testing

**Risk:** Complex refactoring introduces bugs
**Mitigation:** Incremental approach (phases 1-2 first), test after each step

**Risk:** Execution loop changes break assumptions
**Mitigation:** Extract in phase 6 (last), comprehensive testing beforehand

**Risk:** Import cycles in new structure
**Mitigation:** Clear dependency hierarchy: tools → llm_utils → context_strategies → agents

## Next Steps

**Recommended approach:** Start with Phase 1-2 (extract tools and LLM utils)

This gives immediate benefits:
- Breaks circular dependency
- Reduces duplication
- Low risk
- Sets foundation for larger refactoring

Would you like me to:
1. **Start with Phase 1** (extract tools.py)?
2. **Create detailed implementation plan** for a specific phase?
3. **Prototype the new structure** to validate the approach?
