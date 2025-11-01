# Jetbox Agent Behaviors Refactoring Plan

**Status**: Phase 0 - Planning Complete
**Last Updated**: 2025-01-01
**Target Completion**: 6 weeks (Phases 1-6)

## Executive Summary

Refactor Jetbox from dual architecture (context strategies + enhancements) into a unified **composable agent behaviors system**.

**Key Changes**:
- Everything is an **AgentBehavior** (renamed from "Enhancement")
- Context strategies → Context management behaviors
- Tool definitions → Tool behaviors
- Utilities → Utility behaviors
- Agents become behavior orchestrators (no business logic)

**Core Principle**: All agent capabilities provided by composable behaviors that inject context, provide tools, and handle events.

---

## 1. Current State Analysis

### Current Context Strategies (context_strategies.py)

1. **HierarchicalStrategy** (lines 326-651)
   - Goal/Task/Subtask hierarchy
   - Last N messages (default 12)
   - Clear on subtask transitions
   - Tools: decompose_task, mark_subtask_complete
   - Loop detection, jetbox notes enabled

2. **AppendUntilFullStrategy** (lines 653-891)
   - Append all messages until token limit
   - Compacts at 75% via LLM summarization
   - Tools: mark_goal_complete
   - No subtask transitions

3. **SubAgentStrategy** (lines 893-1170)
   - For delegated work
   - 128K token limit
   - Tools: mark_complete, mark_failed
   - "DELEGATED GOAL" injection
   - Timeout nudging

4. **ArchitectStrategy** (lines 1172-1343)
   - Architecture design conversations
   - 32K token limit
   - Optimized for verbose discussions
   - No jetbox notes (artifacts are output)

### Current Enhancements

1. **JetboxNotesEnhancement** (lines 1414-1505)
   - Injects notes summary into context
   - No tools

2. **TaskManagementEnhancement** (lines 1507-1653)
   - Injects task breakdown status
   - Tools: read_task_breakdown, get_next_task, mark_task_status, update_task

### Business Logic in Agent Files

**task_executor_agent.py**:
- Tools: write_file, read_file, list_dir, run_bash, server management (lines 104-149)
- Tool dispatch (lines 189-262)
- LLM timeout nudging (lines 605-665)
- Loop detection injection (lines 247-257)

**orchestrator_agent.py**:
- Tools: consult_architect, delegate_to_executor (lines 77-243)
- Clarification tools: clarify_with_user, create_task_plan
- Workspace tools: list_workspaces, find_workspace

**architect_agent.py**:
- Tools: write_architecture_doc, write_module_spec, write_task_list (lines 233-258)
- Tool dispatch (lines 195-223, 352-382)

**Scattered tool definitions**:
- tools.py: File operations, command execution, server management
- architect_tools.py: Architecture artifact creation
- task_management_tools.py: Task CRUD operations

### Problems to Solve

1. **Dual architecture confusion** - Strategies vs enhancements
2. **Business logic in agents** - Tools shouldn't be in agent files
3. **Code duplication** - Message summarization, loop detection, compaction
4. **Not composable** - Can't mix strategies easily
5. **Hard to test** - Tools tightly coupled to agents

---

## 2. Target Architecture

### AgentBehavior Base Class

```python
from abc import ABC, abstractmethod
from typing import Any

class AgentBehavior(ABC):
    """
    Base class for composable agent behaviors.

    Behaviors extend agent capabilities through:
    - Context injection (enhance_context)
    - Tool registration (get_tools)
    - Event handling (on_* methods)

    All behaviors are:
    - Self-contained (no dependencies on other behaviors)
    - Composable (multiple behaviors work together)
    - Optional (agents choose which behaviors to use)
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return behavior identifier (e.g., 'file_tools', 'loop_detection')."""
        pass

    # Context injection
    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Modify context before LLM call.

        Args:
            context: Current context (system + messages)
            **kwargs: agent, workspace, round_number, context_manager

        Returns:
            Modified context
        """
        return context  # Default: no modification

    # Tool registration
    def get_tools(self) -> list[dict[str, Any]]:
        """Return tools provided by this behavior."""
        return []  # Default: no tools

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """
        Handle tool call for this behavior's tools.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: agent, workspace, context_manager

        Returns:
            Tool result dict
        """
        raise NotImplementedError(
            f"Behavior {self.get_name()} does not handle tool: {tool_name}"
        )

    # Instructions
    def get_instructions(self) -> str:
        """Return instructions to add to system prompt."""
        return ""  # Default: no instructions

    # Event handlers
    def on_goal_start(self, goal: str, **kwargs) -> None:
        """Called when goal starts."""
        pass

    def on_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: dict,
        **kwargs
    ) -> None:
        """Called after each tool execution."""
        pass

    def on_round_end(self, round_number: int, **kwargs) -> None:
        """Called at end of each round."""
        pass

    def on_timeout(self, elapsed_seconds: float, **kwargs) -> None:
        """Called when goal times out."""
        pass

    def on_goal_complete(self, success: bool, **kwargs) -> None:
        """Called when goal completes."""
        pass
```

### Event System

Events are triggered by agent's main loop:

1. **on_goal_start** - When agent.set_goal() or agent.run() called
2. **on_tool_call** - After dispatch_tool() completes (for loop detection, stats)
3. **on_round_end** - After each LLM call + tool execution cycle
4. **on_timeout** - When goal wall-clock limit exceeded
5. **on_goal_complete** - When goal succeeds or fails

Events delivered to all behaviors in registration order.

### Behavior Composition

**Composition rules**:
1. **Context enhancement order matters** - Behaviors modify context in registration order
2. **Tool namespaces must be unique** - Each behavior owns its tool names
3. **Events are independent** - Behaviors don't see each other's event handlers
4. **No inter-behavior dependencies** - Behaviors can't require other behaviors

**Conflict resolution**:
- Tool name conflicts → Error at registration time
- Context injection → All behaviors get to modify (in order)

### Generic Agent Base Class

```python
class BaseAgent(ABC):
    """Base agent with behavior support."""

    def __init__(self, name: str, workspace: Path, config: Any):
        self.name = name
        self.workspace = workspace
        self.config = config
        self.behaviors: list[AgentBehavior] = []
        self.tool_registry: dict[str, AgentBehavior] = {}

    def add_behavior(self, behavior: AgentBehavior) -> None:
        """Register a behavior with this agent."""
        # Check for tool name conflicts
        for tool in behavior.get_tools():
            tool_name = tool["function"]["name"]
            if tool_name in self.tool_registry:
                raise ValueError(
                    f"Tool '{tool_name}' already registered by "
                    f"{self.tool_registry[tool_name].get_name()}"
                )
            self.tool_registry[tool_name] = behavior

        self.behaviors.append(behavior)
        behavior.on_goal_start(goal=..., agent=self, workspace=self.workspace)

    def get_tools(self) -> list[dict[str, Any]]:
        """Collect tools from all behaviors."""
        tools = []
        for behavior in self.behaviors:
            tools.extend(behavior.get_tools())
        return tools

    def build_context(self) -> list[dict[str, Any]]:
        """Build context with behavior modifications."""
        context = [
            {"role": "system", "content": self.get_system_prompt()},
            *self.state.messages
        ]

        # Let behaviors modify context
        for behavior in self.behaviors:
            context = behavior.enhance_context(
                context,
                agent=self,
                workspace=self.workspace,
                round_number=self.state.total_rounds
            )

        return context

    def dispatch_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Dispatch tool call to appropriate behavior."""
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        # Find behavior that owns this tool
        behavior = self.tool_registry.get(tool_name)
        if not behavior:
            return {"error": f"Unknown tool: {tool_name}"}

        # Dispatch to behavior
        result = behavior.dispatch_tool(
            tool_name=tool_name,
            args=args,
            agent=self,
            workspace=self.workspace,
            context_manager=getattr(self, 'context_manager', None)
        )

        # Notify all behaviors of tool call
        for beh in self.behaviors:
            beh.on_tool_call(
                tool_name=tool_name,
                args=args,
                result=result,
                agent=self
            )

        return result
```

---

## 3. Migration Strategy

### Context Strategies → Context Management Behaviors

1. **HierarchicalStrategy** → **HierarchicalContextBehavior**
   - Context hierarchy injection
   - Last N messages
   - Tools: decompose_task, mark_subtask_complete
   - Clear on subtask transitions

2. **AppendUntilFullStrategy + ContextCompaction** → **CompactWhenNearFullBehavior**
   - Append all messages
   - **Compact at 75% via LLM summarization** (merged from ContextCompaction)
   - Tools: mark_goal_complete
   - Token limit: 8K default

3. **SubAgentStrategy** → **SubAgentContextBehavior**
   - DELEGATED GOAL injection
   - Tools: mark_complete, mark_failed
   - Token limit: 128K
   - Timeout nudging

4. **ArchitectStrategy** → **ArchitectContextBehavior**
   - Higher token limit (32K)
   - Architecture-focused summarization
   - No jetbox notes

### Agent Tools → Tool Behaviors

1. **File tools** (from tools.py) → **FileToolsBehavior**
   - write_file, read_file, list_dir
   - Workspace-aware path resolution
   - Safety checks, ledger logging

2. **Command tools** (from tools.py) → **CommandToolsBehavior**
   - run_bash tool
   - Command whitelist (configurable)
   - Output capture

3. **Server tools** (from tools.py) → **ServerToolsBehavior**
   - start_server, stop_server, check_server, list_servers

4. **Delegation tools** (from orchestrator_agent.py) → **DelegationBehavior**
   - consult_architect, delegate_to_executor
   - find_workspace, list_workspaces

5. **Architecture tools** (from architect_tools.py) → **ArchitectToolsBehavior**
   - write_architecture_doc, write_module_spec, write_task_list

### New Behaviors

1. **LoopDetectionBehavior** (extract from ContextStrategy base)
   - record_action() on every tool call
   - Inject loop warnings into context

2. **StatusDisplayBehavior** (extract from agent.py)
   - on_round_end: render status
   - Track LLM timing, tokens, success rate

3. **WorkspaceManagementBehavior**
   - Initialize workspace on goal start
   - Path resolution

4. **JetboxNotesBehavior** (already exists as enhancement)
   - on_goal_complete: create summary
   - on_timeout: create timeout summary
   - enhance_context: inject notes

5. **TaskManagementBehavior** (already exists as enhancement)
   - enhance_context: inject task status
   - Tools: task CRUD operations

---

## 4. Implementation Plan

### Phase 0: Documentation and Planning ✅ COMPLETE

**Deliverables**:
- [x] This document created
- [x] Plan reviewed and approved
- [x] Git commit: "Add agent behaviors refactoring plan"

---

### Phase 1: Create AgentBehavior Base Class (Week 1)

**Status**: NOT STARTED
**Estimated Effort**: 3-5 days

**Files to create**:
- `behaviors/base.py` - AgentBehavior base class
- `behaviors/__init__.py` - Export AgentBehavior
- `tests/test_agent_behavior_base.py` - Unit tests

**Tasks**:
- [ ] Define AgentBehavior abstract base class with all methods
- [ ] Define event signatures (on_goal_start, on_tool_call, etc.)
- [ ] Add comprehensive docstrings and type hints
- [ ] Write unit tests for AgentBehavior interface
- [ ] Test that behaviors can be instantiated and registered

**Validation**:
- [ ] Can create a minimal behavior that does nothing
- [ ] Can register behavior with an agent (stub)
- [ ] Event methods are called at the right times (stub)
- [ ] All tests pass

**Git Checkpoint**: `git commit -m "Phase 1: Add AgentBehavior base class and event system"`

---

### Phase 2: Extract Tools from Agents (Week 2)

**Status**: NOT STARTED
**Estimated Effort**: 5-7 days

**Files to create**:
- `behaviors/file_tools.py` - FileToolsBehavior
- `behaviors/command_tools.py` - CommandToolsBehavior
- `behaviors/server_tools.py` - ServerToolsBehavior
- `behaviors/delegation_tools.py` - DelegationBehavior
- `behaviors/architect_tools.py` - ArchitectToolsBehavior
- `tests/test_file_tools_behavior.py` - Tests
- `tests/test_command_tools_behavior.py` - Tests
- `tests/test_delegation_behavior.py` - Tests

**Tasks**:
- [ ] Extract file tool definitions from tools.py → FileToolsBehavior
- [ ] Extract command tool from tools.py → CommandToolsBehavior
- [ ] Extract server tools from tools.py → ServerToolsBehavior
- [ ] Extract delegation tools from orchestrator_agent.py → DelegationBehavior
- [ ] Extract architecture tools from architect_tools.py → ArchitectToolsBehavior
- [ ] Implement get_tools() and dispatch_tool() for each behavior
- [ ] Add workspace/ledger dependency injection
- [ ] Update tools.py to be compatibility wrapper (deprecated)
- [ ] Write unit tests for each tool behavior

**Validation**:
- [ ] Tools work identically when called via behavior
- [ ] Workspace path resolution works correctly
- [ ] Ledger logging still functions
- [ ] All existing tests pass
- [ ] Parameter invention tolerance (**kwargs) preserved

**Git Checkpoint**: `git commit -m "Phase 2: Extract tool behaviors from agent files"`

---

### Phase 3: Convert Context Strategies to Behaviors (Week 3)

**Status**: NOT STARTED
**Estimated Effort**: 5-7 days

**Files to create**:
- `behaviors/hierarchical_context.py` - HierarchicalContextBehavior
- `behaviors/compact_when_near_full.py` - CompactWhenNearFullBehavior (merged AppendUntilFull + ContextCompaction)
- `behaviors/subagent_context.py` - SubAgentContextBehavior
- `behaviors/architect_context.py` - ArchitectContextBehavior
- `behaviors/loop_detection.py` - LoopDetectionBehavior
- `tests/test_hierarchical_context_behavior.py` - Tests
- `tests/test_compact_when_near_full_behavior.py` - Tests
- `tests/test_loop_detection_behavior.py` - Tests

**Tasks**:
- [ ] Extract HierarchicalStrategy build_context() → HierarchicalContextBehavior.enhance_context()
- [ ] **Merge AppendUntilFullStrategy + message summarization → CompactWhenNearFullBehavior**
- [ ] Extract SubAgentStrategy → SubAgentContextBehavior
- [ ] Extract ArchitectStrategy → ArchitectContextBehavior
- [ ] Extract loop detection from ContextStrategy base → LoopDetectionBehavior
- [ ] Extract strategy tools into get_tools()
- [ ] Extract strategy instructions into get_instructions()
- [ ] Write unit tests for each context behavior
- [ ] Test behavior composition (multiple context behaviors)

**Validation**:
- [ ] Context built by behavior matches old strategy output
- [ ] Tools work identically
- [ ] Loop detection still triggers correctly
- [ ] CompactWhenNearFullBehavior summarizes messages at 75% threshold
- [ ] All tests pass

**Git Checkpoint**: `git commit -m "Phase 3: Convert context strategies to behaviors"`

---

### Phase 4: Update Agents to Use Behaviors (Week 4)

**Status**: NOT STARTED
**Estimated Effort**: 5-7 days

**Files to modify**:
- `base_agent.py` - Add behavior support
- `task_executor_agent.py` - Use behaviors instead of strategies
- `orchestrator_agent.py` - Use behaviors
- `architect_agent.py` - Use behaviors

**Tasks**:
- [ ] Update BaseAgent with add_behavior() and behavior-aware dispatch
- [ ] Update TaskExecutorAgent to register behaviors instead of strategy
- [ ] Update OrchestratorAgent to register behaviors
- [ ] Update ArchitectAgent to register behaviors
- [ ] Update get_tools(), build_context(), dispatch_tool() to use behaviors
- [ ] Write integration tests for each agent with behaviors
- [ ] Test behavior composition (multiple behaviors together)
- [ ] Performance benchmarking (compare to old system)

**Example TaskExecutorAgent**:
```python
class TaskExecutorAgent(BaseAgent):
    def __init__(self, workspace, goal, **kwargs):
        super().__init__(name="task_executor", workspace=workspace)

        # Add context management behavior
        self.add_behavior(SubAgentContextBehavior())

        # Add tool behaviors
        self.add_behavior(FileToolsBehavior())
        self.add_behavior(CommandToolsBehavior())
        self.add_behavior(ServerToolsBehavior())

        # Add utility behaviors
        self.add_behavior(LoopDetectionBehavior())
        self.add_behavior(JetboxNotesBehavior())
        self.add_behavior(StatusDisplayBehavior())

        if goal:
            self.set_goal(goal)
```

**Validation**:
- [ ] All existing agent tests pass
- [ ] Context matches old strategy output
- [ ] Tools work identically
- [ ] Performance is comparable (within 10%)
- [ ] L1-L6 evaluation suite passes (≥ 83.3%)

**Git Checkpoint**: `git commit -m "Phase 4: Update agents to use behavior system"`

---

### Phase 5: Remove Old Code and Add Deprecations (Week 5)

**Status**: NOT STARTED
**Estimated Effort**: 3-5 days

**Files to modify**:
- `context_strategies.py` - Mark deprecated, add wrappers
- `tools.py` - Mark deprecated, add wrappers
- `CLAUDE.md` - Update architecture documentation
- `AGENT_ARCHITECTURE.md` - Update with behaviors architecture

**Tasks**:
- [ ] Mark ContextStrategy classes as deprecated
- [ ] Add deprecation warnings to old strategy imports
- [ ] Create backward compatibility wrappers
- [ ] Update CLAUDE.md with behavior architecture
- [ ] Update AGENT_ARCHITECTURE.md
- [ ] Create migration guide for users
- [ ] Update agent_config.yaml examples

**Migration Guide**:
```yaml
# OLD (deprecated)
context:
  strategy: "hierarchical"
  history_keep: 12

# NEW
behaviors:
  context:
    type: "hierarchical"
    history_keep: 12
```

**Validation**:
- [ ] All tests pass without using old strategies
- [ ] Deprecation warnings show correctly
- [ ] Documentation is accurate
- [ ] Migration guide is clear

**Git Checkpoint**: `git commit -m "Phase 5: Deprecate old strategies, add migration guide"`

---

### Phase 6: Testing and Validation (Week 6)

**Status**: NOT STARTED
**Estimated Effort**: 5-7 days

**Tasks**:
- [ ] Run full test suite (unit + integration)
- [ ] Run evaluation suite (L1-L6 tests)
- [ ] Test edge cases (timeout, loop detection, context overflow)
- [ ] Test behavior composition (multiple behaviors together)
- [ ] Performance benchmarking (compare to old system)
- [ ] Update test reports
- [ ] Code review and cleanup
- [ ] Documentation review

**Success Criteria**:
- [ ] Pass rate ≥ 83.3% (current baseline)
- [ ] No performance regression (within 10%)
- [ ] All edge cases handled correctly
- [ ] Behaviors compose cleanly without conflicts
- [ ] Documentation complete and accurate

**Deliverables**:
- [ ] Test report (unit + integration + evaluation)
- [ ] Performance benchmark report
- [ ] Updated CLAUDE.md
- [ ] Updated AGENT_ARCHITECTURE.md
- [ ] Migration guide
- [ ] Release notes

**Git Checkpoint**: `git commit -m "Phase 6: Complete behavior refactoring - all tests pass"`

---

## 5. Key Behavior Definitions

### CompactWhenNearFullBehavior (Merged from AppendUntilFullStrategy + ContextCompaction)

**Purpose**: Append all messages until near token limit, then compact via LLM summarization.

**Features**:
- Appends all messages to context (no truncation)
- Monitors token usage (estimate via character count)
- Compacts at 75% threshold via LLM summarization
- Preserves recent messages (last 3-5 exchanges)
- Tools: mark_goal_complete

**Configuration**:
```yaml
behaviors:
  compact_when_near_full:
    max_tokens: 8000
    compact_threshold: 0.75  # 75%
    keep_recent: 5  # exchanges
```

**Implementation Notes**:
- Merge _summarize_messages() from AppendUntilFullStrategy
- Merge compaction logic from old ContextCompactionEnhancement concept
- Single behavior handles both appending and compacting
- Simpler than having two separate behaviors

---

## 6. Risks and Mitigations

### Risk: Performance regression from event overhead
- **Mitigation**: Benchmark before/after, optimize hot paths
- **Mitigation**: Events are optional (behaviors can ignore them)
- **Mitigation**: Use lazy evaluation where possible

### Risk: Behavior composition conflicts
- **Mitigation**: Enforce tool namespace uniqueness at registration
- **Mitigation**: Document behavior order dependencies
- **Mitigation**: Provide clear error messages for conflicts

### Risk: Breaking existing code during migration
- **Mitigation**: Maintain backward compatibility wrappers
- **Mitigation**: Incremental migration (one agent at a time)
- **Mitigation**: Comprehensive test coverage before migration

### Risk: Increased complexity from too many behaviors
- **Mitigation**: Provide sensible defaults (pre-configured agent classes)
- **Mitigation**: Document recommended behavior sets
- **Mitigation**: Keep behavior count manageable (10-15 max per agent)

### Risk: Loss of strategy-specific optimizations
- **Mitigation**: Preserve optimizations in behavior implementations
- **Mitigation**: Benchmark to ensure no regression
- **Mitigation**: Profile and optimize behavior code

---

## 7. Rollback Strategy

### Git Branching
- Create `agent-behaviors-refactor` branch
- Keep `main` branch stable during migration
- Only merge when all tests pass

### Feature Flags
- Environment variable `USE_BEHAVIORS=true/false`
- Agents check flag and use old or new system
- Easy rollback by setting flag to false

### Backup Old Code
- Move old strategies to `legacy/` directory
- Keep old agent implementations as `*_agent_legacy.py`
- Can revert by copying files back

### Phased Rollout
- Phase 1-3: Build behaviors in parallel (no breakage)
- Phase 4: TaskExecutor only
- Phase 4: Orchestrator only
- Phase 4: Architect only
- Rollback at any phase if issues arise

---

## 8. Benefits Summary

### What Problems Does This Solve?

1. **Eliminates dual architecture confusion** - One system (behaviors) instead of strategies + enhancements
2. **Removes business logic from agents** - Agents are behavior orchestrators
3. **Enables true composability** - Mix and match any behaviors
4. **Simplifies testing** - Test each behavior in isolation
5. **Improves discoverability** - All behaviors in behaviors/ directory

### What New Capabilities Does It Enable?

1. **Mix-and-match context strategies** - Use hierarchical + task management together
2. **Custom agent configurations** - Users create specialized agents easily
3. **Plugin architecture** - Third-party behaviors can be added
4. **Dynamic behavior loading** - Enable/disable at runtime
5. **Better separation of concerns** - Context, tools, monitoring all separate

### Code Quality Improvements

1. **Reduced duplication** - Summarization, loop detection, compaction in one place each
2. **Better encapsulation** - Each behavior owns its tools and logic
3. **Improved testability** - Unit tests for each behavior
4. **Clearer organization** - behaviors/ directory with clear structure
5. **Type safety** - AgentBehavior interface with clear types

---

## 9. Progress Tracking

| Phase | Status | Start Date | End Date | Git Commit |
|-------|--------|------------|----------|------------|
| Phase 0: Planning | ✅ COMPLETE | 2025-01-01 | 2025-01-01 | `035aba9` |
| Phase 1: Base Class | 🔲 Not Started | - | - | - |
| Phase 2: Tool Behaviors | 🔲 Not Started | - | - | - |
| Phase 3: Context Behaviors | 🔲 Not Started | - | - | - |
| Phase 4: Update Agents | 🔲 Not Started | - | - | - |
| Phase 5: Deprecations | 🔲 Not Started | - | - | - |
| Phase 6: Testing | 🔲 Not Started | - | - | - |

**Legend**: ✅ Complete | 🔄 In Progress | 🔲 Not Started | ⚠️ Blocked

---

## 10. Next Steps

1. Review and approve this plan
2. Start Phase 1: Create AgentBehavior base class
3. Commit progress at each phase milestone
4. Update this document as implementation progresses

---

**Document Version**: 1.0
**Last Updated**: 2025-01-01
**Next Review**: After Phase 1 completion
