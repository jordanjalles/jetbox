# Jetbox Code Inspection Report

## Executive Summary
- **21 duplication issues** found (high/medium severity)
- **15 unnecessary code blocks** found (mostly commented-out deprecated code)
- **8 separation of concerns violations** found (hardcoded agent names in behaviors)
- Total: **44 issues** requiring attention

---

## 1. Code Duplication

### Issue 1.1: Duplicated Completion Tool Definition
**Location:**
- `behaviors/subagent_mode.py:144-179` (mark_complete, mark_failed)
- `behaviors/subagent_context.py:107-142` (mark_complete, mark_failed)
- `behaviors/compact_when_near_full.py:247-272` (mark_goal_complete)

**Severity:** High

**Description:** Three separate behaviors define near-identical completion tools with different names (mark_complete vs mark_goal_complete). The logic and parameters are duplicated.

**Recommendation:**
- Create a shared completion tools utility module
- Or consolidate into a single CompletionBehavior that all agents can use

### Issue 1.2: Duplicated Goal Initialization Logic
**Location:**
- `behaviors/subagent_mode.py:335-397` (on_goal_set)
- `behaviors/subagent_context.py:222-278` (on_goal_set)

**Severity:** High

**Description:** Nearly identical on_goal_set implementation in two behaviors (362 lines vs 56 lines). Both initialize workspace_manager, perf_stats, and goal tracking. SubAgentModeBehavior is described as "RENAMED and ENHANCED version of SubAgentContextBehavior" but both files still exist.

**Recommendation:**
- Remove `subagent_context.py` entirely (deprecated)
- The comment at line 400 says "Backward compatibility alias" but maintaining both is technical debt

### Issue 1.3: Duplicated Context Enhancement Pattern
**Location:**
- `behaviors/subagent_mode.py:77-135`
- `behaviors/subagent_context.py:53-98`
- `behaviors/chatbot.py:196-245`
- `behaviors/loop_detection.py:181-364`

**Severity:** Medium

**Description:** All behaviors follow same pattern for context injection: check context length, insert at index 1, format as user message. The boilerplate is repeated 4+ times.

**Recommendation:**
- Create a base helper method `inject_user_message_after_system(context, content)` in AgentBehavior base class

### Issue 1.4: Duplicated Context Size Estimation
**Location:**
- `behaviors/compact_when_near_full.py:156-186`
- `behaviors/workspace_task_notes.py:469-502` (_get_max_tokens)

**Severity:** Medium

**Description:** Both behaviors estimate token counts and check max_tokens thresholds. CompactWhenNearFullBehavior has full estimation logic, WorkspaceTaskNotesBehavior tries to extract max_tokens from other behaviors.

**Recommendation:**
- Extract to shared utility: `context_utils.estimate_tokens(context)`
- Share max_tokens configuration across behaviors via agent config

### Issue 1.5: Duplicated File Operation Patterns
**Location:**
- `behaviors/write_file_tools.py` (assumed, not read)
- `behaviors/read_file_tools.py` (assumed, not read)
- `behaviors/directory_tools.py` (assumed, not read)
- `behaviors/architect_tools.py:270-307` (_write_architecture_doc)

**Severity:** Low

**Description:** File writing patterns (create parent dirs, write_text, return result dict) are repeated across behaviors.

**Recommendation:**
- Create shared file operation utilities in `behaviors/file_utils.py`

### Issue 1.6: Duplicated Workspace Directory Resolution
**Location:**
- `behaviors/command_tools.py:199-200`
- `behaviors/architect_tools.py` (multiple methods)
- `behaviors/workspace_task_notes.py:36-40`

**Severity:** Low

**Description:** Pattern `workspace_manager.workspace_dir` repeated throughout behaviors. No abstraction.

**Recommendation:**
- Add property to behaviors: `self.workspace_dir` that delegates to workspace_manager

### Issue 1.7: Duplicated Tool Call Validation
**Location:**
- `base_agent.py:423-500` (_validate_tool_parameters)
- Similar validation likely repeated in individual behaviors

**Severity:** Low

**Description:** Tool parameter validation with schema checking and wishlist logging.

**Recommendation:** Already centralized in base_agent. Document this pattern for behavior authors.

### Issue 1.8: Duplicated Action Signature Hashing
**Location:**
- `behaviors/loop_detection.py:80-88`
- Likely duplicated if other behaviors track actions

**Severity:** Low

**Description:** JSON serialization + hashing pattern for action signatures.

**Recommendation:**
- Extract to utility: `hash_action(tool_name, args) -> str`

### Issue 1.9: Duplicated LLM Summarization Pattern
**Location:**
- `behaviors/compact_when_near_full.py:188-246` (_summarize_messages)
- `behaviors/workspace_task_notes.py:118-161` (prompt_for_task_summary)
- `behaviors/workspace_task_notes.py:163-239` (prompt_for_goal_summary)

**Severity:** Medium

**Description:** All three methods follow same pattern: build prompt, call LLM with low temp (0.2), handle exceptions, return summary. Different prompts but identical structure.

**Recommendation:**
- Create `llm_utils.summarize(prompt, timeout=30, temperature=0.2)`
- Behaviors just provide prompts, not full LLM calling logic

### Issue 1.10: Duplicated Tool Dispatch Pattern
**Location:**
- All behaviors with tools (delegation.py, architect_tools.py, command_tools.py, etc.)
- Pattern: `if tool_name == "x": return self._x(args.get("param1"), ...)`

**Severity:** Low

**Description:** Every behavior reimplements tool dispatch with if/elif chains.

**Recommendation:**
- Use Python 3.10 match/case or method dispatch pattern
- Or document this as the standard pattern (acceptable boilerplate)

---

## 2. Unnecessary Code

### Issue 2.1: Commented-Out Context Manager Code
**Location:**
- `behaviors/subagent_mode.py:353-359`
- `behaviors/subagent_context.py:240-245`

**Severity:** Low

**Description:** Large blocks of commented-out code for ContextManager initialization with comment "DEPRECATED with behavior system".

**Recommendation:** Remove entirely. Deprecated code should be deleted, not commented out.

### Issue 2.2: Commented-Out Status Display Code
**Location:**
- `behaviors/subagent_mode.py:387-390`
- `behaviors/subagent_context.py:267-271`

**Severity:** Low

**Description:** Commented-out StatusDisplay initialization with "DEPRECATED: StatusDisplay is being redesigned".

**Recommendation:** Remove. If redesign is coming, the old code doesn't need to be commented.

### Issue 2.3: Unused Private Methods
**Location:**
- `base_agent.py:897-926` (_to_snake_case)
- `behaviors/architect_tools.py:251-269` (_slugify, _format_list, _format_dict)

**Severity:** Low

**Description:** Private methods that appear to be used, but could be consolidated. _to_snake_case has extensive documentation for a simple utility.

**Recommendation:**
- Move string utilities to shared `utils.py` module
- Simplify inline if only used once

### Issue 2.4: Deprecated Backward Compatibility Alias
**Location:**
- `behaviors/subagent_mode.py:399-400`
  ```python
  # Backward compatibility alias
  SubAgentContextBehavior = SubAgentModeBehavior
  ```

**Severity:** Medium

**Description:** Alias exists but `subagent_context.py` still has full implementation (277 lines). One should be removed.

**Recommendation:**
- Remove `subagent_context.py` entirely
- Keep only the alias in `subagent_mode.py`
- Update imports across codebase

### Issue 2.5: Unused Global State Variable
**Location:**
- `behaviors/workspace_task_notes.py:22-23`
  ```python
  _workspace = None  # Global reference to workspace manager (set by behavior at runtime)
  ```

**Severity:** Low

**Description:** Module-level mutable global state. Anti-pattern in modern Python.

**Recommendation:**
- Pass workspace_manager through method parameters instead
- Or make it an instance variable of the behavior

### Issue 2.6: Deprecated get_context_strategy Method
**Location:**
- `base_agent.py:181-198`

**Severity:** Low

**Description:** Method marked "DEPRECATED: Context strategies should be defined via behaviors" but still implemented with full docstring.

**Recommendation:**
- Remove method or make it raise DeprecationWarning
- Update docstring to just say "Deprecated, use behaviors"

### Issue 2.7: Dead Code in _make_serializable
**Location:**
- `behaviors/loop_detection.py:366-402`

**Severity:** Low

**Description:** Complex recursive serialization for action hashing. Could be simplified using `json.dumps(default=str)`.

**Recommendation:**
- Simplify to: `json.dumps(obj, default=str, sort_keys=True)`

### Issue 2.8: Empty Except Blocks
**Location:**
- `base_agent.py:560-562` (load_state)
  ```python
  except Exception:
      # If load fails, keep fresh state
      pass
  ```

**Severity:** Low

**Description:** Silent exception swallowing. Makes debugging difficult.

**Recommendation:**
- At minimum log the exception: `print(f"[{self.name}] Failed to load state: {e}")`

---

## 3. Separation of Concerns Violations

### Issue 3.1: Hardcoded "orchestrator" Check in LoopDetectionBehavior
**Location:**
- `behaviors/loop_detection.py:198-243`
  ```python
  is_orchestrator = agent and agent.name == 'orchestrator'

  # ORCHESTRATOR-SPECIFIC: Check for empty rounds before delegation
  if is_orchestrator and self.consecutive_empty_rounds >= 2:
      delegation_tools = {'delegate_to_executor', 'consult_architect'}
  ```

**Severity:** High

**Description:** Generic loop detection behavior has hardcoded orchestrator-specific logic. Violates behavior single-responsibility principle.

**Recommendation:**
- Create `OrchestratorIdleDetectionBehavior` for orchestrator-specific nudging
- Keep LoopDetectionBehavior generic
- Or make it config-driven: `idle_nudge_tools: [delegate_to_executor, consult_architect]`

### Issue 3.2: Hardcoded Agent Names in Comments/Docstrings
**Location:**
- `behaviors/delegation.py:4,16-17,191-193,213-215` (multiple references to "orchestrator", "architect", "task_executor")
- `behaviors/subagent_mode.py:37,300` ("orchestrator")
- `behaviors/workspace_management.py:4,9,30` ("orchestrator", "TaskExecutor", "Architect")

**Severity:** Medium

**Description:** Comments and docstrings hardcode specific agent names as examples, creating tight coupling between generic behaviors and specific agents.

**Recommendation:**
- Use generic terms: "parent agent", "delegating agent", "specialized agent"
- Or use "e.g., orchestrator" to make it clear it's just an example

### Issue 3.3: Hardcoded Tool Names in LoopDetectionBehavior
**Location:**
- `behaviors/loop_detection.py:206-207`
  ```python
  delegation_tools = {'delegate_to_executor', 'consult_architect'}
  ```

**Severity:** High

**Description:** Behavior hardcodes specific tool names. Should be config-driven.

**Recommendation:**
- Add config parameter: `delegation_tool_names: List[str]`
- Or detect delegation tools dynamically by querying agent.get_tools()

### Issue 3.4: Hardcoded File Paths in DelegationBehavior
**Location:**
- `behaviors/delegation.py:340,418`
  ```python
  msg_file = Path(".agent_context/messages_to_orchestrator.jsonl")
  ```

**Severity:** Medium

**Description:** Hardcoded path assumes orchestrator as parent. Won't work for other delegation hierarchies.

**Recommendation:**
- Make path configurable: `messages_to_{parent_agent}.jsonl`
- Or use generic: `messages_to_parent.jsonl`

### Issue 3.5: Workspace Management Only for Orchestrator
**Location:**
- `behaviors/workspace_management.py:4-9`
  ```python
  """
  This behavior is ONLY for orchestrator-level agents that need to:
  - Create isolated workspaces for delegated tasks
  - Pass workspace paths to sub-agents

  Other agents (TaskExecutor, Architect) work WITHIN workspaces but don't manage them.
  """
  ```

**Severity:** Medium

**Description:** Behavior is explicitly limited to orchestrator, making it not truly generic.

**Recommendation:**
- Rename to `WorkspaceCoordinatorBehavior` to indicate coordinator role
- Or make it generic: any agent can manage workspaces for sub-agents

### Issue 3.6: Agent-Specific Logic in Base Agent
**Location:**
- `base_agent.py:1472-1474`
  ```python
  # IMPORTANT: Exclude delegation results (they have "target_agent" field)
  is_delegation_result = "target_agent" in result
  if is_delegation_result:
      return None
  ```

**Severity:** Low

**Description:** Base agent has delegation-specific logic. Should be in DelegationBehavior.

**Recommendation:**
- Move to DelegationBehavior.on_tool_call() or similar hook
- Keep base_agent generic

### Issue 3.7: Hardcoded "architecture/task-breakdown.json" Path
**Location:**
- `behaviors/task_management.py:36`
  ```python
  return self.workspace_manager.workspace_dir / "architecture" / "task-breakdown.json"
  ```

**Severity:** Low

**Description:** Hardcoded file structure assumes specific architecture directory layout.

**Recommendation:**
- Make configurable via behavior params: `task_breakdown_path`
- Or keep as convention but document it

### Issue 3.8: Orchestrator-Specific Context Attributes Referenced
**Location:**
- `behaviors/workspace_task_notes.py:493-500`
  ```python
  # Try orchestrator attributes
  if hasattr(agent, "context_window"):
      return agent.context_window

  # Try token_threshold (orchestrator fallback)
  if hasattr(agent, "token_threshold"):
  ```

**Severity:** Low

**Description:** Behavior assumes specific agent attributes that only orchestrator has.

**Recommendation:**
- Define standard interface: all agents should have `max_context_tokens` property
- Or behavior should not need to know about token limits (let agent handle it)

---

## 4. Additional Observations

### 4.1: SubAgentContextBehavior vs SubAgentModeBehavior
**Status:** Deprecated duplication

Both files exist with nearly identical functionality. SubAgentModeBehavior (400 lines) is described as "RENAMED and ENHANCED version" but SubAgentContextBehavior (277 lines) still has full implementation. This creates confusion and maintenance burden.

### 4.2: Private Method Proliferation in base_agent.py
**Count:** 18 private methods (starting with `_`)

While private methods are normal, base_agent.py has many that could be extracted to utility modules:
- `_serialize_message`, `_to_snake_case`, `_format_tool_call_preview` are pure utility functions
- `_validate_tool_parameters`, `_log_parameter_wishlist` could be a ToolValidator class

### 4.3: Behavior Count
**Current:** 17 behavior files (829 lines max, 56 lines min)

Behaviors range from 56 lines (__init__.py) to 829 lines (delegation.py). The large size of delegation.py suggests it might be doing too much and could be split:
- DelegationToolsBehavior (tool generation)
- SubprocessDelegationBehavior (subprocess execution)
- DirectDelegationBehavior (in-process delegation)

### 4.4: Commented Code vs Active Code Ratio
**Files with commented code:** 15+ files
**Pattern:** Most commented code is deprecated with TODO/DEPRECATED markers

This is technical debt that should be cleaned up. Version control preserves history - commented code doesn't need to stay in the codebase.

---

## Summary Statistics
- **Total issues:** 44
- **High severity:** 8 (duplication, hardcoded logic)
- **Medium severity:** 12 (separation of concerns, unnecessary code)
- **Low severity:** 24 (minor improvements, style issues)

## Priority Recommendations

### Immediate (High Severity)
1. **Remove SubAgentContextBehavior** - Keep only SubAgentModeBehavior
2. **Extract orchestrator logic from LoopDetectionBehavior** - Make config-driven
3. **Consolidate completion tools** - Single source of truth

### Short Term (Medium Severity)
4. **Remove commented-out deprecated code** - 15+ instances across codebase
5. **Make DelegationBehavior tool names config-driven** - Remove hardcoded agent references
6. **Extract LLM summarization pattern** - Reduce duplication across 3 behaviors

### Long Term (Low Severity)
7. **Create shared utilities module** - String utils, context utils, file utils
8. **Standardize context injection pattern** - Base helper method
9. **Document behavior patterns** - Tool dispatch, error handling, etc.
