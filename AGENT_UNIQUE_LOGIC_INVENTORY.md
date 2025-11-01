# Agent Unique Logic Inventory

**Date**: 2025-11-01
**Purpose**: Document what logic is truly unique to each agent class vs what should be in base_agent

---

## TaskExecutorAgent (task_executor_agent.py)

### Truly Unique Logic (Keep in agent class)

1. **set_goal() method** (lines 384-443)
   - **Why unique**: Sets up hierarchical task management, workspace manager, jetbox notes
   - **Dependencies**: Context manager goal setup, workspace creation logic, behavior events
   - **Keep**: Yes - this is specific to TaskExecutor's workflow

2. **run() method** (lines 614-938)
   - **Why unique**: Complex execution loop with status display, timeout handling, completion detection, context isolation
   - **Dependencies**: Context manager, status display, completion detector
   - **Keep**: Yes - each agent type has different run loop requirements

3. **Timeout handling** (lines 454-590)
   - **Why unique**: Goal-level timeouts with jetbox notes, context dumps, task tree serialization
   - **Dependencies**: Workspace manager, jetbox notes, context manager task tree
   - **Keep**: Yes - specific to TaskExecutor's goal-based execution

4. **Goal success/failure handlers** (lines 940-979)
   - **Why unique**: Jetbox notes integration for goal completion
   - **Dependencies**: Jetbox notes system
   - **Keep**: Yes - TaskExecutor-specific workflow

5. **Cleanup method** (lines 981-992)
   - **Why unique**: Clears Ollama context after task completion
   - **Dependencies**: LLM utils
   - **Keep**: Yes - TaskExecutor-specific cleanup

6. **LLM caller for jetbox** (lines 445-452)
   - **Why unique**: Wrapper for jetbox notes LLM calls
   - **Dependencies**: LLM utils
   - **Keep**: Yes - TaskExecutor-specific helper

### Logic That Should Move to BaseAgent

1. **dispatch_tool() legacy path** (lines 236-316)
   - **Problem**: Duplicates tool mapping and context injection logic
   - **Solution**: Move to base_agent as default implementation
   - **Justification**: This is boilerplate that every agent needs

2. **build_context() behavior system path** (lines 318-366)
   - **Problem**: Identical pattern in all three agents
   - **Solution**: Move behavior system path to base_agent
   - **Justification**: All agents use same pattern for behaviors

3. **get_system_prompt() behavior system path** (lines 178-230)
   - **Problem**: Identical pattern in all three agents
   - **Solution**: Move behavior instructions/tool docs logic to base_agent
   - **Justification**: All agents follow same pattern

4. **get_tools() behavior system path** (lines 123-176)
   - **Problem**: All agents check use_behaviors and call get_behavior_tools()
   - **Solution**: Move to base_agent as default implementation
   - **Justification**: Reduces duplication

---

## OrchestratorAgent (orchestrator_main.py)

### Truly Unique Logic (Keep in agent class)

1. **Delegation tracking** (line 66)
   - **Why unique**: Tracks tasks delegated to sub-agents
   - **Dependencies**: None
   - **Keep**: Yes - orchestrator-specific state

2. **Token estimation & compaction** (lines 616-886)
   - **Why unique**: Legacy context compaction with LLM summarization
   - **Dependencies**: LLM utils
   - **Keep**: Yes - orchestrator-specific legacy feature (deprecated but still supported)

3. **Model context window detection** (lines 570-614)
   - **Why unique**: Queries Ollama for num_ctx parameter
   - **Dependencies**: Ollama API
   - **Keep**: Yes - orchestrator-specific optimization

4. **Task management auto-add** (lines 532-568)
   - **Why unique**: Auto-detects task breakdown files and adds enhancement
   - **Dependencies**: Task management tools, workspace manager
   - **Keep**: Yes - orchestrator-specific workflow

5. **execute_round() method** (lines 888-902)
   - **Why unique**: Simple round execution for orchestrator
   - **Dependencies**: None
   - **Keep**: Yes - orchestrator doesn't have complex run loop like TaskExecutor

6. **Conversation summary** (lines 913-929)
   - **Why unique**: Returns orchestrator-specific stats
   - **Dependencies**: None
   - **Keep**: Yes - orchestrator-specific reporting

### Logic That Should Move to BaseAgent

1. **build_context() behavior path** (lines 498-504)
   - **Problem**: Identical to TaskExecutor
   - **Solution**: Move to base_agent

2. **get_system_prompt() behavior path** (lines 292-307)
   - **Problem**: Identical to TaskExecutor
   - **Solution**: Move to base_agent

3. **get_tools() behavior path** (lines 121-123)
   - **Problem**: Identical to TaskExecutor
   - **Solution**: Move to base_agent

---

## ArchitectAgent (architect_agent.py)

### Truly Unique Logic (Keep in agent class)

1. **configure_workspace() method** (lines 185-223)
   - **Why unique**: Configures architect tools with workspace, auto-adds task management
   - **Dependencies**: Architect tools, task management tools
   - **Keep**: Yes - architect-specific setup

2. **set_project() method** (lines 254-261)
   - **Why unique**: Sets project description for architecture work
   - **Dependencies**: Context manager
   - **Keep**: Yes - architect-specific workflow

3. **dispatch_tool() architect tools** (lines 431-461)
   - **Why unique**: Maps architect-specific tools (write_architecture_doc, etc.)
   - **Dependencies**: Architect tools module
   - **Keep**: Yes - architect has unique tools

4. **consult() method** (lines 463-562)
   - **Why unique**: Main entry point for architecture consultation
   - **Dependencies**: Architect tools
   - **Keep**: Yes - architect-specific workflow

5. **Task management auto-add** (lines 388-429)
   - **Why unique**: Auto-detects task breakdown and adds enhancement (legacy mode only)
   - **Dependencies**: Task management enhancement
   - **Keep**: Yes - architect-specific workflow

### Logic That Should Move to BaseAgent

1. **build_context() behavior path** (lines 354-360)
   - **Problem**: Identical to TaskExecutor and Orchestrator
   - **Solution**: Move to base_agent

2. **get_system_prompt() behavior path** (lines 309-321)
   - **Problem**: Identical to TaskExecutor and Orchestrator
   - **Solution**: Move to base_agent

3. **get_tools() behavior path** (lines 277-279)
   - **Problem**: Identical to TaskExecutor and Orchestrator
   - **Solution**: Move to base_agent

---

## Summary

### Code to Move to BaseAgent

1. **Default build_context() for behavior system** (~15 lines)
   - Pattern: Build basic context, call enhance_context_with_behaviors()
   - Agents can override if they need legacy strategy support

2. **Default get_system_prompt() for behavior system** (~10 lines)
   - Pattern: Get config prompt or fallback, add behavior instructions, add tool docs
   - Agents can override to provide fallback prompt

3. **Default get_tools() for behavior system** (~3 lines)
   - Pattern: Check use_behaviors, return get_behavior_tools()
   - Agents can override for legacy strategy support

4. **Default dispatch_tool() for behavior system** (~3 lines)
   - Pattern: Check use_behaviors, call dispatch_tool_to_behavior()
   - Agents can override for legacy tool dispatch

### Line Count Reduction Estimate

- **TaskExecutorAgent**: Currently ~1011 lines → Target ~900 lines (10% reduction)
- **OrchestratorAgent**: Currently ~930 lines → Target ~850 lines (8% reduction)
- **ArchitectAgent**: Currently ~563 lines → Target ~520 lines (7% reduction)

### Benefits

1. **DRY**: Eliminates 3x duplication of behavior system boilerplate
2. **Maintainability**: Changes to behavior system only need to happen in base_agent
3. **Clarity**: Agent classes become more focused on their unique logic
4. **Testing**: Easier to test common patterns in base_agent

### Risks

1. **Breaking legacy mode**: Need to ensure legacy strategy/enhancement system still works
2. **Breaking behavior override**: Some agents might need to override base behavior
3. **Testing**: Need to verify diagnose_completion_issue.py still passes

---

## Decision: What to Refactor Now

### Priority 1 (Do Now)
- ✅ Move default behavior system methods to base_agent
- ✅ Keep legacy paths in agent classes (for backward compatibility)
- ✅ Add docstrings explaining override pattern

### Priority 2 (Future)
- Move legacy strategy/enhancement code to base_agent (more complex, less urgent)
- Standardize run() loops (very agent-specific, risky)
- Remove legacy mode entirely (breaking change, v2.0)

---

*Created: 2025-11-01*
*Status: Analysis complete, ready for refactoring*
