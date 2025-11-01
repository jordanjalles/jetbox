# Unified Delegation System

**Date**: 2025-11-01
**Status**: ✅ IMPLEMENTED

## Overview

The Jetbox agent system now uses a **unified bidirectional delegation system** where DelegationBehavior handles BOTH directions of delegation:

1. **Delegating TO other agents** (delegator mode)
2. **Being delegated to BY other agents** (delegatee mode)

This replaces the previous dual-behavior system (DelegationBehavior + SubAgentContextBehavior) with a single, composable behavior.

## Architecture

### DelegationBehavior - Unified Bidirectional Delegation

**File**: `behaviors/delegation.py`

**Modes**:
- **Delegator mode**: Agent can delegate to others
  - Provides delegation tools: `consult_X`, `delegate_to_X`
  - Injects agent descriptions into context
  - Tracks delegations for reporting

- **Delegatee mode**: Agent can receive delegated work
  - Provides completion tools: `mark_complete`, `mark_failed`
  - Injects "DELEGATED GOAL" context header
  - Handles goal initialization via `on_goal_set` event

**Configuration**:
```python
DelegationBehavior(
    agent_relationships=relationships_dict,
    is_delegatee=True/False  # Controls delegatee mode
)
```

### Auto-Configuration via agents.yaml

**File**: `agents.yaml`

Defines which agents can delegate to which:

```yaml
agents:
  orchestrator:
    class: OrchestratorAgent
    can_delegate_to:
      - architect
      - task_executor

  architect:
    class: ArchitectAgent
    can_delegate_to: []  # Terminal agent (doesn't delegate)

  task_executor:
    class: TaskExecutorAgent
    can_delegate_to: []  # Terminal agent (doesn't delegate)
```

**Auto-detection logic** (in `base_agent.py`):
1. Check if agent has `can_delegate_to` list → enables delegator mode
2. Check if agent appears in another agent's `can_delegate_to` list → enables delegatee mode
3. Auto-add DelegationBehavior with appropriate `is_delegatee` flag

### Current Agent Modes

| Agent | Delegator? | Delegatee? | Tools Provided |
|-------|-----------|-----------|----------------|
| **Orchestrator** | ✅ Yes (architect, task_executor) | ❌ No | `consult_architect`, `delegate_to_executor` |
| **Architect** | ❌ No | ✅ Yes | `mark_complete`, `mark_failed` |
| **TaskExecutor** | ❌ No | ✅ Yes | `mark_complete`, `mark_failed` |

**Note**: Orchestrator could also be made delegatable in the future by adding it to another agent's `can_delegate_to` list, enabling multi-level orchestration.

## Completion Tools

The system now provides TWO types of completion tools:

### 1. mark_complete / mark_failed (DelegationBehavior - delegatee mode)
**Purpose**: Signal completion of **delegated work**
**When to use**: Agent was invoked by another agent via delegation tool
**Behavior**: Marks goal as complete/failed, returns result to delegating agent

### 2. mark_goal_complete (CompactWhenNearFullBehavior)
**Purpose**: Signal completion of **standalone work**
**When to use**: Agent was invoked directly (CLI, standalone execution)
**Behavior**: Marks goal as complete, triggers goal summary, exits agent

**Why both exist**:
- Agents can work in BOTH modes (delegated and standalone)
- Different completion semantics for different invocation modes
- Flexible architecture supports multiple usage patterns

## Benefits

### 1. Simplicity
- Single behavior instead of two separate ones
- Clear separation of concerns
- Unified configuration

### 2. Composability
- All agents can participate in delegation (no special cases)
- Behaviors work independently
- Mix and match for different agent roles

### 3. Config-Driven
- No hardcoded delegation logic in agent classes
- agents.yaml defines relationships
- Individual agent configs define delegation tools

### 4. Backward Compatibility
- SubAgentContextBehavior deprecated but kept as no-op stub
- Existing code using SubAgentContextBehavior continues to work
- Migration path available (see Migration Notes below)

## Migration from Old System

### Old System (Deprecated)
```python
# Orchestrator only
- DelegationBehavior (delegator only)

# TaskExecutor, Architect only
- SubAgentContextBehavior (delegatee only)
```

### New System
```python
# All agents
- DelegationBehavior (unified bidirectional)
  - is_delegatee=True/False controls mode
  - Auto-configured via agents.yaml
```

### Migration Steps

1. **Remove manual behavior additions**: Auto-add handles everything
2. **Update code references**: SubAgentContextBehavior → DelegationBehavior
3. **Verify agents.yaml**: Ensure relationships are correct
4. **Test delegation flows**: Verify delegator → delegatee works

## Code Examples

### Delegator Agent (Orchestrator)
```python
orchestrator = OrchestratorAgent(workspace=".agent_workspace/orch", use_behaviors=True)

# Auto-added DelegationBehavior with:
#   - is_delegatee=False
#   - can_delegate_to=['architect', 'task_executor']
#   - Tools: consult_architect, delegate_to_executor

# Can delegate to TaskExecutor:
result = orchestrator.run(user_request="Build a calculator app")
# Orchestrator uses delegate_to_executor tool internally
```

### Delegatee Agent (TaskExecutor)
```python
task_executor = TaskExecutorAgent(
    workspace=".agent_workspace/calc",
    goal="Create calculator.py with add/subtract functions",
    use_behaviors=True
)

# Auto-added DelegationBehavior with:
#   - is_delegatee=True
#   - can_delegate_to=[]
#   - Tools: mark_complete, mark_failed

result = task_executor.run()
# Agent completes work, calls mark_complete(summary="...")
```

### Both Modes (Hypothetical Multi-Level Orchestrator)
```yaml
# agents.yaml
senior_orchestrator:
  can_delegate_to: [orchestrator, architect, task_executor]

orchestrator:
  can_delegate_to: [architect, task_executor]
```

```python
orchestrator = OrchestratorAgent(...)

# Auto-added DelegationBehavior with:
#   - is_delegatee=True (can be delegated to by senior_orchestrator)
#   - can_delegate_to=['architect', 'task_executor'] (can delegate)
#   - Tools: consult_architect, delegate_to_executor, mark_complete, mark_failed
```

## Testing

### Verification Test
```bash
python -c "
from task_executor_agent import TaskExecutorAgent
from orchestrator_agent import OrchestratorAgent

# Test delegatee (TaskExecutor)
te = TaskExecutorAgent(workspace='.test', goal='test', use_behaviors=True)
delegation_beh = [b for b in te.behaviors if b.get_name() == 'delegation'][0]
assert delegation_beh.is_delegatee == True
assert delegation_beh.agent_relationships.get('can_delegate_to') == []
assert 'mark_complete' in [t['function']['name'] for t in delegation_beh.get_tools()]

# Test delegator (Orchestrator)
orch = OrchestratorAgent(workspace='.test_orch', use_behaviors=True)
delegation_beh = [b for b in orch.behaviors if b.get_name() == 'delegation'][0]
assert delegation_beh.is_delegatee == False
assert 'architect' in delegation_beh.agent_relationships.get('can_delegate_to')
assert 'consult_architect' in [t['function']['name'] for t in delegation_beh.get_tools()]

print('✓ All tests passed!')
"
```

## Files Modified

### Core Implementation
- `behaviors/delegation.py` - Merged SubAgentContextBehavior functionality
- `base_agent.py` - Updated `_auto_add_delegation_behavior()` to detect is_delegatee
- `base_agent.py` - Deprecated `_auto_add_subagent_context_behavior()` (now no-op stub)

### Configuration
- `agents.yaml` - Defines delegation relationships (unchanged)
- `*_config.yaml` - Individual agent configs (unchanged)

### Deprecated
- `behaviors/subagent_context.py` - Functionality merged into DelegationBehavior (file kept for backward compat)

## Future Enhancements

### 1. Multi-Level Orchestration
Enable Orchestrator to be both delegator AND delegatee:
```yaml
senior_orchestrator:
  can_delegate_to: [orchestrator, architect, task_executor]
```

### 2. Chatbot Integration
All agents support chat mode when no goal provided:
```python
# No goal → enter chat mode
agent = TaskExecutorAgent(workspace=".test", use_behaviors=True)
agent.run()  # Enters interactive chat until goal is clear
```

ChatbotBehavior provides `set_goal` tool to transition from chat → execution.

### 3. CLI and Tool Call Delegation
DelegationBehavior already supports both:
- **Tool call**: `orchestrator.delegate_to_executor(task_description="...")`
- **CLI**: `python task_executor_agent.py "goal description"`

## Conclusion

The unified delegation system provides a clean, config-driven architecture for bidirectional delegation. All agents can now participate in delegation without special-case logic, and the system is fully composable and backward compatible.

**Key Principle**: Delegation is a relationship, not a role. Any agent can be a delegator, delegatee, or both.
