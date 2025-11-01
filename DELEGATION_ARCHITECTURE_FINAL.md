# Delegation Architecture (Final Design)

**Date**: 2025-11-01
**Status**: ✅ IMPLEMENTED

## Overview

The Jetbox delegation system uses **TWO separate behaviors** with clear, distinct responsibilities:

1. **SubAgentModeBehavior** - Universal (ALL agents)
2. **DelegationBehavior** - Special (delegator agents only)

This separation prevents LLM confusion and provides a clear mental model.

## Design Principle

**Key Insight**: Delegation has TWO orthogonal capabilities:
- **Being delegatable** (universal) - ALL agents can receive work
- **Being a delegator** (special) - SOME agents can delegate TO others

These are SEPARATE concerns and should NOT be combined in one behavior.

## Architecture

### 1. SubAgentModeBehavior (Universal)

**Purpose**: Makes agents delegatable - they can receive work via CLI or tool calls.

**Added to**: ALL agents (TaskExecutor, Architect, Orchestrator)

**Provides**:
- `mark_complete(summary)` - Signal successful completion
- `mark_failed(reason)` - Signal failure

**Context injection**:
- Adds "DELEGATED GOAL:" or "GOAL:" header
- Provides completion instructions

**When used**:
- CLI: `python task_executor_agent.py "Create calculator"`
- Tool call: `orchestrator.delegate_to_executor(task_description="...")`

**File**: `behaviors/subagent_mode.py`

### 2. DelegationBehavior (Special)

**Purpose**: Enables delegating work TO other agents.

**Added to**: Only agents with `can_delegate_to` in agents.yaml (currently: Orchestrator)

**Provides**:
- `consult_architect(project_description, requirements, constraints)`
- `delegate_to_executor(task_description, workspace_mode, workspace_path)`

**Context injection**:
- Lists available agents for delegation
- Provides delegation guidelines from agent blurbs

**When used**:
- Orchestrator needs to delegate complex work
- Multi-agent coordination required

**File**: `behaviors/delegation.py`

## Current Agent Modes

| Agent | SubAgentMode? | Delegation? | Tools |
|-------|--------------|-------------|-------|
| **TaskExecutor** | ✅ Universal | ❌ No | mark_complete, mark_failed |
| **Architect** | ✅ Universal | ❌ No | mark_complete, mark_failed |
| **Orchestrator** | ✅ Universal | ✅ Special | mark_complete, mark_failed, consult_architect, delegate_to_executor |

**All agents can receive work** (universal delegatable capability).
**Only Orchestrator can delegate** (special delegator capability).

## Why Split (Not Unified)?

### Original Attempt: Unified DelegationBehavior
- Combined both directions in one behavior with `is_delegatee` flag
- Caused confusion: Agents might think they can delegate when they can't
- Mixed universal and special capabilities

### Current Design: Separate Behaviors
- **SubAgentModeBehavior**: Universal - makes sense for ALL agents
- **DelegationBehavior**: Special - only for delegators
- Clear separation prevents LLM confusion
- Simpler mental model for both humans and LLMs

## Auto-Configuration

### Base Agent Auto-Add Logic

**SubAgentModeBehavior** (in `_auto_add_subagent_context_behavior()`):
```python
# Added to ALL agents - no conditions
self.add_behavior(SubAgentModeBehavior(is_subagent=True))
```

**DelegationBehavior** (in `_auto_add_delegation_behavior()`):
```python
# Added only if agent has can_delegate_to in agents.yaml
if can_delegate_to:
    self.add_behavior(DelegationBehavior(agent_relationships))
```

## agents.yaml Configuration

```yaml
agents:
  orchestrator:
    class: OrchestratorAgent
    can_delegate_to:
      - architect
      - task_executor

  architect:
    class: ArchitectAgent
    can_delegate_to: []  # No delegation capability

  task_executor:
    class: TaskExecutorAgent
    can_delegate_to: []  # No delegation capability
```

## Completion Tools

The system provides **TWO types of completion tools** for different use cases:

### 1. mark_complete / mark_failed (SubAgentModeBehavior)
**Purpose**: Signal completion when invoked by another agent
**Provided by**: SubAgentModeBehavior (ALL agents have this)
**Returns**: `{"success": True/False, "summary": "...", "reason": "..."}`
**Use case**: Delegated work

### 2. mark_goal_complete (CompactWhenNearFullBehavior)
**Purpose**: Signal completion for standalone goals
**Provided by**: CompactWhenNearFullBehavior
**Returns**: `{"status": "goal_complete", "message": "...", "summary": "..."}`
**Use case**: Standalone execution (not delegated)

**Why both?**:
- Different invocation contexts (delegated vs standalone)
- Different return semantics (success/failure vs goal status)
- Flexible architecture supporting multiple usage patterns

## Benefits

### 1. Clarity
- Universal vs special capabilities are explicit
- No confusion about what an agent can do
- Clear mental model for LLM and humans

### 2. Simplicity
- Each behavior has ONE clear purpose
- No conditional logic within behaviors
- Easy to understand and maintain

### 3. Composability
- Mix and match behaviors as needed
- No coupling between universal and special capabilities
- Easy to add new delegator agents (just add can_delegate_to)

### 4. Prevents Confusion
- TaskExecutor won't think it can delegate
- Architect won't think it can delegate
- Only Orchestrator knows it can delegate

## Testing

### Verification Test
```bash
python -c "
from task_executor_agent import TaskExecutorAgent
from orchestrator_agent import OrchestratorAgent

# TaskExecutor: SubAgentMode only
te = TaskExecutorAgent(workspace='.test', goal='test', use_behaviors=True)
assert any(b.get_name() in ['subagent_mode', 'subagent_context'] for b in te.behaviors)
assert not any(b.get_name() == 'delegation' for b in te.behaviors)

# Orchestrator: BOTH behaviors
orch = OrchestratorAgent(workspace='.test_orch', use_behaviors=True)
assert any(b.get_name() in ['subagent_mode', 'subagent_context'] for b in orch.behaviors)
assert any(b.get_name() == 'delegation' for b in orch.behaviors)

print('✓ All assertions passed!')
"
```

## Future Enhancements

### Multi-Level Orchestration
Orchestrator could delegate to another Orchestrator:

```yaml
# agents.yaml
senior_orchestrator:
  can_delegate_to: [orchestrator, architect, task_executor]

orchestrator:
  can_delegate_to: [architect, task_executor]
```

Both orchestrators would have:
- SubAgentModeBehavior (can be delegated to)
- DelegationBehavior (can delegate)

### Chatbot Integration
All agents already have ChatbotBehavior for interactive mode when no goal is provided.

## Migration from Previous Attempts

### Attempt 1: Unified DelegationBehavior ❌
- Combined delegator + delegatee in one behavior
- Used `is_delegatee` flag to toggle mode
- **Problem**: Confusion about capabilities, complex behavior

### Current: Separate Behaviors ✅
- SubAgentModeBehavior: Universal (ALL agents)
- DelegationBehavior: Special (delegator agents)
- **Benefit**: Clear separation, no confusion

## Conclusion

The split delegation architecture provides a clean, understandable model:

- **ALL agents are delegatable** (via SubAgentModeBehavior)
- **SOME agents can delegate** (via DelegationBehavior)

This separation is intuitive, prevents confusion, and supports future expansion (multi-level orchestration, new delegator types, etc.).

**Key Principle**: Delegation is not a single capability - it's TWO separate capabilities that should be modeled separately.
