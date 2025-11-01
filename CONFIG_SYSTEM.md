# Configuration System Documentation

This document describes Jetbox's configuration system for agent behavior, task management, and behavior parameters.

## Overview

Jetbox uses a multi-layered YAML configuration system:

1. **Global Configuration** (`agent_config.yaml`) - Shared settings for all agents
2. **Agent-Specific Configurations** (`{agent}_config.yaml`) - Per-agent behavior and prompt customization
3. **Agent Relationships** (`agents.yaml`) - Delegation relationships and agent registry

## Global Configuration (`agent_config.yaml`)

This file contains global settings that apply to all agents unless overridden.

### Global Behavior Parameter Defaults

Behavior parameters (max_tokens, max_repeats, whitelists, etc.) are defined globally under the `behavior_defaults` section.

```yaml
# agent_config.yaml
behavior_defaults:
  CompactWhenNearFullBehavior:
    max_tokens: 8000          # Model-based: 8K for qwen2.5-coder:3b/7b
    compact_threshold: 0.75   # Compact at 75% capacity
    keep_recent: 20           # Keep last 20 message exchanges

  LoopDetectionBehavior:
    max_repeats: 5            # Warn after 5 identical actions

  CommandToolsBehavior:
    whitelist:                # Safe commands for coding tasks
      - python
      - pytest
      - ruff
      - pip
```

**How It Works**:

1. **Global Defaults**: `agent_config.yaml` defines default parameters for all behaviors
2. **Agent Overrides**: Individual agent configs can override specific parameters
3. **Merge Strategy**: Agent-specific params override global defaults (shallow merge)

**Benefits**:

- **DRY**: Define parameters once, use everywhere
- **Consistency**: All agents use same defaults unless explicitly overridden
- **Easy Updates**: Change one value to update all agents
- **Model-Agnostic**: Adjust `max_tokens` globally when switching models

**Example Usage**:

```yaml
# agent_config.yaml - Global defaults
behavior_defaults:
  LoopDetectionBehavior:
    max_repeats: 5

# task_executor_config.yaml - Uses global default
behaviors:
  - type: LoopDetectionBehavior
    # No params - uses global default (max_repeats=5)

# orchestrator_config.yaml - Overrides global default
behaviors:
  - type: LoopDetectionBehavior
    params:
      max_repeats: 10  # Override: orchestrator gets more repeats
```

**Current State**:

All agents currently use global defaults - no overrides are configured. To change a behavior parameter globally, edit `agent_config.yaml`. To override for a specific agent, add `params:` to that agent's config.

### LLM Settings

```yaml
llm:
  model: "qwen3:14b"        # Default model (override with OLLAMA_MODEL env var)
  temperature: 0.2          # Inference temperature (0.0 = deterministic)
  system_prompt: |          # Base system prompt (behaviors can extend)
    You are a local coding agent...
```

**Environment Variable Override**:
```bash
# PowerShell
$env:OLLAMA_MODEL = "gpt-oss:20b"

# Bash
export OLLAMA_MODEL="gpt-oss:20b"
```

### Round Limits

Controls how many LLM calls are allowed before escalation:

```yaml
rounds:
  max_per_subtask: 12       # Rounds per subtask before forced escalation
  max_global: 256           # Safety cap for entire agent run
```

**Example**: If a subtask takes 12 rounds without completion, the agent will either decompose into smaller subtasks or zoom out to a higher level.

### Time Limits

```yaml
timeouts:
  max_goal_time: 600                    # 10 minutes max (0 = no limit)
  create_summary_on_timeout: true       # Create jetbox notes on timeout
  save_context_dump: true               # Save context to .agent_context/timeout_dumps/
```

**Use Cases**:
- Prevent runaway agent processes
- Force graceful shutdown with context preservation
- Useful for CI/CD integration

### Hierarchy Limits

```yaml
hierarchy:
  max_depth: 5              # Maximum task nesting levels
  max_siblings: 8           # Max subtasks per level
```

**Example Hierarchy**:
```
Goal (depth 0)
├─ Task 1 (depth 1)
│  ├─ Subtask 1.1 (depth 2)
│  │  ├─ Action 1.1.1 (depth 3)
│  │  └─ Action 1.1.2 (depth 3)
│  └─ Subtask 1.2 (depth 2)
└─ Task 2 (depth 1)
   └─ Subtask 2.1 (depth 2)
```

At `max_depth=5`, no subtask can be nested more than 5 levels deep. At `max_siblings=8`, no level can have more than 8 sibling subtasks.

### Escalation Strategy

Controls what happens when a subtask fails repeatedly:

```yaml
escalation:
  strategy: "force_decompose"           # Options: "force_decompose", "agent_decides"
  zoom_out_target: "smart"              # Options: "parent", "task", "root", "smart"
  max_approach_retries: 3               # Retry attempts at root before final failure
  block_failed_paths: true              # Mark failed approaches as blocked
```

**Strategies**:

- `force_decompose`: Always decompose into smaller subtasks (no give-up option)
- `agent_decides`: Let agent choose between decompose/zoom_out/give_up

**Zoom Out Targets**:

- `parent`: Go up one level to parent subtask
- `task`: Go to task level (top of current task)
- `root`: Go to root goal and reconsider entire approach
- `smart`: Analyze subtask tree to find actual root of problem

**Example Flow** (`force_decompose` + `zoom_out_target: smart`):

1. Subtask fails after 12 rounds
2. Agent decomposes into 3 smaller subtasks
3. If at `max_depth`, zooms out to smart target (e.g., root if fundamental approach issue)
4. At root, agent gets 3 approach retries before final failure

### Loop Detection

```yaml
loop_detection:
  max_action_repeats: 3     # Same action attempted N times = loop
  max_subtask_repeats: 2    # Same subtask failed N times = escalate
  max_context_age: 300      # Context age in seconds before considered stale
```

**Note**: This is **separate** from `LoopDetectionBehavior.max_repeats`. These settings control escalation logic, while behavior settings control warning triggers.

### Decomposition Behavior

```yaml
decomposition:
  min_children: 2           # Minimum child subtasks when decomposing
  max_children: 6           # Maximum child subtasks
  temperature: 0.2          # LLM temperature for decomposition (higher = more creative)
  prefer_granular: true     # Prefer more, smaller subtasks vs fewer, larger ones
```

### Approach Reconsideration

```yaml
approach_retry:
  enabled: true                         # Track approach attempts per task
  reset_subtasks_on_retry: true         # Reset all subtasks when retrying
  preserve_completed: true              # Keep successful subtasks
  retry_style: "learn_from_failures"    # Options: "fresh_start", "learn_from_failures"
```

**Retry Styles**:

- `fresh_start`: Start over with blank slate
- `learn_from_failures`: Review what failed and try different approach

### Context Management

```yaml
context:
  history_keep: 12                      # Message exchanges to keep (should match max_per_subtask)
  max_tokens: 8000                      # Maximum tokens in context (0 = no limit)
  recent_actions_limit: 10              # Recent actions to include in context
  enable_compression: false             # Enable context compression (not yet implemented)
  compression_threshold: 20             # Compress when context exceeds this many messages
```

**Note**: These are **legacy settings** for non-behavior mode. In behavior mode, context is managed by `CompactWhenNearFullBehavior` using its own parameters from `behavior_defaults`.

## Agent-Specific Configuration

Each agent can have its own config file: `{agent}_config.yaml`

### Structure

```yaml
# Agent blurb (for parent agents)
blurb: |
  TaskExecutor handles focused implementation work...

# Delegation tool configuration (for agents that can be delegated to)
delegation_tool:
  name: "delegate_to_executor"
  description: "Delegate a coding task to the TaskExecutor agent"
  parameters:
    task_description:
      type: string
      description: "Clear description of the task"
      required: true

# System prompt (overrides global if present)
system_prompt: |
  You are a specialized agent that...

# Behaviors (composable capabilities)
behaviors:
  - type: CompactWhenNearFullBehavior
    # No params - uses global defaults

  - type: LoopDetectionBehavior
    params:
      max_repeats: 10  # Override global default
```

### Examples

**task_executor_config.yaml**:
```yaml
behaviors:
  - type: SubAgentContextBehavior
  - type: CompactWhenNearFullBehavior
  - type: FileToolsBehavior
  - type: CommandToolsBehavior
  - type: ServerToolsBehavior
  - type: LoopDetectionBehavior
  - type: WorkspaceTaskNotesBehavior
```

**orchestrator_config.yaml**:
```yaml
behaviors:
  - type: CompactWhenNearFullBehavior
  - type: LoopDetectionBehavior
  # DelegationBehavior auto-added from agents.yaml
```

**architect_config.yaml**:
```yaml
behaviors:
  - type: CompactWhenNearFullBehavior
  - type: ArchitectToolsBehavior
  - type: LoopDetectionBehavior
```

## Agent Relationships (`agents.yaml`)

Defines the agent registry and delegation relationships:

```yaml
agents:
  orchestrator:
    type: OrchestratorAgent
    can_delegate_to:
      - architect
      - task_executor

  architect:
    type: ArchitectAgent

  task_executor:
    type: TaskExecutorAgent
```

**How Delegation Works**:

1. `agents.yaml` defines `can_delegate_to` list
2. `BaseAgent._auto_add_delegation_behavior()` reads this list
3. For each target agent, reads its `{agent}_config.yaml` for `delegation_tool` schema
4. Creates `DelegationBehavior` with appropriate tools
5. Orchestrator gets tools: `consult_architect`, `delegate_to_executor`

**Example Delegation Tool** (from `task_executor_config.yaml`):

```yaml
delegation_tool:
  name: "delegate_to_executor"
  description: "Delegate a coding task to the TaskExecutor agent"
  parameters:
    task_description:
      type: string
      description: "Clear description of the task to execute"
      required: true
    workspace_mode:
      type: string
      description: "Workspace mode: 'new' for new projects, 'existing' for updates"
      enum: ["new", "existing"]
      required: true
```

## Behavior System

Behaviors are composable modules that provide:
- **Context enhancements** (inject prompts, summaries, etc.)
- **Tools** (file operations, commands, delegation, etc.)
- **Event handlers** (on_tool_call, on_goal_start, etc.)
- **Instructions** (workflow guidance for LLM)

See [BEHAVIORS_DOCUMENTATION.md](BEHAVIORS_DOCUMENTATION.md) for complete behavior reference.

### Available Behaviors

**Context Management**:
- `SubAgentContextBehavior` - For delegated work
- `CompactWhenNearFullBehavior` - Append until full, then compact
- `ArchitectContextBehavior` - For architecture discussions
- `HierarchicalContextBehavior` - Goal/Task/Subtask hierarchy

**Tools**:
- `FileToolsBehavior` - read_file, write_file, list_dir
- `CommandToolsBehavior` - run_bash
- `ServerToolsBehavior` - start_server, stop_server, check_server
- `ArchitectToolsBehavior` - Architecture artifact creation
- `DelegationBehavior` - Auto-configured from agents.yaml

**Utilities**:
- `LoopDetectionBehavior` - Detect infinite loops
- `WorkspaceTaskNotesBehavior` - Persistent context summaries

### Configuring Behaviors

**Global Defaults** (recommended):
```yaml
# agent_config.yaml
behavior_defaults:
  LoopDetectionBehavior:
    max_repeats: 5
```

**Agent-Specific Override** (only when needed):
```yaml
# orchestrator_config.yaml
behaviors:
  - type: LoopDetectionBehavior
    params:
      max_repeats: 10  # Override for this agent only
```

**No Parameters** (uses global defaults):
```yaml
behaviors:
  - type: LoopDetectionBehavior
    # No params field - uses global default (5)
```

## Configuration Best Practices

1. **Use Global Defaults**: Define behavior parameters in `agent_config.yaml`
2. **Override Sparingly**: Only override when agent truly needs different behavior
3. **Document Overrides**: Add comments explaining why override is needed
4. **Model-Based Settings**: Adjust `max_tokens` globally when switching models
5. **Test After Changes**: Run existing tests to verify configuration changes
6. **Version Control**: Commit all config files together

## Testing Configuration

**Verify Global Defaults**:
```bash
python test_global_behavior_defaults.py
```

**Check Agent Startup**:
```bash
python agent.py "test task"
# Look for: "[agent_name] Loaded behavior: X (using global defaults)"
```

## Migration from Legacy Mode

**Old (Deprecated)**:
```python
agent = TaskExecutorAgent(
    workspace=".",
    goal="test",
    use_behaviors=False  # Legacy mode
)
```

**New (Recommended)**:
```python
agent = TaskExecutorAgent(
    workspace=".",
    goal="test",
    use_behaviors=True  # Behavior system
)
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for complete migration instructions.

## Troubleshooting

**Behavior not getting expected parameters**:
- Check `agent_config.yaml` has `behavior_defaults` section
- Verify behavior type name matches exactly (case-sensitive)
- Look for `[agent_name] Loaded behavior: X (using global defaults)` in output
- Check for typos in YAML (use YAML linter)

**Agent-specific override not working**:
- Verify `params:` field is present in agent config
- Check merge order: agent params should override global
- Look for `[agent_name] Loaded behavior: X (agent-specific params: {...})` in output

**DelegationBehavior not auto-added**:
- Check `agents.yaml` has `can_delegate_to` list for this agent
- Verify target agent has `delegation_tool` in its config file
- Look for `[agent_name] Auto-added DelegationBehavior` in output

**Configuration file not found**:
- Ensure config file is in repository root directory
- Check filename matches pattern: `{agent_name}_config.yaml`
- Verify file has `.yaml` extension (not `.yml`)

## Reference

**Config Files**:
- `/workspace/agent_config.yaml` - Global configuration
- `/workspace/task_executor_config.yaml` - TaskExecutor agent config
- `/workspace/orchestrator_config.yaml` - Orchestrator agent config
- `/workspace/architect_config.yaml` - Architect agent config
- `/workspace/agents.yaml` - Agent relationships and registry

**Code**:
- `/workspace/base_agent.py:330-444` - `load_behaviors_from_config()`
- `/workspace/base_agent.py:419-444` - `_load_global_behavior_defaults()`
- `/workspace/behaviors/` - Behavior implementations

**Tests**:
- `/workspace/test_global_behavior_defaults.py` - Global defaults validation

**Documentation**:
- [BEHAVIORS_DOCUMENTATION.md](BEHAVIORS_DOCUMENTATION.md) - Behavior reference
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Legacy to behavior system migration
- [CLAUDE.md](CLAUDE.md) - Main project documentation
