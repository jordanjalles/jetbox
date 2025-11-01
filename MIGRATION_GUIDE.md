# Migration Guide: Context Strategies → Agent Behaviors

**Version**: 1.0
**Last Updated**: 2025-01-01
**Target Version**: 2.0 (removal of deprecated code)

## Overview

Jetbox has refactored from a dual architecture (context strategies + enhancements) to a **unified composable behavior system**. This guide helps you migrate your code and configurations.

## Why Migrate?

### Benefits of the New System

1. **Simpler Architecture**: One unified system instead of two (strategies + enhancements)
2. **More Flexible**: Mix and match any behaviors without constraints
3. **Better Tested**: Each behavior has comprehensive unit tests
4. **Easier to Extend**: Create custom behaviors with clear interfaces
5. **Config-Driven**: Change agent behavior without code changes
6. **Better Separation**: Context, tools, and utilities are separate concerns

### What's Changing?

**Old System (Deprecated)**:
- Context Strategies: `HierarchicalStrategy`, `AppendUntilFullStrategy`, `SubAgentStrategy`, `ArchitectStrategy`
- Enhancements: `JetboxNotesEnhancement`, `TaskManagementEnhancement`
- Hardcoded in agent `__init__` methods
- Tools scattered across multiple files

**New System**:
- Unified `AgentBehavior` base class
- All capabilities as composable behaviors
- Config-driven loading from YAML files
- No hardcoded dependencies

## Migration Timeline

- **Now**: Old system deprecated with warnings
- **Version 1.x**: Both systems supported (backward compatible)
- **Version 2.0** (June 2025): Old system removed

**Recommendation**: Migrate when convenient. No urgent action required.

---

## How to Migrate

### Step 1: Enable Behavior Mode (Optional)

The new behavior system can coexist with the old system. To enable behaviors explicitly:

```python
# OLD (still works - uses deprecated strategies)
agent = TaskExecutorAgent(workspace=".", goal="Create calculator")

# NEW (recommended - uses behavior system)
agent = TaskExecutorAgent(
    workspace=".",
    goal="Create calculator",
    use_behaviors=True  # Enable new behavior system
)
```

**Note**: In future versions, `use_behaviors=True` will become the default.

### Step 2: Understand the Behavior Mapping

| Old Strategy/Enhancement | New Behavior | Config File |
|--------------------------|--------------|-------------|
| `HierarchicalStrategy` | `HierarchicalContextBehavior` | `task_executor_config.yaml` |
| `AppendUntilFullStrategy` | `CompactWhenNearFullBehavior` | `orchestrator_config.yaml` |
| `SubAgentStrategy` | `SubAgentContextBehavior` | `task_executor_config.yaml` |
| `ArchitectStrategy` | `ArchitectContextBehavior` | `architect_config.yaml` |
| `JetboxNotesEnhancement` | `JetboxNotesBehavior` | (included in configs) |
| `TaskManagementEnhancement` | `TaskManagementBehavior` | `orchestrator_config.yaml` |

### Step 3: Customize Config (If Needed)

If you were using custom strategy parameters, create a custom config file:

```yaml
# my_custom_config.yaml
behaviors:
  - type: SubAgentContextBehavior
    params:
      max_tokens: 128000
      recent_keep: 20
  - type: FileToolsBehavior
    params: {}
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "pip"]
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5
```

Use it:

```python
agent = TaskExecutorAgent(
    workspace=".",
    goal="Create calculator",
    use_behaviors=True,
    config_file="my_custom_config.yaml"
)
```

### Step 4: Update Tests (If Applicable)

No changes needed! The old system still works in legacy mode:

```python
# This still works (backward compatible)
agent = TaskExecutorAgent(workspace=".", goal="Test goal")
agent.run()
```

---

## Configuration Examples

### TaskExecutor Agent

**Before (Old System)**:
```python
from context_strategies import SubAgentStrategy

class TaskExecutorAgent:
    def __init__(self, workspace, goal):
        self.strategy = SubAgentStrategy(max_tokens=128000)
        # ... hardcoded tool setup
```

**After (New System)**:
```yaml
# task_executor_config.yaml
behaviors:
  # Context management
  - type: SubAgentContextBehavior
    params:
      max_tokens: 128000
      recent_keep: 20

  # Tool behaviors
  - type: FileToolsBehavior
    params: {}
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "pip"]
  - type: ServerToolsBehavior
    params: {}

  # Utility behaviors
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5
```

```python
# Agent code simplified
class TaskExecutorAgent(BaseAgent):
    def __init__(self, workspace, goal, use_behaviors=True, **kwargs):
        super().__init__(
            name="task_executor",
            workspace=workspace,
            config_file="task_executor_config.yaml" if use_behaviors else None,
            **kwargs
        )
        if goal:
            self.set_goal(goal)
```

### Orchestrator Agent

**Before (Old System)**:
```python
from context_strategies import AppendUntilFullStrategy, TaskManagementEnhancement

class OrchestratorAgent:
    def __init__(self, workspace):
        self.strategy = AppendUntilFullStrategy(max_tokens=131072)
        self.enhancements = [TaskManagementEnhancement(workspace)]
        # ... hardcoded delegation tools
```

**After (New System)**:
```yaml
# orchestrator_config.yaml
behaviors:
  # Context management
  - type: HierarchicalContextBehavior
    params:
      history_keep: 12

  # Delegation tools
  - type: DelegationBehavior
    params: {}

  # Utility behaviors
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5
```

### Architect Agent

**Before (Old System)**:
```python
from context_strategies import ArchitectStrategy

class ArchitectAgent:
    def __init__(self, workspace):
        self.strategy = ArchitectStrategy(max_tokens=32000)
        # ... hardcoded architecture tools
```

**After (New System)**:
```yaml
# architect_config.yaml
behaviors:
  # Context management
  - type: ArchitectContextBehavior
    params:
      max_tokens: 32000
      recent_keep: 20

  # Architecture tools
  - type: ArchitectToolsBehavior
    params: {}

  # Utility behaviors
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5
```

---

## Code Migration Examples

### Example 1: Simple TaskExecutor

**Before**:
```python
from context_strategies import SubAgentStrategy
from task_executor_agent import TaskExecutorAgent

# Create agent with custom strategy
agent = TaskExecutorAgent(
    workspace="./my_workspace",
    goal="Create a calculator package"
)
agent.strategy = SubAgentStrategy(max_tokens=64000)  # Override default
agent.run()
```

**After**:
```python
from task_executor_agent import TaskExecutorAgent

# Create custom config
# my_config.yaml:
# behaviors:
#   - type: SubAgentContextBehavior
#     params:
#       max_tokens: 64000
#   - type: FileToolsBehavior
#     params: {}
#   - type: CommandToolsBehavior
#     params:
#       whitelist: ["python", "pytest", "ruff"]

agent = TaskExecutorAgent(
    workspace="./my_workspace",
    goal="Create a calculator package",
    use_behaviors=True,
    config_file="my_config.yaml"
)
agent.run()
```

### Example 2: Orchestrator with Task Management

**Before**:
```python
from context_strategies import AppendUntilFullStrategy, TaskManagementEnhancement
from orchestrator_agent import OrchestratorAgent

agent = OrchestratorAgent(workspace="./workspace")
agent.strategy = AppendUntilFullStrategy(max_tokens=100000)
agent.enhancements.append(TaskManagementEnhancement(agent.workspace_manager))
agent.run()
```

**After**:
```python
from orchestrator_agent import OrchestratorAgent

# orchestrator_config.yaml already includes TaskManagementBehavior
agent = OrchestratorAgent(
    workspace="./workspace",
    use_behaviors=True  # Uses orchestrator_config.yaml by default
)
agent.run()
```

### Example 3: Custom Behavior

Want to add custom functionality? Create a behavior!

```python
# my_custom_behavior.py
from behaviors import AgentBehavior

class LoggingBehavior(AgentBehavior):
    """Logs all tool calls to a file."""

    def get_name(self) -> str:
        return "logging"

    def on_tool_call(self, tool_name, args, result, **kwargs):
        """Log each tool call."""
        with open("tool_calls.log", "a") as f:
            f.write(f"{tool_name}: {args} -> {result}\n")
```

**Config**:
```yaml
# my_config.yaml
behaviors:
  - type: SubAgentContextBehavior
    params:
      max_tokens: 128000
  - type: LoggingBehavior  # Your custom behavior
    params: {}
  - type: FileToolsBehavior
    params: {}
```

---

## Troubleshooting

### Error: "Behavior not found"

**Symptom**: `ImportError` or `AttributeError` when loading config.

**Cause**: Behavior name misspelled or behavior file missing.

**Solution**:
1. Check behavior name spelling (case-sensitive)
2. Ensure behavior exists in `behaviors/` directory
3. Check PYTHONPATH includes workspace

```python
# Debug: Check available behaviors
import behaviors
print(behaviors.__all__)
```

### Error: "Tool conflict"

**Symptom**: `ValueError: Tool 'write_file' already registered`

**Cause**: Two behaviors provide the same tool name.

**Solution**: Remove duplicate behavior from config.

```yaml
# BAD - FileToolsBehavior provides write_file twice
behaviors:
  - type: FileToolsBehavior
    params: {}
  - type: FileToolsBehavior  # Duplicate!
    params: {}

# GOOD
behaviors:
  - type: FileToolsBehavior
    params: {}
```

### Warning: DeprecationWarning

**Symptom**: `DeprecationWarning: context_strategies module is deprecated`

**Cause**: Code is using old strategies (still works, but deprecated).

**Solution**: Migrate to behaviors when convenient. To suppress warning:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

### Performance Slower

**Symptom**: Agent runs slower with behavior system.

**Diagnosis**:
1. Check behavior count (< 15 recommended)
2. Profile with `use_behaviors=False` for comparison

**Solution**:
- Disable unused behaviors in config
- Combine related behaviors if possible

---

## FAQ

### Q: Can I use both systems at once?

**A**: Yes! Set `use_behaviors=False` for legacy mode, `use_behaviors=True` for new system. Default is legacy mode for backward compatibility.

### Q: When will the old system be removed?

**A**: Version 2.0 (not before June 2025). You have plenty of time to migrate.

### Q: Do I need to migrate now?

**A**: No, legacy mode is fully supported. Migrate when you're ready or when you need new behavior features.

### Q: How do I create a custom behavior?

**A**: See `docs/behaviors/README.md` for complete documentation. Short version:

```python
from behaviors import AgentBehavior

class MyBehavior(AgentBehavior):
    def get_name(self) -> str:
        return "my_behavior"

    def get_tools(self) -> list[dict]:
        return [...]  # Tool definitions

    def dispatch_tool(self, tool_name, args, **kwargs):
        # Handle tool calls
        return {"result": "..."}
```

### Q: Will my tests break?

**A**: No, existing tests continue to work in legacy mode. Tests using deprecated classes will show warnings, but functionality is preserved.

### Q: Can I mix legacy strategies with new behaviors?

**A**: Not recommended. Choose one system per agent. Both work independently, but mixing causes confusion.

### Q: What if I find a bug in the new system?

**A**: Report it! Meanwhile, use `use_behaviors=False` to fall back to the old system.

### Q: How do I know which behaviors to use?

**A**: Use the default config files as templates:
- `task_executor_config.yaml` - For delegated work (SubAgent)
- `orchestrator_config.yaml` - For orchestration and delegation
- `architect_config.yaml` - For architecture design

---

## Getting Help

### Documentation

- **Behavior System**: `docs/behaviors/README.md`
- **Architecture**: `AGENT_ARCHITECTURE.md`
- **Configuration**: `CONFIG_SYSTEM.md`
- **Main Guide**: `CLAUDE.md`

### Examples

- **Config Files**: `task_executor_config.yaml`, `orchestrator_config.yaml`, `architect_config.yaml`
- **Behavior Implementations**: `behaviors/` directory
- **Tests**: `tests/test_*_behavior.py`

### Support

- **Issues**: File a GitHub issue with `[migration]` tag
- **Questions**: See FAQ above or check documentation
- **Bugs**: Report with `[bug]` tag and steps to reproduce

---

## Summary

**Key Points**:
1. Old system deprecated but still works
2. New system is config-driven and composable
3. Migration is optional until version 2.0
4. Both systems can coexist during transition
5. Benefits: simpler, more flexible, better tested

**Next Steps**:
1. Read this guide
2. Try new system with `use_behaviors=True`
3. Customize configs if needed
4. Migrate when convenient
5. Report any issues

**Remember**: No rush! The old system works fine and will be supported throughout version 1.x.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-01
**Applies To**: Jetbox 1.x → 2.0
