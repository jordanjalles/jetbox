# CLI Flags for Dynamic Behavior Injection - Usage Guide

## Quick Start

Inject any behavior at runtime without modifying config files:

```bash
# Short name (auto-appends "Behavior")
python agent.py --ContextInspector "Create calculator"

# Full name
python agent.py --ContextInspectorBehavior "Create calculator"

# Multiple behaviors
python agent.py --StatusDisplay --ContextInspector "Complex task"
```

## Syntax

### Basic Format

```bash
python agent.py --BehaviorName [other flags] [goal]
```

**Rules**:
- Flag must start with `--` followed by capital letter
- CamelCase name matching behavior class
- Can omit "Behavior" suffix (auto-appended)

### Examples

| CLI Flag | Loads Behavior | Notes |
|----------|----------------|-------|
| `--ContextInspector` | `ContextInspectorBehavior` | Recommended (shorter) |
| `--ContextInspectorBehavior` | `ContextInspectorBehavior` | Also works (explicit) |
| `--StatusDisplay` | `StatusDisplayBehavior` | Multiple words, no space |
| `--LoopDetection` | `LoopDetectionBehavior` | Already in config → skipped |

## Use Cases

### 1. Context Inspection

Capture context snapshots for analysis:

```bash
python agent.py --ContextInspector "Build Flask app with tests"
```

After completion, analyze:

```bash
python tools/analyze_context.py .context_inspection
```

### 2. Status Display During Debugging

Show detailed progress during execution:

```bash
python agent.py --StatusDisplay "Complex multi-step task"
```

### 3. Testing New Behaviors

Test a new behavior without adding to config:

```bash
python agent.py --MyNewBehavior "test goal"
```

### 4. Session-Wide Inspection

Inspect all agents in a multi-agent workflow:

```bash
export JETBOX_EXTRA_BEHAVIORS="ContextInspectorBehavior"
python agent.py "Multi-agent project setup"
```

All spawned sub-agents will also capture contexts.

## Environment Variable

For persistent injection across multiple runs:

### Bash/Linux

```bash
export JETBOX_EXTRA_BEHAVIORS="ContextInspectorBehavior,StatusDisplayBehavior"
python agent.py "goal 1"
python agent.py "goal 2"
python agent.py "goal 3"
```

### PowerShell/Windows

```powershell
$env:JETBOX_EXTRA_BEHAVIORS = "ContextInspectorBehavior,StatusDisplayBehavior"
python agent.py "goal 1"
python agent.py "goal 2"
```

### Clear Environment Variable

```bash
unset JETBOX_EXTRA_BEHAVIORS  # Bash
$env:JETBOX_EXTRA_BEHAVIORS = ""  # PowerShell
```

## Combining with Other Flags

Behavior flags work with all standard flags:

```bash
# With team selection
python agent.py --team solo --ContextInspector "Create calculator"

# With workspace
python agent.py --workspace /tmp/test --ContextInspector "Build app"

# With timeout
python agent.py --timeout 1200 --ContextInspector "Long task"

# With chat mode
python agent.py --chat --StatusDisplay
```

**Order doesn't matter**:
```bash
python agent.py --ContextInspector --team solo "goal"
python agent.py --team solo --ContextInspector "goal"
```

## Behavior Validation

### Check if Behavior Loaded

Look for confirmation in output:

```bash
$ python agent.py --ContextInspector "test"
[agent.py] Extra behaviors enabled: ContextInspectorBehavior
...
[task_executor] Loading extra behaviors: ['ContextInspectorBehavior']
[task_executor] Loaded extra behavior: ContextInspectorBehavior
```

### Common Errors

**Module not found**:
```
[task_executor] Failed to load extra behavior MyBehavior: No module named 'behaviors.my_behavior'
```
→ Check behavior class name, ensure file exists in `behaviors/`

**Already loaded**:
```
[task_executor] Extra behavior ChatbotBehavior already loaded, skipping
```
→ Behavior is in agent config, no need to inject

**Excluded**:
```
[task_executor] Skipping excluded extra behavior: ChatbotBehavior
```
→ Behavior is in exclude list (e.g., autonomous mode excludes ChatbotBehavior)

## Available Behaviors

### Core Behaviors (Usually in Config)

- `ChatbotBehavior` - Interactive chat mode
- `CompactWhenNearFullBehavior` - Context compaction
- `FileToolsBehavior` - File operations (split into Read/Write/Directory)
- `CommandToolsBehavior` - Shell command execution
- `LoopDetectionBehavior` - Infinite loop detection
- `WorkspaceTaskNotesBehavior` - Persistent context notes

### Optional Behaviors (Good for CLI Injection)

- `ContextInspectorBehavior` - Capture context snapshots
- `StatusDisplayBehavior` - Show progress bars and stats
- `DelegationBehavior` - Enable agent delegation
- `ServerToolsBehavior` - Server management tools

### Custom Behaviors

Any behavior in `behaviors/` can be injected:

```bash
ls behaviors/*.py  # List available behaviors
python agent.py --YourBehavior "test"
```

## Tips and Best Practices

### 1. Use Short Names

✅ **Recommended**:
```bash
python agent.py --ContextInspector "goal"
```

❌ **Avoid**:
```bash
python agent.py --ContextInspectorBehavior "goal"
```

### 2. Session-Wide for Multi-Agent

When using orchestrator or multi-agent workflows:

```bash
export JETBOX_EXTRA_BEHAVIORS="ContextInspectorBehavior"
python agent.py "Complex project with multiple agents"
```

All agents (orchestrator, task_executor, architect) will load the behavior.

### 3. Combine with --once for Quick Tests

```bash
python agent.py --once --StatusDisplay "What's the weather?"
```

### 4. Don't Inject Behaviors Already in Config

Check agent config first:

```bash
cat config/agents/task_executor.yaml | grep behaviors -A 20
```

If behavior is listed, no need to inject.

### 5. Use Environment Variable for Analysis Sessions

When doing performance analysis or debugging:

```bash
# Start session
export JETBOX_EXTRA_BEHAVIORS="ContextInspectorBehavior,StatusDisplayBehavior"

# Run multiple tests
python agent.py "test 1"
python agent.py "test 2"
python agent.py "test 3"

# Analyze all results
python tools/analyze_context.py .context_inspection

# End session
unset JETBOX_EXTRA_BEHAVIORS
```

## Troubleshooting

### Behavior Not Loading

**Symptom**: No "Loaded extra behavior" message

**Checklist**:
1. Check spelling: Must match class name exactly
2. Check capitalization: Must start with capital letter
3. Check file exists: `ls behaviors/your_behavior.py`
4. Check class name: Open file, verify class name

### Duplicate Prevention Issues

**Symptom**: Behavior loads twice

**Cause**: Rare, likely bug in duplicate detection

**Debug**:
```python
# Check loaded behaviors
agent.behaviors  # List of behavior instances
[b.get_name() for b in agent.behaviors]  # List of names
```

### Environment Variable Not Persisting

**Symptom**: Variable lost after terminal close

**Solution**: Add to shell profile
```bash
# Add to ~/.bashrc or ~/.zshrc
export JETBOX_EXTRA_BEHAVIORS="ContextInspectorBehavior"
```

## Advanced Usage

### Inject Behavior in Python Code

```python
from base_agent import BaseAgent

agent = BaseAgent(
    name="my_agent",
    workspace=Path("/tmp/test"),
    config_file="config/agents/task_executor.yaml",
    extra_behaviors=["ContextInspectorBehavior", "StatusDisplayBehavior"]
)
```

### Conditional Injection

```bash
# Only inject if debugging
if [ "$DEBUG" = "1" ]; then
    export JETBOX_EXTRA_BEHAVIORS="ContextInspectorBehavior,StatusDisplayBehavior"
fi

python agent.py "goal"
```

### Scripted Analysis Runs

```bash
#!/bin/bash
# analysis_run.sh

# Set up inspection
export JETBOX_EXTRA_BEHAVIORS="ContextInspectorBehavior"

# Run test scenarios
python agent.py "scenario 1"
python agent.py "scenario 2"
python agent.py "scenario 3"

# Analyze results
python tools/analyze_context.py .context_inspection > analysis_report.md

# Clean up
unset JETBOX_EXTRA_BEHAVIORS
```

## Related Documentation

- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Full technical specification
- [Phase 2 Summary](PHASE2_SUMMARY.md) - Implementation details
- [Behaviors Documentation](../../BEHAVIORS_DOCUMENTATION.md) - Available behaviors
- [Context Inspector](../../behaviors/context_inspector.py) - Context inspection behavior

## Support

For issues or questions:
1. Check error messages (usually clear about what went wrong)
2. Verify behavior exists: `ls behaviors/`
3. Check agent config: `cat config/agents/task_executor.yaml`
4. Review logs: Look for "Loading extra behaviors" messages
