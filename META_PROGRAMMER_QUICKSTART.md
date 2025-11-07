# MetaProgrammer Quick Start Guide

**Status**: ✅ OPERATIONAL (Phase 5 Complete)

The MetaProgrammerAgent allows you to create new behaviors and agents interactively, extending Jetbox's capabilities through natural language conversation.

## Running MetaProgrammer

```python
from task_executor_agent import TaskExecutorAgent
from pathlib import Path

# Create workspace
workspace = Path('my_workspace')
workspace.mkdir(exist_ok=True)

# Load MetaProgrammer
meta = TaskExecutorAgent(
    workspace=workspace,
    config_file='config/agents/meta_programmer.yaml',
    timeout_seconds=300
)

# Now interact with it!
# The agent has ChatbotBehavior enabled, so you can have a conversation
```

## What Can MetaProgrammer Do?

### 1. Create New Behaviors

```python
# Use the create_behavior tool
result = meta.dispatch_tool(
    tool_name="create_behavior",
    args={
        "behavior_name": "MyCustomBehavior",
        "description": "What it does",
        "tool_specs": [...tool specifications...],
        "safety_mode": "review"  # review, auto, dryrun, strict
    }
)
```

### 2. Create New Agents

```python
# Use the create_agent tool
result = meta.dispatch_tool(
    tool_name="create_agent",
    args={
        "agent_name": "MyCustomAgent",
        "role": "Role description",
        "blurb": "What it does",
        "behaviors": ["ChatbotBehavior", "..."],
        "system_prompt": "Instructions...",
        "delegation_tool": {...tool spec...},
        "safety_mode": "review"
    }
)
```

## Example: Creating a Calculator Behavior

```python
from task_executor_agent import TaskExecutorAgent
from pathlib import Path

# Setup
workspace = Path('.meta_workspace')
workspace.mkdir(exist_ok=True)

meta = TaskExecutorAgent(
    workspace=workspace,
    config_file='config/agents/meta_programmer.yaml',
    timeout_seconds=300
)

# Create calculator behavior
result = meta.dispatch_tool(
    tool_name="create_behavior",
    args={
        "behavior_name": "CalculatorBehavior",
        "description": "Provides arithmetic calculation tools",
        "tool_specs": [
            {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "a": {"type": "number", "description": "First number", "required": True},
                    "b": {"type": "number", "description": "Second number", "required": True}
                }
            },
            {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "a": {"type": "number", "description": "First number", "required": True},
                    "b": {"type": "number", "description": "Second number", "required": True}
                }
            }
        ],
        "lifecycle_hooks": [],  # No hooks needed for simple tools
        "safety_mode": "auto"  # Auto-install after validation
    }
)

# Check results
if result.get("success"):
    print(f"✅ Behavior created: {result['behavior_file']}")
    print(f"✅ Tests created: {result['test_file']}")
    print(f"✅ Validation: {'PASS' if result['validation_results'].get('valid') else 'FAIL'}")
    print(f"✅ Installed: {result.get('installed', False)}")
else:
    print(f"❌ Error: {result.get('error')}")
```

## Safety Modes

MetaProgrammer supports 4 safety modes:

| Mode | Description | When to Use |
|------|-------------|-------------|
| `dryrun` | Generate files but don't install | Testing/experimentation |
| `review` | Require manual approval | Production (default) |
| `auto` | Auto-install after validation | Trusted environments |
| `strict` | Extra validation checks | High-security environments |

**Default**: Always use `review` mode unless you explicitly want auto-installation.

## What Gets Generated?

When you create a behavior, MetaProgrammer generates:

1. **Behavior code** (`.agent_generated/staging/BehaviorName.py`)
   - Inherits from `AgentBehavior`
   - Implements your tool specifications
   - Follows Jetbox architecture patterns
   - Zero cross-behavior dependencies

2. **Test code** (`.agent_generated/staging/test_BehaviorName.py`)
   - Unit tests for all tools
   - Success and error cases
   - Proper mocking

3. **Validation results**
   - Syntax validation (AST parsing)
   - Independence check (no cross-imports)
   - Behavior class structure validation
   - Tool schema validation

4. **Installation** (if safety_mode=auto)
   - Copies to `behaviors/` and `tests/`
   - Creates backups in `.agent_generated/backups/`
   - Rollback available if needed

## Example Workflow

### Step 1: Request a Behavior

"Create a behavior that fetches weather data from an API"

### Step 2: MetaProgrammer Asks Clarifying Questions

- "Which weather API? (OpenWeatherMap, WeatherAPI, NOAA?)"
- "What data points do you need? (temperature, humidity, forecast?)"
- "Should it cache results?"

### Step 3: Generation

MetaProgrammer:
1. Reads templates for patterns
2. Generates behavior code
3. Generates comprehensive tests
4. Validates everything
5. Runs tests in sandbox

### Step 4: Review (if safety_mode=review)

You review the generated code and tests, then approve or request changes.

### Step 5: Installation

Files are installed to `behaviors/` and `tests/` with automatic backups.

## Available Behaviors in MetaProgrammer

The MetaProgrammerAgent comes loaded with:

- **ChatbotBehavior**: Interactive conversation mode
- **CreateBehaviorBehavior**: Behavior generation
- **CreateAgentBehavior**: Agent config generation
- **ValidationBehavior**: Code quality validation
- **SandboxTestBehavior**: Isolated test execution
- **ReadFileToolsBehavior**: Read templates and code
- **WriteFileToolsBehavior**: Manual edits (rare)
- **DirectoryToolsBehavior**: Navigate codebase
- **CommandToolsBehavior**: Run validation commands
- **CompactWhenNearFullBehavior**: Context management
- **LoopDetectionBehavior**: Detect repeated actions
- **WorkspaceTaskNotesBehavior**: Persistent context

## Architecture Principles

All generated behaviors follow these rules:

1. **Single Responsibility**: One behavior = one purpose
2. **Zero Dependencies**: No cross-behavior imports
3. **Composability**: Works with any behavior combination
4. **Safety First**: Validation before installation

## Testing

Run the end-to-end test to verify everything works:

```bash
python test_meta_e2e.py
```

This test demonstrates:
- Loading MetaProgrammer
- Creating a behavior
- Validation pipeline
- Auto-installation
- Complete workflow

## Next Steps

1. **Try it out**: Create a simple behavior
2. **Read templates**: Check `behaviors/templates/` for patterns
3. **Experiment**: Use `dryrun` mode to see what gets generated
4. **Build agents**: Once you have behaviors, create specialized agents

## Metadata Tracking

All generated files include provenance metadata headers for queries and cleanup:

```python
# META: GENERATED_BY=MetaProgrammer
# META: GENERATOR=CreateBehaviorBehavior
# META: AUTHOR=MetaProgrammer
# META: TIMESTAMP=2025-11-07T12:34:56
# META: VERSION=1.0.0
# META: PARENT_REQUEST="Create a calculator behavior"
```

### Query Generated Files

```python
from utils.generation_metadata import (
    get_files_today,
    find_generated_files,
    parse_metadata,
    remove_generated_files
)

# Find all files created today
files_today = get_files_today('.agent_generated')

# Find files by generator
behavior_files = find_generated_files(
    '.agent_generated',
    generator='CreateBehaviorBehavior'
)

# Find files by date range
from datetime import datetime
files = find_generated_files(
    '.agent_generated',
    since=datetime(2025, 11, 1),
    before=datetime(2025, 11, 30)
)

# Parse metadata from file
metadata = parse_metadata('behaviors/MyBehavior.py')
print(metadata['timestamp'])  # 2025-11-07T12:34:56

# Remove files with backup
remove_generated_files(files_today, backup_dir='.agent_generated/removed')
```

## Troubleshooting

**Q: Validation fails with "cross-behavior dependency"**
A: Remove any `from behaviors.X import Y` imports. Behaviors must be independent.

**Q: Tests fail in sandbox**
A: Check the test code - may need adjustment for your tool's actual behavior.

**Q: Installation fails**
A: Check permissions and that the behavior name is unique.

**Q: How do I rollback?**
A: Backups are in `.agent_generated/backups/` - copy back to `behaviors/` or `tests/`.

**Q: How do I remove all behaviors created today?**
A: Use `get_files_today()` and `remove_generated_files()` from `utils/generation_metadata`.

## Documentation

- Full architecture: `docs/SELF_EXTENSIBILITY_PLAN.md`
- Behavior templates: `behaviors/templates/`
- Test templates: `behaviors/templates/behavior_test_template.py`
- Agent templates: `behaviors/templates/agent_config_template.yaml`

---

**Status**: MetaProgrammer is production-ready for creating new behaviors and agents!

🤖 Your Jetbox system can now extend itself autonomously while maintaining safety and composability.
