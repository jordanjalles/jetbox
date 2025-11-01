# Agent Behaviors System

**Version**: 1.0
**Last Updated**: 2025-01-01

## Overview

The Jetbox agent behaviors system provides a composable, config-driven architecture for extending agent capabilities. All agent functionality—context management, tools, utilities—is implemented as pluggable behaviors.

## Key Concepts

### What is a Behavior?

A behavior is a self-contained module that:
- **Injects context** into LLM prompts
- **Provides tools** for the agent to use
- **Handles events** from the agent lifecycle
- **Adds instructions** to the system prompt

### Why Use Behaviors?

1. **Composable**: Mix and match any behaviors
2. **Config-Driven**: Change agent behavior without code changes
3. **Testable**: Each behavior has isolated unit tests
4. **Extensible**: Create custom behaviors easily
5. **Maintainable**: Clear separation of concerns

---

## Available Behaviors

### Context Management Behaviors

These behaviors control how context is built for LLM calls.

#### HierarchicalContextBehavior

**Purpose**: Hierarchical task management with goal → task → subtask structure.

**Use Case**: Task-focused agents (TaskExecutor)

**Features**:
- Goal/Task/Subtask hierarchy in context
- Last N message exchanges (configurable)
- Clears messages on subtask transitions
- Loop detection warnings
- Jetbox notes integration

**Configuration**:
```yaml
behaviors:
  - type: HierarchicalContextBehavior
    params:
      history_keep: 12  # Number of message exchanges to keep
      use_jetbox_notes: true  # Load jetbox notes for context
```

**Tools Provided**:
- `decompose_task(subtasks)` - Break goal into tasks
- `mark_subtask_complete(success, reason)` - Complete current subtask

---

#### CompactWhenNearFullBehavior

**Purpose**: Append all messages until context is near full, then compact via LLM summarization.

**Use Case**: Conversational agents (Orchestrator)

**Features**:
- Appends all messages (no truncation)
- Monitors token usage (75% threshold)
- LLM-based summarization of old messages
- Preserves recent messages

**Configuration**:
```yaml
behaviors:
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 8000  # Token limit before compaction
      recent_keep: 20  # Recent messages to preserve
```

**Tools Provided**:
- `mark_goal_complete(summary)` - Mark goal complete

---

#### SubAgentContextBehavior

**Purpose**: Context for agents doing delegated work.

**Use Case**: TaskExecutor when invoked by Orchestrator

**Features**:
- "DELEGATED GOAL" header
- Append-until-full style
- Higher token limit (128K)
- Jetbox notes for context continuity

**Configuration**:
```yaml
behaviors:
  - type: SubAgentContextBehavior
    params:
      max_tokens: 128000
      recent_keep: 20
```

**Tools Provided**:
- `mark_complete(summary)` - Report success to controlling agent
- `mark_failed(reason)` - Report failure to controlling agent

---

#### ArchitectContextBehavior

**Purpose**: Context for architecture design discussions.

**Use Case**: Architect agent

**Features**:
- Higher token limit for verbose discussions
- Focus on architecture decisions
- No jetbox notes (artifacts are output)

**Configuration**:
```yaml
behaviors:
  - type: ArchitectContextBehavior
    params:
      max_tokens: 32000
      recent_keep: 20
```

**Tools Provided**: None (architect tools come from ArchitectToolsBehavior)

---

### Tool Behaviors

These behaviors provide tools for agents to use.

#### FileToolsBehavior

**Purpose**: File system operations.

**Tools Provided**:
- `write_file(path, content)` - Create/modify files
- `read_file(path, start_line, end_line)` - Read file contents
- `list_dir(path)` - List directory contents

**Configuration**:
```yaml
behaviors:
  - type: FileToolsBehavior
    params: {}
```

**Features**:
- Workspace-aware path resolution
- Automatic parent directory creation
- Ledger logging of all operations
- Safety checks (path traversal prevention)

---

#### CommandToolsBehavior

**Purpose**: Execute commands in the workspace.

**Tools Provided**:
- `run_bash(command, background)` - Run shell commands

**Configuration**:
```yaml
behaviors:
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "pip"]  # Allowed commands
```

**Features**:
- Command whitelist for safety
- Output capture (stdout + stderr)
- Background execution support
- Ledger logging

---

#### ServerToolsBehavior

**Purpose**: Manage long-running servers (dev servers, apps, etc.).

**Tools Provided**:
- `start_server(name, command, port)` - Start a background server
- `stop_server(name)` - Stop a running server
- `check_server(name)` - Check server status
- `list_servers()` - List all servers

**Configuration**:
```yaml
behaviors:
  - type: ServerToolsBehavior
    params: {}
```

**Features**:
- Background process management
- Output capture
- Health checking

---

#### DelegationBehavior

**Purpose**: Delegate work to sub-agents (Orchestrator → TaskExecutor/Architect).

**Tools Provided**:
- `delegate_to_executor(goal, workspace)` - Delegate to TaskExecutor
- `consult_architect(question)` - Consult Architect for design
- `list_workspaces()` - List available workspaces
- `find_workspace(goal_substring)` - Find workspace by goal

**Configuration**:
```yaml
behaviors:
  - type: DelegationBehavior
    params: {}
```

**Features**:
- Automatic workspace creation
- Result capture
- Jetbox notes integration

---

#### ArchitectToolsBehavior

**Purpose**: Create architecture artifacts.

**Tools Provided**:
- `write_architecture_doc(content)` - Write architecture document
- `write_module_spec(module_name, content)` - Write module specification
- `write_task_list(content)` - Write task breakdown

**Configuration**:
```yaml
behaviors:
  - type: ArchitectToolsBehavior
    params: {}
```

**Features**:
- Structured artifact creation
- Workspace-relative paths
- Markdown formatting

---

### Utility Behaviors

These behaviors provide cross-cutting concerns.

#### LoopDetectionBehavior

**Purpose**: Detect infinite loops (repeated actions with same results).

**Features**:
- Tracks action history (last 20 actions)
- Detects identical action+result pairs
- Injects warnings into context
- Suggests alternative approaches

**Configuration**:
```yaml
behaviors:
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5  # Trigger warning after N repeats
```

**Events Used**:
- `on_tool_call` - Record each action

**Context Injection**:
```
⚠️  LOOP DETECTION WARNING:
You appear to be repeating actions:
  • write_file repeated 5x
  • run_bash repeated 5x

Consider trying a different approach.
```

---

## Creating Custom Behaviors

### Basic Template

```python
from behaviors import AgentBehavior
from typing import Any

class MyBehavior(AgentBehavior):
    """Brief description of what this behavior does."""

    def __init__(self, param1: str = "default"):
        """Initialize with configurable parameters."""
        self.param1 = param1

    def get_name(self) -> str:
        """Return unique behavior identifier."""
        return "my_behavior"

    def get_tools(self) -> list[dict[str, Any]]:
        """Provide tools for this behavior."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "What this tool does",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg1": {
                                "type": "string",
                                "description": "First argument"
                            }
                        },
                        "required": ["arg1"]
                    }
                }
            }
        ]

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> dict:
        """Handle tool calls."""
        if tool_name == "my_tool":
            result = self._handle_my_tool(args["arg1"])
            return {"result": result}

        raise NotImplementedError(f"Unknown tool: {tool_name}")

    def _handle_my_tool(self, arg1: str) -> str:
        """Business logic for my_tool."""
        return f"Processed: {arg1}"
```

### Advanced: Context Injection

```python
class ContextInjectionBehavior(AgentBehavior):
    """Injects custom context."""

    def get_name(self) -> str:
        return "context_injection"

    def enhance_context(self, context: list[dict], **kwargs) -> list[dict]:
        """Add custom context after system prompt."""
        # Insert after system prompt (index 0)
        context.insert(1, {
            "role": "user",
            "content": "CUSTOM CONTEXT: Remember to be concise."
        })
        return context
```

### Advanced: Event Handling

```python
class MetricsBehavior(AgentBehavior):
    """Tracks agent metrics."""

    def __init__(self):
        self.tool_counts = {}

    def get_name(self) -> str:
        return "metrics"

    def on_tool_call(self, tool_name: str, args: dict, result: dict, **kwargs):
        """Track tool usage."""
        self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1

    def on_goal_complete(self, success: bool, **kwargs):
        """Report metrics."""
        print("\n=== METRICS ===")
        for tool, count in self.tool_counts.items():
            print(f"{tool}: {count} calls")
```

---

## Configuration Reference

### Config File Structure

```yaml
# agent_config.yaml
behaviors:
  - type: BehaviorClassName  # Must match class name exactly
    params:  # Optional parameters passed to __init__
      param1: value1
      param2: value2

  - type: AnotherBehavior
    params: {}  # Empty params if no configuration needed
```

### Loading Behaviors

**From Agent**:
```python
from base_agent import BaseAgent

agent = BaseAgent(
    name="my_agent",
    workspace="./workspace",
    config_file="my_config.yaml"
)
```

**Programmatically**:
```python
from behaviors import FileToolsBehavior, LoopDetectionBehavior

agent.add_behavior(FileToolsBehavior())
agent.add_behavior(LoopDetectionBehavior(max_repeats=3))
```

---

## Event System

### Available Events

Behaviors can hook into these agent lifecycle events:

1. **on_goal_start(goal, **kwargs)** - When agent.set_goal() called
2. **on_tool_call(tool_name, args, result, **kwargs)** - After each tool execution
3. **on_round_end(round_number, **kwargs)** - After each LLM call + tool execution cycle
4. **on_timeout(elapsed_seconds, **kwargs)** - When goal exceeds time limit
5. **on_goal_complete(success, **kwargs)** - When goal finishes (success or failure)

### Event Delivery

- Events are delivered to **all behaviors** in registration order
- Behaviors can't see each other's event handlers
- Events are independent (one behavior's handler doesn't affect others)

### Example

```python
class LoggingBehavior(AgentBehavior):
    """Logs all events."""

    def get_name(self) -> str:
        return "logging"

    def on_goal_start(self, goal, **kwargs):
        print(f"[LOG] Goal started: {goal}")

    def on_tool_call(self, tool_name, args, result, **kwargs):
        print(f"[LOG] Tool called: {tool_name}")

    def on_goal_complete(self, success, **kwargs):
        print(f"[LOG] Goal {'succeeded' if success else 'failed'}")
```

---

## Best Practices

### Do's

✅ **Keep behaviors focused**: One responsibility per behavior

✅ **Use parameters**: Make behaviors configurable via `__init__` params

✅ **Test in isolation**: Write unit tests for each behavior

✅ **Document clearly**: Add docstrings explaining purpose and usage

✅ **Handle errors gracefully**: Return error dicts instead of raising exceptions

### Don'ts

❌ **Don't depend on other behaviors**: Behaviors should be self-contained

❌ **Don't share state**: Each behavior instance is independent

❌ **Don't use global variables**: Use instance variables instead

❌ **Don't modify kwargs**: Treat event kwargs as read-only

❌ **Don't block**: Keep event handlers fast (< 100ms)

---

## Troubleshooting

### Behavior Not Loading

**Error**: `ModuleNotFoundError: No module named 'behaviors.my_behavior'`

**Solution**: Ensure behavior file exists in `behaviors/` directory with matching filename:
- Class: `MyBehavior`
- File: `behaviors/my_behavior.py`

### Tool Name Conflict

**Error**: `ValueError: Tool 'write_file' already registered`

**Solution**: Two behaviors provide the same tool name. Remove duplicate or rename tool.

### Event Not Firing

**Issue**: Event handler not being called

**Checklist**:
1. Is behavior registered? (`agent.add_behavior(...)`)
2. Is event method named correctly? (`on_tool_call`, not `onToolCall`)
3. Is event actually happening? (Add debug print to verify)

---

## Examples

### Example 1: Minimal Behavior

```python
from behaviors import AgentBehavior

class HelloBehavior(AgentBehavior):
    """Prints hello on goal start."""

    def get_name(self) -> str:
        return "hello"

    def on_goal_start(self, goal, **kwargs):
        print(f"Hello! Starting goal: {goal}")
```

### Example 2: Tool Behavior

```python
from behaviors import AgentBehavior

class CalculatorBehavior(AgentBehavior):
    """Provides calculator tools."""

    def get_name(self) -> str:
        return "calculator"

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ]

    def dispatch_tool(self, tool_name, args, **kwargs):
        if tool_name == "add":
            return {"result": args["a"] + args["b"]}
        raise NotImplementedError(f"Unknown tool: {tool_name}")
```

### Example 3: Context Enhancement

```python
from behaviors import AgentBehavior

class TimestampBehavior(AgentBehavior):
    """Adds timestamp to context."""

    def get_name(self) -> str:
        return "timestamp"

    def enhance_context(self, context, **kwargs):
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        context.insert(1, {
            "role": "user",
            "content": f"Current time: {now}"
        })
        return context
```

---

## API Reference

### AgentBehavior Base Class

```python
class AgentBehavior(ABC):
    """Base class for all behaviors."""

    @abstractmethod
    def get_name(self) -> str:
        """Return unique behavior identifier."""

    def enhance_context(self, context: list[dict], **kwargs) -> list[dict]:
        """Modify context before LLM call."""

    def get_tools(self) -> list[dict]:
        """Return tools provided by this behavior."""

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> dict:
        """Handle tool call."""

    def get_instructions(self) -> str:
        """Return instructions to add to system prompt."""

    # Event handlers
    def on_goal_start(self, goal: str, **kwargs) -> None:
        """Called when goal starts."""

    def on_tool_call(self, tool_name: str, args: dict, result: dict, **kwargs) -> None:
        """Called after each tool execution."""

    def on_round_end(self, round_number: int, **kwargs) -> None:
        """Called at end of each round."""

    def on_timeout(self, elapsed_seconds: float, **kwargs) -> None:
        """Called when goal times out."""

    def on_goal_complete(self, success: bool, **kwargs) -> None:
        """Called when goal completes."""
```

---

## See Also

- **Migration Guide**: `MIGRATION_GUIDE.md`
- **Architecture Documentation**: `AGENT_ARCHITECTURE.md`
- **Main Documentation**: `CLAUDE.md`
- **Configuration System**: `CONFIG_SYSTEM.md`

---

**Document Version**: 1.0
**Last Updated**: 2025-01-01
