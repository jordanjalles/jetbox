# Behaviors Documentation

This document provides complete documentation for Jetbox's composable behavior system, including how to create custom behaviors using the `@tool` decorator.

## Table of Contents

1. [What is a Behavior?](#what-is-a-behavior)
2. [Creating Custom Behaviors](#creating-custom-behaviors)
3. [Tool Decorator (@tool)](#tool-decorator-tool)
4. [Available Behaviors](#available-behaviors)
5. [Configuration](#configuration)
6. [Security (Rule of Two)](#security-rule-of-two)

---

## What is a Behavior?

A behavior is a self-contained module that extends agent capabilities by:

- **Providing tools** for the agent to use (via `@tool` decorator or manual `get_tools()`)
- **Injecting context** into LLM prompts (via `on_initial_context()`, `on_compact()`)
- **Handling events** from the agent lifecycle (via event system)
- **Adding instructions** to the system prompt (via `get_instructions()`)

**Core Principles:**
- **Single Responsibility**: Each behavior does ONE thing
- **Composability**: Behaviors work independently and in any combination
- **No Hidden Dependencies**: No behavior embeds functionality from another
- **Config-Driven**: Behaviors configured via YAML, not hardcoded
- **Event-Driven**: Behaviors respond to lifecycle events independently

---

## Creating Custom Behaviors

### Minimal Example (with @tool decorator)

```python
from behaviors.base import AgentBehavior
from behaviors.tool_decorator import tool

class GreetingBehavior(AgentBehavior):
    """Simple behavior that provides greeting tools."""

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "greeting"

    @tool
    def say_hello(self, name: str, enthusiastic: bool = False) -> str:
        """Greet someone by name.

        Args:
            name: Person's name to greet
            enthusiastic: Add excitement (default: false)

        Returns:
            Greeting message
        """
        greeting = f"Hello, {name}!"
        if enthusiastic:
            greeting += " 🎉"
        return greeting
```

**That's it!** The `@tool` decorator automatically:
- Generates JSON schema from type hints
- Registers the tool for auto-discovery
- Handles dispatch routing by method name
- Extracts parameter descriptions from docstring

### Advanced Example (with context injection)

```python
from behaviors.base import AgentBehavior
from behaviors.tool_decorator import tool
from typing import Any

class WeatherBehavior(AgentBehavior):
    """Provides weather tools and injects current conditions into context."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.current_weather = None

    def get_name(self) -> str:
        return "weather"

    def on_initial_context(self, agent, context: list[dict]) -> list[dict]:
        """Inject current weather into initial context."""
        if self.current_weather:
            context.append({
                "role": "system",
                "content": f"Current weather: {self.current_weather['temp']}°F, {self.current_weather['conditions']}"
            })
        return context

    @tool
    def get_weather(self, location: str) -> dict[str, Any]:
        """Get current weather for a location.

        Args:
            location: City name or ZIP code

        Returns:
            Weather data with temperature and conditions
        """
        # Call weather API
        self.current_weather = {"temp": 72, "conditions": "sunny", "location": location}
        return self.current_weather
```

---

## Tool Decorator (@tool)

The `@tool` decorator is the **recommended way** to create agent tools. It eliminates manual JSON schema boilerplate.

### Basic Usage

```python
from behaviors.tool_decorator import tool

class MyBehavior(AgentBehavior):
    @tool
    def my_tool(self, param1: str, param2: int = 10) -> dict:
        """Short tool description goes here as first line.

        Args:
            param1: Description of param1
            param2: Description of param2 (default: 10)

        Returns:
            Result dictionary
        """
        # Access agent via self.agent (injected automatically)
        workspace = self.agent.workspace

        return {"result": f"Processed {param1} with {param2}"}
```

### Type Hint Mappings

The decorator automatically converts Python type hints to JSON schema:

| Python Type | JSON Schema |
|------------|-------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list[T]` | `{"type": "array", "items": {...}}` |
| `dict` | `{"type": "object"}` |
| `Optional[T]` / `T \| None` | Not in `required` array |
| `Literal['a', 'b']` | `{"type": "string", "enum": ["a", "b"]}` |

### Docstring Format

The decorator parses **Google-style** and **NumPy-style** docstrings:

**Google-style:**
```python
@tool
def read_file(self, path: str, encoding: str = "utf-8") -> str:
    """Read file contents.

    Args:
        path: File path to read
        encoding: Text encoding (default: utf-8)

    Returns:
        File contents as string
    """
```

**NumPy-style:**
```python
@tool
def write_file(self, path: str, content: str) -> str:
    """Write content to a file.

    Parameters
    ----------
    path : str
        File path to write
    content : str
        Content to write

    Returns
    -------
    str
        Success message
    """
```

### Accessing Agent Context

Decorated methods can access the agent instance via `self.agent`:

```python
@tool
def list_workspace_files(self) -> list[str]:
    """List all files in the agent's workspace.

    Returns:
        List of file paths
    """
    # Access agent properties
    workspace_dir = self.agent.workspace
    ledger_file = self.agent.ledger_file

    # Access workspace manager
    workspace_manager = getattr(self.agent, 'workspace_manager', None)
    if workspace_manager:
        return workspace_manager.list_files()

    return []
```

### Optional Description Override

By default, the decorator uses the **first line** of the docstring as the tool description. You can override this:

```python
@tool(description="Custom tool description here")
def my_tool(self, param: str) -> str:
    """This docstring is ignored for the tool description.

    Args:
        param: Parameter description
    """
    pass
```

**Recommended:** Use docstring-based descriptions (DRY principle).

---

## Available Behaviors

### File Operations

#### **WriteFileToolsBehavior**
- `write_file(path, content, append, encoding, line_end, overwrite)` - Write/overwrite files

#### **ReadFileToolsBehavior**
- `read_file(path, encoding, max_size)` - Read text files (up to 1MB)

#### **DirectoryToolsBehavior**
- `list_dir(path, depth)` - List directory contents (optionally recursive)

### Command Execution

#### **CommandToolsBehavior**
- `run_bash(command, timeout)` - Run bash commands with whitelist validation

### Server Management

#### **ServerToolsBehavior**
- `start_server(cmd, name)` - Start background server process
- `stop_server(server_id)` - Stop running server
- `check_server(server_id, tail_lines)` - Check status and logs
- `list_servers()` - List all running servers

### Context Management

#### **CompactWhenNearFullBehavior**
- Automatically compacts message history when context nears token limit
- Uses LLM summarization to preserve important information

#### **WorkspaceTaskNotesBehavior**
- Auto-summarizes completed tasks/goals
- Persists summaries to `workspace_task_notes.md`
- Loads notes on subsequent runs for context continuity

### Delegation

#### **DelegationBehavior**
- Dynamically generates delegation tools based on `can_delegate_to` relationships
- Example: `delegate_to_architect(project_description, requirements, constraints)`
- Tracks delegations for reporting

### Smart Home Control

#### **HomeAssistantBehavior**
- `ha_list_devices(domain, area)` - List devices/entities
- `ha_get_state(entity_id)` - Get device state
- `ha_call_service(domain, service, entity_id, data)` - Control devices
- `ha_list_automations()` - List automations
- `ha_trigger_automation(entity_id)` - Trigger automation

### Agent Lifecycle

#### **ExecutionModeBehavior**
- `activate_execution_mode()` - Activate task execution mode
- Enforces "at least one tool per round" requirement
- Provides completion nudging heuristics

#### **ChatbotBehavior**
- `activate_chat_mode()` - Switch to conversational mode
- `ask_followup(question)` - Ask user clarifying questions
- `continue_conversation(response)` - Continue multi-turn chat

### Task Management

#### **TaskManagementBehavior**
- `mark_complete(summary)` - Mark current task complete
- `mark_failed(error, details)` - Mark current task failed
- `mark_blocked(blocker)` - Mark current task blocked
- `reassess_approach()` - Trigger approach reconsideration

### Architecture Design

#### **ArchitectToolsBehavior**
- `write_architecture_doc(content)` - Write architecture.md
- `write_requirements(content)` - Write requirements.md
- `write_implementation_plan(content)` - Write implementation_plan.md
- `update_architecture_notes(section, content)` - Update specific sections

### Code Quality

#### **ValidationBehavior**
- `validate_python(path)` - Run ruff on Python files
- `validate_typescript(path)` - Run ESLint/Prettier
- `validate_tests(path, framework)` - Run test suite
- `validate_build(command)` - Run build process
- And 4 more validation tools...

### Utilities

#### **LoopDetectionBehavior**
- Detects infinite loops (repeated identical tool calls)
- Auto-terminates after configurable threshold

#### **TimeBoxBehavior**
- `schedule_reminder(at_percent, message)` - Schedule future reminders
- Provides automatic time nudges at 25%, 50%, 75%

### Workspace Management

#### **WorkspaceManagementBehavior**
- `create_workspace(name)` - Create isolated workspace
- `list_workspaces()` - List all workspaces
- `switch_workspace(name)` - Switch to different workspace
- `cleanup_workspace(name)` - Remove workspace

### Development Tools

#### **CreateAgentBehavior**
- `create_agent(name, description, behaviors)` - Generate new agent config

#### **CreateBehaviorBehavior**
- `create_behavior(name, description, tools)` - Scaffold new behavior

#### **SandboxTestBehavior**
- `run_sandboxed_test(code, timeout)` - Execute code in isolated env
- `verify_sandbox_safety(code)` - Check code safety

---

## Configuration

Behaviors are configured in YAML files (e.g., `config/agents/orchestrator.yaml`):

```yaml
behaviors:
  # File operations
  - type: WriteFileToolsBehavior
    params: {}

  - type: ReadFileToolsBehavior
    params: {}

  # Commands with whitelist
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "pip", "git"]

  # Context management
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 128000
      compact_at_percent: 75

  # Delegation (auto-configured from agents.yaml)
  - type: DelegationBehavior
    params:
      workspace_strategy: "enforce_inherit"

  # Task management
  - type: TaskManagementBehavior
    params: {}

  # Loop detection
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5
```

### Behavior Parameters

Most behaviors accept optional parameters:

- **CompactWhenNearFullBehavior**: `max_tokens`, `compact_at_percent`
- **CommandToolsBehavior**: `whitelist`
- **DelegationBehavior**: `workspace_strategy` ("enforce_inherit", "enforce_new", "llm_chooses")
- **LoopDetectionBehavior**: `max_repeats`
- **TimeBoxBehavior**: `max_rounds`, `reminder_percentages`

---

## Security (Rule of Two)

Behaviors declare their security properties for **Rule of Two** validation:

```python
from behaviors.rule_of_two_types import RuleOfTwoProperty

class MyBehavior(AgentBehavior):
    rule_of_two_properties = {
        RuleOfTwoProperty.EXTERNAL_ACTION  # Communicates externally
    }
```

**Available properties:**
- `UNTRUSTED_INPUT` - Processes external/untrusted data
- `SENSITIVE_ACCESS` - Accesses credentials, API keys, .env files
- `EXTERNAL_ACTION` - Communicates externally (network, git push, npm publish)

**Rule of Two**: An agent should satisfy ≤2 of [UNTRUSTED_INPUT, SENSITIVE_ACCESS, EXTERNAL_ACTION].

See [docs/security/README.md](docs/security/README.md) for complete security documentation.

---

## Migration from Manual Tool Registration

**Old way (deprecated):**
```python
def get_tools(self):
    return [{
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file...",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "..."},
                    "content": {"type": "string", "description": "..."}
                },
                "required": ["path", "content"]
            }
        }
    }]

def dispatch_tool(self, agent, tool_name, args):
    if tool_name == "write_file":
        return self._write_file(args["path"], args["content"])
    return super().dispatch_tool(agent, tool_name, args)

def _write_file(self, path, content):
    # implementation
```

**New way (recommended):**
```python
from behaviors.tool_decorator import tool

@tool
def write_file(self, path: str, content: str) -> str:
    """Write a file to disk.

    Args:
        path: File path
        content: Content to write

    Returns:
        Success message
    """
    # implementation (access agent via self.agent)
```

**Benefits:**
- 60-75% less code per behavior
- No manual JSON schema writing
- Type-safe parameters via Python type hints
- Auto-extracted docstring descriptions
- Easier to maintain and extend

See [docs/tool_decorator_migration/](docs/tool_decorator_migration/) for complete migration documentation.

---

## Best Practices

1. **Keep behaviors focused** - One responsibility per behavior
2. **Use @tool decorator** - Eliminates boilerplate, improves maintainability
3. **Document thoroughly** - Write clear docstrings with Args/Returns
4. **Type hint everything** - Enables automatic schema generation
5. **Handle errors gracefully** - Return error dicts instead of raising exceptions
6. **Test independently** - Each behavior should have isolated unit tests
7. **Avoid hidden dependencies** - Don't call other behaviors' methods directly
8. **Use event system** - For cross-behavior communication

---

## Related Documentation

- [CLAUDE.md](CLAUDE.md) - Complete project guide
- [Tool Decorator Migration](docs/tool_decorator_migration/) - Migration guide and metrics
- [Security Architecture](docs/security/SECURITY_ARCHITECTURE.md) - Rule of Two system
- [Workspace Delegation](docs/WORKSPACE_DELEGATION_STRATEGY.md) - Delegation patterns
