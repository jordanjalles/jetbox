# Agent Behaviors System

**Version**: 1.0
**Last Updated**: 2025-01-01

## Overview

The Jetbox agent behaviors system provides a composable, config-driven architecture for extending agent capabilities. All agent functionality—context management, tools, utilities—is implemented as pluggable behaviors.

## Architecture Principles

The behavior system is built on six core principles that ensure composability and maintainability:

### 1. Single Responsibility
Each behavior does ONE thing and does it well. A behavior should not mix multiple concerns.

**Good**: `FileToolsBehavior` - provides file operations only
**Bad**: `FileAndCommandBehavior` - mixing file and command tools

### 2. Composability
Behaviors work independently and in any combination. You can mix and match behaviors without conflicts.

**Example**: `SubAgentContextBehavior + CompactWhenNearFullBehavior + WorkspaceTaskNotesBehavior` all work together without interference.

### 3. No Hidden Dependencies
No behavior embeds functionality from another. Each behavior is self-contained.

**Good**: `WorkspaceTaskNotesBehavior` loads notes, `SubAgentContextBehavior` injects goal header
**Bad**: `SubAgentContextBehavior` that also loads notes internally

### 4. Config-Driven
Behaviors are configured via YAML files, not hardcoded in agent constructors.

**Good**: Load from `task_executor_config.yaml`
**Bad**: Hardcoded `agent.add_behavior(FileToolsBehavior())` in `__init__`

### 5. Event-Driven
Behaviors respond to lifecycle events (`on_goal_start`, `on_tool_call`, etc.) without knowing about other behaviors.

**Example**: `LoopDetectionBehavior` tracks actions via `on_tool_call` independently.

### 6. Clear Interfaces
All behaviors implement standardized methods: `get_tools()`, `enhance_context()`, `dispatch_tool()`, event handlers.

**Example**: Every context behavior implements `enhance_context()` but doesn't require other methods.

---

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
- "DELEGATED GOAL" header injection
- COMPOSABLE: Does NOT handle compaction (use CompactWhenNearFullBehavior)
- COMPOSABLE: Does NOT load notes (use WorkspaceTaskNotesBehavior)
- ONLY manages delegated goal context and completion tools

**Configuration**:
```yaml
behaviors:
  - type: SubAgentContextBehavior
    params: {}  # No params - pure context injection
```

**Tools Provided**:
- `mark_complete(summary)` - Report success to controlling agent
- `mark_failed(reason)` - Report failure to controlling agent

**Composition Pattern**:
```yaml
# Full TaskExecutor stack
behaviors:
  - type: SubAgentContextBehavior  # Delegated goal context
  - type: CompactWhenNearFullBehavior  # Context compaction
  - type: WorkspaceTaskNotesBehavior  # Notes loading
  - type: FileToolsBehavior  # File operations
  - type: CommandToolsBehavior  # Command execution
```

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

**Features**:
- **Auto-configured from agents.yaml**: Reads `can_delegate_to` relationships
- Dynamically generates delegation tools based on relationships
- Injects delegatable agent descriptions into context
- No hardcoded agent references

**Tools Provided** (auto-generated):
- `consult_architect(project_description, requirements, constraints)` - Consult Architect for design
- `delegate_to_executor(task_description, workspace_mode, workspace_path)` - Delegate to TaskExecutor

**Configuration**:
```yaml
behaviors:
  # NOTE: DelegationBehavior is auto-added by BaseAgent when can_delegate_to is present
  # You don't need to list it explicitly in config files
```

**Example agents.yaml**:
```yaml
agents:
  orchestrator:
    can_delegate_to:
      - architect
      - task_executor
```

**Auto-Configuration**:
When an agent has `can_delegate_to` relationships in `agents.yaml`, `BaseAgent` automatically:
1. Loads delegation relationships from YAML
2. Creates `DelegationBehavior` with those relationships
3. Adds the behavior to the agent's behavior list
4. Generates appropriate delegation tools dynamically

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

#### WorkspaceTaskNotesBehavior

**Purpose**: Persistent context summaries across task boundaries.

**Features**:
- Loads existing notes from `workspace_task_notes.md` in workspace
- Auto-summarizes completed goals (success/failure)
- Creates timeout summaries when agent times out
- Persists summaries for context continuity
- Warns if notes exceed 10% of max context

**Configuration**:
```yaml
behaviors:
  - type: WorkspaceTaskNotesBehavior
    params: {}
```

**Events Used**:
- `on_goal_start` - Setup workspace and clear cache
- `on_goal_complete` - Generate and save goal summary
- `on_timeout` - Generate timeout summary

**Context Enhancement**:
```
## Previous Context (from workspace task notes)

[Loaded notes content from workspace_task_notes.md]
```

**Note**: Previously named `JetboxNotesBehavior`. The underlying `jetbox_notes.py` module name is kept to avoid breaking imports, but all file outputs use `workspace_task_notes.md`.

---

## Creating Custom Behaviors

### Composability Checklist

Before implementing a behavior, ensure it follows composability principles:

- [ ] **Single Responsibility**: Does ONE thing only
- [ ] **No Hidden Dependencies**: Doesn't embed functionality from other behaviors
- [ ] **Self-Contained**: Can be instantiated and used alone
- [ ] **No Shared State**: Doesn't rely on global variables or external state
- [ ] **Accepts kwargs**: All methods accept `**kwargs` for forward compatibility
- [ ] **Unique Tool Names**: Tools don't conflict with other behaviors
- [ ] **Event Independence**: Event handlers don't assume other behaviors exist

### Design Patterns

#### GOOD Behavior Design

```python
class WorkspaceTaskNotesBehavior(AgentBehavior):
    """
    GOOD: Single responsibility (load/save notes only).
    Does NOT handle compaction or context management.
    """
    def enhance_context(self, context, **kwargs):
        # ONLY loads notes and injects
        notes = self._load_notes()
        if notes:
            context.insert(1, {"role": "user", "content": notes})
        return context
```

#### BAD Behavior Design

```python
class SubAgentWithNotesBehavior(AgentBehavior):
    """
    BAD: Multiple responsibilities (context + notes + compaction).
    Embeds functionality that should be separate behaviors.
    """
    def enhance_context(self, context, **kwargs):
        # BAD: Doing too much in one behavior
        context = self._inject_goal_header(context)  # SubAgent concern
        notes = self._load_notes()  # Notes concern
        context = self._compact_if_full(context)  # Compaction concern
        return context
```

### Basic Template

```python
from behaviors import AgentBehavior
from typing import Any

class MyBehavior(AgentBehavior):
    """Brief description of what this behavior does."""

    def __init__(self, param1: str = "default", **kwargs):
        """
        Initialize with configurable parameters.

        Always accept **kwargs for forward compatibility!
        """
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
                    "name": "my_tool",  # Ensure unique across all behaviors
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
        """
        Handle tool calls.

        Always accept **kwargs for additional context!
        """
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

## Lifecycle API

### Overview

The behavior lifecycle API provides explicit, event-driven hooks with **agent-first signatures**. This makes timing semantics clear and gives behaviors direct access to agent state.

**Key Improvements**:
- ✅ **Explicit timing**: Method names indicate WHEN they're called (once vs every round)
- ✅ **Agent access**: All events receive `agent` as first parameter
- ✅ **No kwargs soup**: Clear, typed parameters instead of `**kwargs`
- ✅ **Context efficiency**: Static content injected once, dynamic content every round

### Lifecycle Event Flow

```
Goal Start
    ↓
on_goal_start(agent, goal)              # Called ONCE at goal initialization
    ↓
on_initial_context(agent, context)      # Called ONCE for first-time context setup
    ↓                                    # (inject tool definitions, goal, notes)
╔═══════════════════════════╗
║   MAIN EXECUTION LOOP     ║
╠═══════════════════════════╣
║ on_round_start(agent,     ║          # Called EVERY round before LLM call
║   round_number, context)  ║          # (inject dynamic warnings, prompts)
║        ↓                  ║
║ → LLM generates response  ║
║        ↓                  ║
║ on_llm_response(agent,    ║          # Called after LLM responds
║   response)               ║
║        ↓                  ║
║ dispatch_tool(agent,      ║          # Execute tool calls
║   tool_name, args)        ║
║        ↓                  ║
║ on_tool_call(agent,       ║          # Called after each tool execution
║   tool_name, args, result)║
║        ↓                  ║
║ on_round_end(agent,       ║          # Called at end of each round
║   round_number)           ║
╚═══════════════════════════╝
    ↓
on_goal_complete(agent, success, summary)  # Called on goal completion
    or
on_timeout(agent, elapsed_seconds)         # Called on timeout
```

### Available Lifecycle Events

#### 1. on_goal_start(agent, goal)

**When**: Called ONCE when agent.set_goal() is called

**Purpose**: Initialize behavior state, setup workspace, prepare for goal execution

**Parameters**:
- `agent` - The agent instance (access via `agent.workspace`, `agent.state`, etc.)
- `goal` - Goal string

**Example**:
```python
def on_goal_start(self, agent, goal):
    """Initialize on goal start."""
    self.workspace = agent.workspace
    self.notes_path = self.workspace / "workspace_task_notes.md"
    print(f"[behavior] Goal started: {goal}")
```

---

#### 2. on_initial_context(agent, context)

**When**: Called ONCE at the start of goal execution (before first LLM call)

**Purpose**: Inject static content that doesn't change across rounds (tool definitions, goal description, loaded notes)

**Parameters**:
- `agent` - The agent instance
- `context` - List of message dicts to be sent to LLM

**Example**:
```python
def on_initial_context(self, agent, context):
    """Inject tool definitions and goal (once only)."""
    # Inject tool definitions
    tools_text = self._format_tool_definitions()
    context.insert(1, {"role": "system", "content": tools_text})

    # Load workspace notes (once only)
    if self.notes_path.exists():
        notes = self.notes_path.read_text()
        context.append({"role": "system", "content": f"## Previous Context\n\n{notes}"})
```

**Why separate from on_round_start?**
- Avoids re-injecting tool definitions every round (~2KB saved per round)
- Loads notes once instead of every round
- Makes it clear what's static vs dynamic

---

#### 3. on_round_start(agent, round_number, context)

**When**: Called at the start of EVERY round (before LLM call)

**Purpose**: Inject dynamic content that changes each round (warnings, prompts, round-specific data)

**Parameters**:
- `agent` - The agent instance
- `round_number` - Current round number (1-indexed)
- `context` - List of message dicts to be sent to LLM

**Example**:
```python
def on_round_start(self, agent, round_number, context):
    """Inject loop detection warning if needed."""
    if self._detect_loop():
        warning = (
            "⚠️  LOOP DETECTION WARNING:\n"
            "You appear to be repeating actions. Consider a different approach."
        )
        context.append({"role": "system", "content": warning})
```

---

#### 4. on_llm_response(agent, response)

**When**: Called after LLM responds, before tool dispatch

**Purpose**: Inspect or modify LLM response, log for debugging, detect completion signals

**Parameters**:
- `agent` - The agent instance
- `response` - LLM response dict with 'content' and 'tool_calls'

**Example**:
```python
def on_llm_response(self, agent, response):
    """Log LLM response for debugging."""
    content = response.get("content", "")
    print(f"[behavior] LLM said: {content[:100]}...")
```

---

#### 5. on_tool_call(agent, tool_name, args, result)

**When**: Called after EACH tool execution

**Purpose**: Track actions, detect patterns, log operations

**Parameters**:
- `agent` - The agent instance
- `tool_name` - Name of tool that was executed
- `args` - Arguments passed to tool
- `result` - Result returned by tool

**Example**:
```python
def on_tool_call(self, agent, tool_name, args, result):
    """Track action history for loop detection."""
    action_key = f"{tool_name}:{str(args)}"
    self.action_history.append(action_key)

    # Detect repeated actions
    if self.action_history.count(action_key) > 5:
        self.loop_detected = True
```

---

#### 6. on_round_end(agent, round_number)

**When**: Called at the end of each round (after tool execution)

**Purpose**: Update statistics, render status display, persist state

**Parameters**:
- `agent` - The agent instance
- `round_number` - Current round number (1-indexed)

**Example**:
```python
def on_round_end(self, agent, round_number):
    """Render status display."""
    if round_number % 5 == 0:  # Every 5 rounds
        status = self._render_status(agent)
        print(status)
```

---

#### 7. on_goal_complete(agent, success, summary)

**When**: Called when goal completes (success or failure)

**Purpose**: Save summaries, persist state, cleanup resources

**Parameters**:
- `agent` - The agent instance
- `success` - True if goal succeeded, False if failed
- `summary` - Optional summary string from mark_complete() tool

**Example**:
```python
def on_goal_complete(self, agent, success, summary):
    """Save goal summary to workspace notes."""
    if success and summary:
        self._append_to_notes(f"✅ {summary}\n")
    else:
        self._append_to_notes(f"❌ Goal failed\n")
```

---

#### 8. on_timeout(agent, elapsed_seconds)

**When**: Called when goal exceeds time limit

**Purpose**: Save timeout summaries, cleanup, log timing

**Parameters**:
- `agent` - The agent instance
- `elapsed_seconds` - Time elapsed since goal start

**Example**:
```python
def on_timeout(self, agent, elapsed_seconds):
    """Generate timeout summary."""
    summary = f"Goal timed out after {elapsed_seconds}s"
    self._append_to_notes(f"⏱️  {summary}\n")
```

---

#### 9. dispatch_tool(agent, tool_name, args)

**When**: Called when agent attempts to execute a tool provided by this behavior

**Purpose**: Execute tool logic with full agent access

**Parameters**:
- `agent` - The agent instance
- `tool_name` - Name of tool to execute
- `args` - Arguments dict for the tool

**Returns**: Tool result (any type - dict, str, etc.)

**Example**:
```python
def dispatch_tool(self, agent, tool_name, args):
    """Execute tool with agent access."""
    if tool_name == "write_file":
        path = agent.workspace / args["path"]
        path.write_text(args["content"])
        return {"success": True, "path": str(path)}

    return {"error": f"Unknown tool: {tool_name}"}
```

---

### Event Delivery

- Events are delivered to **all behaviors** in registration order
- Behaviors can't see each other's event handlers
- Events are independent (one behavior's handler doesn't affect others)
- All events receive `agent` as first parameter for direct state access

### Complete Behavior Example

```python
class LoggingBehavior(AgentBehavior):
    """Logs all lifecycle events."""

    def get_name(self) -> str:
        return "logging"

    def on_goal_start(self, agent, goal):
        """Log goal start."""
        print(f"[LOG] Goal started: {goal}")
        print(f"[LOG] Workspace: {agent.workspace}")

    def on_initial_context(self, agent, context):
        """Log initial context setup."""
        print(f"[LOG] Initial context: {len(context)} messages")

    def on_round_start(self, agent, round_number, context):
        """Log round start."""
        print(f"[LOG] Round {round_number} starting")

    def on_llm_response(self, agent, response):
        """Log LLM response."""
        content = response.get("content", "")[:50]
        print(f"[LOG] LLM: {content}...")

    def on_tool_call(self, agent, tool_name, args, result):
        """Log tool execution."""
        print(f"[LOG] Tool: {tool_name}({args})")

    def on_round_end(self, agent, round_number):
        """Log round end."""
        print(f"[LOG] Round {round_number} complete")

    def on_goal_complete(self, agent, success, summary):
        """Log completion."""
        status = "succeeded" if success else "failed"
        print(f"[LOG] Goal {status}: {summary}")

    def on_timeout(self, agent, elapsed_seconds):
        """Log timeout."""
        print(f"[LOG] Timeout after {elapsed_seconds}s")
```

### Migration from Old API

**Old API (Deprecated)**:
```python
def enhance_context(self, context, **kwargs):
    """Called every round - timing unclear."""
    agent = kwargs.get("agent")  # Awkward kwargs access
    # Injected tool definitions EVERY round (inefficient)
```

**New API (Preferred)**:
```python
def on_initial_context(self, agent, context):
    """Called ONCE - timing explicit."""
    # Inject tool definitions ONCE (efficient)
    # Direct agent access, no kwargs

def on_round_start(self, agent, round_number, context):
    """Called EVERY round - timing explicit."""
    # Inject dynamic warnings only
```

**Benefits**:
- Explicit timing (once vs every round)
- Agent-first signatures (no kwargs soup)
- Context efficiency (static content once, dynamic every round)
- Better type hints and IDE support

---

## Behavior Composition Examples

This section shows how behaviors compose to create different agent capabilities.

### Example 1: TaskExecutor - Full Behavior Stack

The TaskExecutor combines multiple behaviors for comprehensive coding capabilities:

```yaml
# task_executor_config.yaml
behaviors:
  # Context management (2 behaviors)
  - type: SubAgentContextBehavior        # Delegated goal header
  - type: CompactWhenNearFullBehavior    # Context compaction when near full

  # Tool behaviors (3 behaviors)
  - type: FileToolsBehavior              # write_file, read_file, list_dir
  - type: CommandToolsBehavior           # run_bash
  - type: ServerToolsBehavior            # start_server, stop_server, check_server

  # Utility behaviors (3 behaviors)
  - type: LoopDetectionBehavior          # Detect repeated actions
  - type: WorkspaceTaskNotesBehavior     # Load/save persistent notes
  - type: StatusDisplayBehavior          # Progress visualization
```

**Result**: TaskExecutor can:
- Accept delegated tasks with proper context framing
- Manage context efficiently (compaction)
- Perform file operations
- Execute commands
- Manage servers
- Detect and warn about loops
- Persist context across runs
- Display progress

**Composition Benefits**:
- Each behavior is independent
- Can remove behaviors without breaking others (e.g., remove ServerToolsBehavior if not needed)
- Can add new behaviors dynamically
- No conflicts between behaviors

### Example 2: Orchestrator - Minimal Stack

The Orchestrator uses fewer behaviors for conversational coordination:

```yaml
# orchestrator_config.yaml
behaviors:
  # Context management
  - type: CompactWhenNearFullBehavior    # Append-until-full style

  # Utility behaviors
  - type: LoopDetectionBehavior          # Detect repeated actions

  # NOTE: DelegationBehavior auto-added from agents.yaml relationships
```

**Result**: Orchestrator can:
- Maintain long conversations without manual compaction
- Detect conversation loops
- Delegate to architect/task_executor (auto-configured)

**Why Minimal?**
- Orchestrator doesn't need file tools (delegates to TaskExecutor)
- Orchestrator doesn't need server tools (delegates to TaskExecutor)
- Orchestrator doesn't need workspace notes (no persistent task context)

### Example 3: Custom Agent - Cherry-Picked Behaviors

Create a specialized agent for code review:

```yaml
# code_reviewer_config.yaml
behaviors:
  # Context management
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 16000  # Medium context for code review

  # Tool behaviors (file read only, no write)
  - type: FileToolsBehavior
    params: {}

  # Custom behavior for code analysis
  - type: CodeAnalysisBehavior
    params:
      languages: ["python", "javascript", "typescript"]

  # Utility
  - type: LoopDetectionBehavior
    params:
      max_repeats: 3  # Strict loop detection
```

**Result**: Code reviewer can:
- Read code files
- Analyze code patterns
- Provide feedback
- NOT write files (read-only agent)

### Example 4: Testing Composability

Test behaviors in different combinations:

```python
# Test 1: Minimal agent (context only)
agent1 = BaseAgent(name="minimal", workspace=".", config_file="minimal_config.yaml")
# behaviors: [CompactWhenNearFullBehavior]

# Test 2: Tools only (no context management)
agent2 = BaseAgent(name="tools_only", workspace=".", config_file="tools_config.yaml")
# behaviors: [FileToolsBehavior, CommandToolsBehavior]

# Test 3: Full stack
agent3 = BaseAgent(name="full", workspace=".", config_file="task_executor_config.yaml")
# behaviors: [SubAgent, Compact, Files, Commands, Servers, Loop, Notes, Status]

# Test 4: Custom combination
agent4 = BaseAgent(name="custom", workspace=".", config_file=None)
agent4.add_behavior(CompactWhenNearFullBehavior(max_tokens=8000))
agent4.add_behavior(FileToolsBehavior())
agent4.add_behavior(LoopDetectionBehavior(max_repeats=3))
```

**Testing Principles**:
- Each combination should work without errors
- Behaviors should not interfere with each other
- Order shouldn't matter (test different orders)
- Adding/removing behaviors should be safe

---

## Testing Behaviors

### Isolation Testing

Test each behavior independently:

```python
def test_file_tools_behavior_isolation():
    """FileToolsBehavior works without other behaviors."""
    behavior = FileToolsBehavior()

    # Test tool registration
    tools = behavior.get_tools()
    assert len(tools) == 3  # write_file, read_file, list_dir

    # Test tool dispatch
    result = behavior.dispatch_tool("write_file", {
        "path": "test.txt",
        "content": "hello"
    })
    assert result["success"] == True
```

### Composition Testing

Test behaviors working together:

```python
def test_context_behaviors_compose():
    """SubAgent + Compact + Notes compose correctly."""
    behaviors = [
        SubAgentContextBehavior(),
        CompactWhenNearFullBehavior(max_tokens=8000),
        WorkspaceTaskNotesBehavior()
    ]

    # Build context through all behaviors
    context = [{"role": "system", "content": "You are an agent"}]
    for behavior in behaviors:
        context = behavior.enhance_context(context, **test_kwargs)

    # Verify each behavior contributed
    assert any("DELEGATED GOAL" in msg["content"] for msg in context)
    assert any("Previous Context" in msg.get("content", "") for msg in context)
```

### Independence Testing

Test that behaviors don't depend on each other:

```python
def test_behavior_independence():
    """Each behavior can be instantiated alone."""
    behaviors_to_test = [
        FileToolsBehavior,
        CommandToolsBehavior,
        SubAgentContextBehavior,
        CompactWhenNearFullBehavior,
        WorkspaceTaskNotesBehavior,
        LoopDetectionBehavior
    ]

    for BehaviorClass in behaviors_to_test:
        # Should instantiate without errors
        behavior = BehaviorClass()

        # Should have unique name
        assert behavior.get_name()

        # Methods should be callable (even if they return empty)
        tools = behavior.get_tools()
        assert isinstance(tools, list)
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

- **Extension Guide**: `EXTENDING_BEHAVIORS.md` - Complete guide for creating custom behaviors
- **Migration Guide**: `MIGRATION_GUIDE.md` - Migrating from old context strategies
- **Architecture Documentation**: `AGENT_ARCHITECTURE.md` - Overall system architecture
- **Main Documentation**: `CLAUDE.md` - User guide and quick reference
- **Configuration System**: `CONFIG_SYSTEM.md` - Agent configuration reference

## Quick Links

- **Config Files**: `task_executor_config.yaml`, `orchestrator_config.yaml`, `architect_config.yaml`
- **Behavior Source**: `behaviors/` directory
- **Tests**: `tests/test_*behavior*.py`
- **Agent Relationships**: `agents.yaml`

---

**Document Version**: 1.1
**Last Updated**: 2025-01-01
**Changes**: Added architecture principles, composability examples, and comprehensive testing guidance
