# Multi-Agent Architecture

This document describes the multi-agent system architecture implemented in Jetbox.

## Overview

Jetbox now supports multiple agent types working together:

- **Orchestrator**: User-facing conversational agent that plans and delegates
- **TaskExecutor**: Coding agent that executes specific tasks with hierarchical decomposition

## Architecture Components

### 1. BaseAgent (base_agent.py)

Abstract base class providing common functionality for all agents:

**Properties:**
- `name`: Agent identifier
- `role`: Human-readable description
- `workspace`: Working directory
- `config`: Configuration from agent_config.yaml
- `state`: AgentState tracking messages and rounds

**Abstract Methods (must implement):**
- `get_tools()`: Returns tool definitions for this agent
- `get_system_prompt()`: Returns system prompt
- `get_context_strategy()`: Returns context management strategy name
- `build_context()`: Builds context for LLM call

**Shared Methods:**
- `call_llm()`: Calls LLM with context and tools
- `dispatch_tool()`: Executes tool calls
- `persist_state()` / `load_state()`: State management
- `add_message()` / `get_message_history()`: Message management

### 2. OrchestratorAgent (orchestrator_agent.py)

User-facing agent for conversation and delegation.

**Context Strategy:** `append_until_full`
- Appends all messages until approaching token limit (80% of max_tokens)
- When near limit, performs compaction pass:
  - Keeps recent 20 messages intact
  - Summarizes older messages into single summary message

**Tools:**
- `delegate_to_executor`: Send coding task to TaskExecutor
- `clarify_with_user`: Ask user clarifying questions
- `create_task_plan`: Structure complex requests into tasks
- `get_executor_status`: Check TaskExecutor progress

**System Prompt:**
```
You are an orchestrator agent that helps users plan and execute software projects.

Your responsibilities:
1. Understand user requirements through conversation
2. Clarify ambiguous or incomplete requests
3. Break down complex requests into manageable tasks
4. Delegate coding work to the TaskExecutor agent
5. Track progress and report back to the user
```

### 3. TaskExecutorAgent (task_executor_agent.py)

Coding agent that executes specific tasks.

**Context Strategy:** `hierarchical`
- Uses ContextManager for Goal → Task → Subtask → Action hierarchy
- Keeps last N message exchanges (N = context.history_keep from config)
- Focused on current subtask

**Tools:**
- `write_file`: Write content to files
- `read_file`: Read file contents
- `list_dir`: List directory contents
- `run_cmd`: Execute shell commands (whitelisted: python, pytest, ruff, pip)
- `mark_subtask_complete`: Mark current subtask done
- `decompose_task`: Break task into subtasks

**System Prompt:**
- Uses prompt from agent_config.yaml (llm.system_prompt)
- Focused on executing subtasks and calling mark_subtask_complete

### 4. AgentRegistry (agent_registry.py)

Manages agent instances and delegation relationships.

**Configuration:** `agents.yaml`
```yaml
agents:
  orchestrator:
    class: OrchestratorAgent
    can_delegate_to: [task_executor]

  task_executor:
    class: TaskExecutorAgent
    can_delegate_to: []
```

**Methods:**
- `get_agent(name)`: Get or create agent instance
- `can_delegate(from, to)`: Check delegation permissions
- `delegate_task(from, to, task, context)`: Delegate task between agents
- `get_agent_status(name)`: Get agent status
- `list_agents()`: List available agents
- `get_delegation_graph()`: Get delegation relationships

### 5. Status Displays

**OrchestratorStatusDisplay (orchestrator_status.py):**
- Shows conversation summary (message count, token usage)
- Token usage bar with compaction threshold
- Delegated tasks status
- Recent activity (last 4 messages)
- Performance metrics

**StatusDisplay (status_display.py):**
- Used by TaskExecutor (existing hierarchical display)
- Shows Goal → Task → Subtask tree
- Progress bars for tasks/subtasks
- Turn counter with forced decomposition warnings
- Performance stats

## Usage

### Running the Orchestrator

```bash
# Interactive mode
python orchestrator_main.py

# With initial message
python orchestrator_main.py "Create a Python calculator package"
```

### Example Conversation Flow

```
User: "I need a web calculator"
  ↓
Orchestrator: Clarifies requirements (HTML/CSS/JS? features?)
  ↓
User: "Simple HTML calculator with +, -, *, /"
  ↓
Orchestrator: Creates task plan
  ↓
Orchestrator: Delegates to TaskExecutor
  ↓
TaskExecutor: Executes task (creates files, runs tests)
  ↓
Orchestrator: Reports completion to user
```

### Delegation Flow

1. User sends message to Orchestrator
2. Orchestrator decides to delegate work
3. Orchestrator calls `delegate_to_executor` tool
4. AgentRegistry routes delegation to TaskExecutor
5. TaskExecutor receives goal and begins execution
6. Orchestrator can check status via `get_executor_status`

## Context Management Strategies

### Hierarchical (TaskExecutor)

**Goal:**
- Stay focused on current subtask
- Avoid accumulating too much history

**Strategy:**
- Keep last N message exchanges (default: 12)
- Each "exchange" = assistant message + tool response
- Older messages dropped automatically

**Benefits:**
- Prevents context bloat
- Forces agent to stay focused
- Matches subtask round limits

### Append Until Full (Orchestrator)

**Goal:**
- Maintain full conversation history
- Compress only when necessary

**Strategy:**
- Append all messages until 80% of token limit
- When threshold reached, compact:
  - Keep recent 20 messages intact
  - Summarize older messages

**Benefits:**
- Preserves conversation context
- User can reference earlier discussion
- Compaction is transparent

## Configuration

**agent_config.yaml:**
- `llm`: Model, temperature, system prompt (used by both agents)
- `context.history_keep`: Message history for TaskExecutor (default: 12)
- `context.max_tokens`: Token limit for Orchestrator compaction (default: 8000)
- `rounds.max_per_subtask`: Rounds before TaskExecutor escalation (default: 12)

**agents.yaml:**
- Defines available agents
- Specifies delegation permissions
- Extensible: add new agent types easily

## Adding New Agent Types

To create a new agent:

1. **Create agent class** inheriting from `BaseAgent`:
   ```python
   from base_agent import BaseAgent

   class MyAgent(BaseAgent):
       def get_tools(self):
           return [...]

       def get_system_prompt(self):
           return "..."

       def get_context_strategy(self):
           return "hierarchical"  # or "append_until_full"

       def build_context(self):
           # Implement context building logic
   ```

2. **Register in agents.yaml**:
   ```yaml
   agents:
     my_agent:
       class: MyAgent
       can_delegate_to: [other_agent]
   ```

3. **Update AgentRegistry** to instantiate your agent class

4. **Test**: Registry will automatically handle delegation and lifecycle

## Design Principles

1. **Simple inheritance**: Only override what's different
2. **Config-driven**: Agent relationships in YAML, not hardcoded
3. **Self-modifiable**: Agents can create new agent types by writing config files
4. **Clear separation**: Orchestrator plans, TaskExecutor executes
5. **Backward compatible**: Existing agent.py still works for direct task execution

## Future Extensions

Possible new agent types:

- **AnalyzerAgent**: Analyzes code quality, suggests improvements
- **TestAgent**: Focused on writing and running tests
- **DocumentationAgent**: Generates documentation from code
- **DebugAgent**: Specialized in debugging and error analysis
- **WebAgent**: Specialized in web development (HTML/CSS/JS)

Each would inherit from BaseAgent and be configured in agents.yaml.
