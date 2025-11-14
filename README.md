# Jetbox - Local-First Coding Agent Framework

A composable agent framework for autonomous code generation powered by local LLMs via Ollama. Built for speed, autonomy, and total on-device control.

**JetBox — a local agent framework that makes your fan scream like a jet at takeoff. 🚀**

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Pull the default model (~9GB)
ollama pull qwen3:14b

# 3. Run your first task
python agent.py --team solo "Create a calculator package with add/subtract/multiply/divide"

# 4. Or start interactive chat
python agent.py --team chatbot
```

That's it! The agent will create an isolated workspace, write code, run tests, and mark the task complete.

## Why Jetbox?

### 🏗️ Composable Architecture
Build custom agents by mixing and matching behaviors in YAML. No code changes needed.

### 🔧 Behavior-Driven Design
Every capability is a self-contained behavior. Add, remove, or create behaviors without touching core code.

### 📝 Configuration Over Code
Change agent behavior, tools, prompts, and strategies via YAML files. Iterate fast.

### 🌐 Local-First
All processing via Ollama. No API keys, no cloud services, no data leaving your machine.

### 🔄 Crash-Resilient
Expects interruption. All state persisted to plaintext files. Resume from where you left off.

### 👁️ Human-Inspectable
No databases. Everything is files you can read with `cat` or open in an editor.

## Core Concepts

### 1. Teams: Different Configurations for Different Tasks

Jetbox provides pre-configured teams optimized for different complexities:

```bash
# Solo: Simple tasks, single agent
python agent.py --team solo "Create string utils package"

# Default: Complex tasks, multi-agent coordination
python agent.py --team default "Create Flask API with auth"

# Chatbot: Interactive requirements gathering
python agent.py --team chatbot
```

| Team | Agents | Best For |
|------|--------|----------|
| **solo** | TaskExecutor only | Simple packages, utilities, < 10 files |
| **default** | Orchestrator → Architect + TaskExecutor | Full applications, APIs, 10+ files |
| **chatbot** | Interactive mode | Requirements gathering, Q&A |

### 2. Behaviors: Self-Contained Capability Modules

Every agent capability comes from composable behaviors:

```yaml
# config/agents/task_executor.yaml
behaviors:
  # Execution control
  - type: ExecutionModeBehavior
    params: {}

  # Context management
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 131072
      compact_threshold: 0.75

  # Tools
  - type: WriteFileToolsBehavior
  - type: ReadFileToolsBehavior
  - type: DirectoryToolsBehavior
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "pip"]

  # Utilities
  - type: LoopDetectionBehavior
  - type: WorkspaceTaskNotesBehavior
```

**Each behavior is independent:**
- Provides tools (e.g., `write_file`, `run_bash`)
- Modifies context (e.g., inject completion nudges)
- Reacts to events (e.g., on round end, on goal complete)
- **Never imports other behaviors** (zero coupling)

### 3. Configuration-Driven: YAML All the Way

Change everything via config files:

```
config/
├── llm_config.yaml           # Model, temperature, timeouts
├── agent_runtime.yaml        # Round limits, escalation strategy
├── behavior_defaults.yaml    # Default behavior parameters
├── agents/
│   ├── orchestrator.yaml     # Orchestrator config + system prompt
│   ├── architect.yaml        # Architect config + system prompt
│   └── task_executor.yaml    # TaskExecutor config + system prompt
└── teams/
    ├── default.yaml          # Multi-agent team
    ├── solo.yaml             # Single agent
    └── chatbot.yaml          # Interactive mode
```

**No code changes to:**
- Switch models
- Adjust timeouts
- Add/remove tools
- Change system prompts
- Create new agent types

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed
- 8GB+ RAM (16GB recommended)
- 10GB+ disk space for models

### Install

```bash
# Clone repository
git clone <repository-url>
cd jetbox

# Install in development mode
pip install -e .

# Pull default model
ollama pull qwen3:14b

# Verify installation
python agent.py --team solo "Create a hello.py file"
```

### Optional: Pull Alternative Models

```bash
# Smaller model (resource-constrained systems)
ollama pull qwen3:8b

# Alternative general-purpose model
ollama pull gpt-oss:20b
```

## Usage

### Simple Tasks (Solo Agent)

For simple packages, utilities, or scripts:

```bash
python agent.py --team solo "Create a validators package with email and URL validation"
```

**What happens:**
1. Creates isolated workspace: `.agent_workspaces/create-a-validators-package/`
2. Writes files, runs tests, checks linting
3. Calls `mark_complete()` when done
4. Saves summary to `workspace_task_notes.md`

**Output:**
```
.agent_workspaces/create-a-validators-package/
├── validators/
│   ├── __init__.py
│   ├── email.py
│   └── url.py
├── tests/
│   ├── test_email.py
│   └── test_url.py
├── README.md
└── workspace_task_notes.md  # Persistent context
```

### Complex Tasks (Default Team)

For full applications requiring architecture planning:

```bash
python agent.py --team default "Create a Flask REST API for managing books with CRUD endpoints and SQLite storage"
```

**What happens:**
1. Orchestrator analyzes goal
2. Delegates to Architect for design
   - Creates `architecture.md`
   - Creates `task-breakdown.json`
3. Delegates to TaskExecutor for implementation
   - Writes application files
   - Runs tests and linting
4. Returns results to Orchestrator
5. Marks goal complete

**Output:**
```
.agent_workspaces/create-a-flask-rest-api/
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
│   └── database.py
├── tests/
│   └── test_api.py
├── architecture/
│   ├── architecture.md
│   └── task-breakdown.json
├── requirements.txt
└── workspace_task_notes.md
```

### Interactive Mode

For requirements gathering or Q&A:

```bash
python agent.py --team chatbot
```

Then transition to execution:
```
You: I need a web scraper
Agent: What URL do you want to scrape? What data are you looking for?
You: set_goal Create a web scraper for news articles from example.com
Agent: [Switches to execution mode, creates scraper]
```

### Resume Interrupted Work

All work is resumable:

```bash
# Original run (interrupted by timeout or error)
python agent.py "Create calculator with scientific functions"

# Resume from same workspace
python agent.py --workspace .agent_workspaces/create-calculator "Continue work"
```

The agent loads `workspace_task_notes.md` and continues from where it left off.

### Override Model

```bash
# Temporary override via environment variable
OLLAMA_MODEL=gpt-oss:20b python agent.py "Your goal"

# Permanent override: edit config/llm_config.yaml
# model: "gpt-oss:20b"
```

## Extending Jetbox

### Creating a Custom Behavior

Behaviors are self-contained modules that provide tools, modify context, or react to events.

**Example: Add a custom notification tool**

```python
# behaviors/notification_behavior.py
from typing import Any
from behaviors.base import AgentBehavior

class NotificationBehavior(AgentBehavior):
    """Sends notifications when tasks complete."""

    def get_name(self) -> str:
        return "notification"

    def get_sequence_number(self) -> int:
        """Controls execution order (higher = runs later)."""
        return 50

    def get_tools(self) -> list[dict[str, Any]]:
        """Define tools this behavior provides."""
        return [{
            "type": "function",
            "function": {
                "name": "send_notification",
                "description": "Send a notification message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Notification message"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "normal", "high"],
                            "description": "Notification priority"
                        }
                    },
                    "required": ["message"]
                }
            }
        }]

    def dispatch_tool(self, tool_name: str, args: dict[str, Any], **kwargs):
        """Handle tool calls."""
        if tool_name == "send_notification":
            message = args["message"]
            priority = args.get("priority", "normal")

            # Your notification logic here
            print(f"[{priority.upper()}] {message}")

            return {"success": True, "sent": message}

        return super().dispatch_tool(tool_name, args, **kwargs)

    def on_goal_complete(self, agent: Any, success: bool, summary: str):
        """React to goal completion event."""
        if success:
            print(f"✅ Task completed: {summary}")
```

**Register in config:**

```yaml
# config/agents/my_agent.yaml
behaviors:
  - type: NotificationBehavior
    params: {}
```

**Behavior lifecycle events:**

```python
def on_goal_start(self, agent, goal):
    """Called once at goal start"""

def on_initial_context(self, agent, context):
    """Called once for initial setup"""

def on_round_start(self, agent, round_number, context):
    """Called before each LLM call - can modify context"""

def on_llm_response(self, agent, response):
    """Called after LLM responds"""

def on_tool_call(self, agent, tool_name, args, result):
    """Called after each tool execution"""

def on_round_end(self, agent, round_number):
    """Called after all tools in round"""

def on_goal_complete(self, agent, success, summary):
    """Called when goal finishes"""

def on_timeout(self, agent, elapsed):
    """Called when time budget exceeded"""

def on_custom_event(self, agent, event_name, **event_data):
    """Called for inter-behavior communication"""
```

See [BEHAVIORS_DOCUMENTATION.md](BEHAVIORS_DOCUMENTATION.md) for complete API reference.

### Creating a Custom Agent

Agents are just `BaseAgent` + YAML configuration. Create new agents without subclassing:

**1. Create agent config:**

```yaml
# config/agents/my_specialist.yaml
system_prompt: |
  You are a specialist for database migrations.

  Focus on:
  - Safe schema changes
  - Data preservation
  - Rollback strategies

behaviors:
  - type: WriteFileToolsBehavior
  - type: ReadFileToolsBehavior
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "psql", "mysql"]
  - type: ExecutionModeBehavior
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 131072
```

**2. Register in team:**

```yaml
# config/teams/migration_team.yaml
agents:
  orchestrator:
    class: BaseAgent
    config: orchestrator
    can_delegate_to: [my_specialist]

  my_specialist:
    class: BaseAgent
    config: my_specialist
    can_delegate_to: []
```

**3. Use it:**

```bash
python agent.py --team migration_team "Migrate users table to add email verification"
```

**That's it!** No Python code needed. The framework dynamically loads your config.

### Adding Custom Tools to Existing Behaviors

Extend behaviors via parameter passing:

```yaml
# config/agents/my_agent.yaml
behaviors:
  - type: CommandToolsBehavior
    params:
      whitelist:
        - python
        - pytest
        - ruff
        - docker      # Add docker
        - kubectl     # Add kubectl
```

Or create behavior variants:

```python
# behaviors/docker_tools_behavior.py
from behaviors.command_tools_behavior import CommandToolsBehavior

class DockerToolsBehavior(CommandToolsBehavior):
    """Variant with Docker commands pre-whitelisted."""

    def __init__(self):
        super().__init__(whitelist=[
            "docker", "docker-compose",
            "docker build", "docker run", "docker ps"
        ])
```

## Architecture Deep Dive

### Agent Lifecycle

```
┌─────────────────────────────────────────────┐
│           Agent Execution Flow              │
└─────────────────────────────────────────────┘

1. Goal Start
   └─> on_goal_start() event to all behaviors

2. Initial Context
   └─> on_initial_context() event

3. Main Loop (for each round):
   ├─> Build context (system prompt + messages)
   ├─> on_round_start() → behaviors can modify context
   ├─> Call LLM with context
   ├─> on_llm_response() → behaviors can parse response
   ├─> Execute tool calls
   │   └─> on_tool_call() for each tool
   └─> on_round_end()

4. Completion
   └─> on_goal_complete() or on_timeout()
```

### Team Architecture

```
Default Team (config/teams/default.yaml):

┌──────────────────────────────────────┐
│      Orchestrator (BaseAgent)        │
│                                      │
│  Behaviors:                          │
│  - DelegationBehavior (auto)         │
│  - CompactWhenNearFullBehavior       │
│  - TimeBoxBehavior                   │
│  - StatusDisplayBehavior             │
└─────────┬────────────────────────────┘
          │
          ├──► consult_architect()
          │    └─> Architect (BaseAgent)
          │        - ArchitectToolsBehavior
          │        - Creates architecture.md
          │        - Creates task-breakdown.json
          │
          └──► delegate_to_task_executor()
               └─> TaskExecutor (BaseAgent)
                   - WriteFileToolsBehavior
                   - ReadFileToolsBehavior
                   - CommandToolsBehavior
                   - LoopDetectionBehavior
                   - WorkspaceTaskNotesBehavior
```

**Key insight:** All agents are `BaseAgent` instances. No subclassing. Everything configured via YAML.

### Workspace Isolation

Each goal gets an isolated workspace:

```
.agent_workspaces/
├── create-calculator-package/
│   ├── calculator/
│   │   ├── __init__.py
│   │   └── operations.py
│   ├── tests/
│   │   └── test_calculator.py
│   ├── .agent_context/
│   │   ├── state.json          # Agent state
│   │   ├── history.jsonl       # Action log
│   │   └── stats.json          # Performance
│   └── workspace_task_notes.md # Persistent context
│
└── create-flask-api/
    ├── app/
    ├── tests/
    └── workspace_task_notes.md
```

**Benefits:**
- No cross-project pollution
- Clean slate per goal
- Resumable from any workspace
- Human-readable state

### Behavior Composition

Behaviors communicate through events, not direct imports:

```python
# BAD: Direct coupling
class BehaviorA:
    def __init__(self):
        self.behavior_b = BehaviorB()  # ❌ Tight coupling

# GOOD: Event-driven communication
class BehaviorA(AgentBehavior):
    def on_round_end(self, agent, round_number):
        # Broadcast event to all behaviors
        agent.event_system.fire_custom_event(
            "round_complete",
            round=round_number,
            status="success"
        )

class BehaviorB(AgentBehavior):
    def on_custom_event(self, agent, event_name, **event_data):
        if event_name == "round_complete":
            # React to event from BehaviorA
            print(f"Round {event_data['round']} complete!")
```

**Zero coupling:** Add, remove, or replace behaviors without affecting others.

## Configuration Reference

### Key Configuration Files

| File | Purpose | Example |
|------|---------|---------|
| `config/llm_config.yaml` | Model selection, timeouts | `model: "qwen3:14b"` |
| `config/agent_runtime.yaml` | Round limits, escalation | `max_per_subtask: 50` |
| `config/behavior_defaults.yaml` | Behavior parameters | `max_tokens: 131072` |
| `config/agents/{name}.yaml` | Agent configuration | Behaviors + system prompt |
| `config/teams/{name}.yaml` | Team composition | Agent relationships |

### Example: LLM Configuration

```yaml
# config/llm_config.yaml
model: "qwen3:14b"
temperature: 0.2
max_tokens: 131072  # 128K context window

timeout:
  inactivity_timeout: 30      # Max seconds without LLM activity
  max_call_time: 180          # Max seconds per LLM call
  max_consecutive_timeouts: 3 # Circuit breaker threshold
  auto_restart_ollama: true   # Auto-restart Ollama on failures
```

### Example: Agent Configuration

```yaml
# config/agents/task_executor.yaml
system_prompt: |
  You are a coding agent that implements software.

  Process:
  1. Read any existing files/architecture
  2. Write implementation files
  3. Run tests and linting
  4. Call mark_complete() when done

behaviors:
  - type: ExecutionModeBehavior
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 131072
  - type: WriteFileToolsBehavior
  - type: ReadFileToolsBehavior
  - type: DirectoryToolsBehavior
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "pip"]
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5
  - type: WorkspaceTaskNotesBehavior
```

See [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md) for complete reference.

## Command Reference

```bash
# Basic usage
python agent.py --team TEAM "Your goal here"

# Available teams
--team solo              # Single agent (TaskExecutor)
--team default           # Multi-agent (Orchestrator → Architect + TaskExecutor)
--team chatbot           # Interactive chat

# Resume from workspace
python agent.py --workspace PATH "Continue work"

# Add behaviors dynamically
python agent.py --ContextInspectorBehavior "Debug context"

# Override model
OLLAMA_MODEL=gpt-oss:20b python agent.py "Goal"

# Testing
pytest -q                           # Run unit tests
pytest tests/test_behaviors.py -q  # Run specific test

# Linting
ruff check .                        # Check code
ruff check --fix .                  # Auto-fix issues
```

## Troubleshooting

### LLM call hangs

**Symptom:** Agent starts round but never responds

**Fix:** Check model and timeouts in `config/llm_config.yaml`:

```yaml
model: "qwen3:14b"  # Ensure using a working model
timeout:
  inactivity_timeout: 30
  max_call_time: 180
```

### Agent doesn't mark task complete

**Symptom:** Task exits with "Max rounds reached" but work looks done

**Fix:** Ensure `ExecutionModeBehavior` is present:

```yaml
# config/agents/task_executor.yaml
behaviors:
  - type: ExecutionModeBehavior  # Required for completion detection
```

### Command not allowed

**Symptom:** "Command not in whitelist" error

**Fix:** Add to whitelist in agent config:

```yaml
behaviors:
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "npm", "git"]
```

### Context window exceeded

**Symptom:** Error about context length

**Fix:** Adjust compaction in `config/behavior_defaults.yaml`:

```yaml
CompactWhenNearFullBehavior:
  max_tokens: 131072      # Match model's context window
  compact_threshold: 0.75 # Compact at 75%
```

## Design Philosophy

1. **Composable** - Mix and match behaviors freely
2. **Configuration-driven** - YAML over code changes
3. **Zero coupling** - Behaviors don't import each other
4. **Event-driven** - Communication via lifecycle events
5. **Local-first** - No cloud, no API keys
6. **Crash-resilient** - State persisted to plaintext
7. **Human-inspectable** - No databases, just files

## Documentation

- **[BEHAVIORS_DOCUMENTATION.md](BEHAVIORS_DOCUMENTATION.md)** - All behaviors reference
- **[CONFIG_SYSTEM.md](CONFIG_SYSTEM.md)** - Configuration guide
- **[CLAUDE.md](CLAUDE.md)** - Complete architecture reference
- **[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)** - Detailed architecture
- **[docs/](docs/)** - Implementation notes

## License

MIT

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Credits

Built with:
- [Ollama](https://ollama.ai/) - Local LLM inference
- [qwen3:14b](https://ollama.ai/library/qwen3) - Default model (Alibaba Cloud)
- Python 3.11+
