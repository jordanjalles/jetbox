# Jetbox - Local-First Coding Agent Framework

A composable agent framework for autonomous code generation powered by local LLMs via Ollama. Built for Windows but runs anywhere.

**JetBox — a local agent framework built for speed, autonomy, and total on-device control. Makes your fan scream like a jet at takeoff.**

## Quick Start

```bash
# Install dependencies
pip install -e .

# Pull the default model (qwen3:8b - 5.2GB)
ollama pull qwen3:8b

# Run a simple task with solo agent (TaskExecutor)
python agent.py --team solo "Create a calculator package with add/subtract/multiply/divide"

# Run a complex task with default team (Orchestrator → TaskExecutor)
python agent.py --team default "Create a Flask REST API for managing books with SQLite storage"

# Chat mode (interactive)
python agent.py --team chatbot
```

## Core Concepts

### 1. Agent Types by Task Size

Jetbox provides different agent modes optimized for different task complexities:

| Agent Mode | Best For | Max Rounds | Workspace | Example |
|------------|----------|------------|-----------|---------|
| **Solo Agent** | Simple packages, utilities | 50 | Single isolated dir | "Create string utils package" |
| **Orchestrator + TaskExecutor** | Full applications, APIs | 50 (orchestrator) + 50 (executor) | Project structure | "Create Flask API with auth" |
| **Orchestrator + Multi-Executor** | Complex systems | 50 + N×50 | Multi-module projects | "Create microservices system" |

**When to use which:**

- **Solo Agent** (`--team solo`): Simple tasks
  - Single package/module
  - < 10 files
  - Clear, focused goal
  - Examples: validators package, data structures, file utils

- **Default Team** (`--team default`): Moderate to complex tasks
  - Full applications with multiple modules
  - 10+ files
  - Architecture planning helpful
  - Examples: REST APIs, web apps with auth, CLI tools

- **Chatbot** (`--team chatbot`): Interactive mode
  - Requirements gathering
  - Question answering
  - Exploratory conversations
  - Can transition to task execution via `set_goal`

### 2. Behavior-Based Composition

All agent capabilities are provided by **composable behaviors**. Mix and match behaviors to create custom agents:

```yaml
# task_executor_config.yaml
behaviors:
  # Core execution
  - type: SubAgentModeBehavior
    params:
      enable_completion_nudging: true
      min_rounds_before_nudge: 3

  # Context management
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 96000

  # Tools
  - type: FileToolsBehavior
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "pip"]

  # Utilities
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5
  - type: WorkspaceTaskNotesBehavior
```

**Key behaviors:**

- **SubAgentModeBehavior**: Provides `mark_complete`/`mark_failed` tools, completion nudging
- **CompactWhenNearFullBehavior**: Automatic context management when nearing token limit
- **FileToolsBehavior**: `write_file`, `read_file`, `list_dir`
- **CommandToolsBehavior**: `run_bash` with command whitelist
- **LoopDetectionBehavior**: Detects repeated actions, injects recovery prompts
- **WorkspaceTaskNotesBehavior**: Persistent summaries across runs
- **DelegationBehavior**: Task delegation (orchestrator only)
- **ArchitectToolsBehavior**: Architecture design artifacts (architect only)

### 3. Configuration-Driven Architecture

Configure agent behavior without code changes:

```yaml
# agent_config.yaml
llm:
  model: "qwen3:8b"  # Default model: fast, capable, 128K context
  temperature: 0.2
  timeout:
    inactivity_timeout: 30  # Max seconds without LLM activity
    max_total_time: null    # No limit on total call time

behavior_defaults:
  CompactWhenNearFullBehavior:
    max_tokens: 96000  # 75% of 128K context window

rounds:
  max_per_subtask: 50
  max_global: 256
```

## Usage Patterns

### Pattern 1: Simple Task (Solo Agent)

```bash
python agent.py --team solo "Create a validators package with email, url, phone validation"
```

**What happens:**
1. TaskExecutor starts with goal
2. Creates isolated workspace (`.agent_workspaces/create-a-validators-package-with-email-url`)
3. Executes for up to 50 rounds
4. Writes files, runs tests, calls `mark_complete` when done
5. Returns summary

**Use when:** Single-purpose packages, utilities, simple scripts

### Pattern 2: Complex Task (Default Team - Orchestrator)

```bash
python agent.py --team default "Create a Flask REST API for managing books with CRUD endpoints and SQLite storage"
```

**What happens:**
1. Orchestrator analyzes goal
2. Delegates to TaskExecutor with detailed requirements
3. TaskExecutor creates files, runs tests, verifies linting
4. TaskExecutor marks subtask complete
5. Orchestrator marks overall goal complete

**Use when:** Full applications, APIs, multi-module systems

### Pattern 3: Resume Interrupted Work

```bash
# Original run (interrupted)
python orchestrator_main.py "Create calculator with scientific functions"

# Resume from same workspace
python orchestrator_main.py --workspace .agent_workspaces/create-calculator-with-scientific "Continue work"
```

**What happens:**
- Loads `workspace_task_notes.md` with previous progress
- Agent sees what was already done
- Continues from where it left off
- No duplicate work

### Pattern 4: Custom Agent Configuration

```python
from task_executor_agent import TaskExecutorAgent

agent = TaskExecutorAgent(
    workspace=".",
    goal="Create custom package",
    use_behaviors=True,
    config_file="my_custom_config.yaml"
)

result = agent.execute()
```

**Use when:** Need custom behavior composition, specialized workflows

## Extending the Framework

### Adding a New Behavior

```python
# behaviors/my_behavior.py
from typing import Any
from behaviors.base import AgentBehavior

class MyCustomBehavior(AgentBehavior):
    def get_name(self) -> str:
        return "my_custom"

    def get_tools(self) -> list[dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does something custom",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg": {"type": "string"}
                    }
                }
            }
        }]

    def dispatch_tool(self, tool_name: str, args: dict[str, Any], **kwargs):
        if tool_name == "my_tool":
            # Implement tool logic
            return {"result": f"Processed: {args['arg']}"}
        return super().dispatch_tool(tool_name, args, **kwargs)

    def enhance_context(self, context: list[dict[str, Any]], **kwargs):
        # Modify context before LLM call
        return context

    def on_round_end(self, round_number: int, **kwargs):
        # Hook into agent lifecycle
        pass
```

**Then add to config:**

```yaml
# my_agent_config.yaml
behaviors:
  - type: MyCustomBehavior
    params:
      custom_param: value
```

### Creating a Custom Agent

```python
# my_agent.py
from base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, workspace: str, goal: str, **kwargs):
        super().__init__(
            workspace=workspace,
            use_behaviors=True,
            config_file="my_agent_config.yaml"
        )
        self.goal = goal

    def execute(self):
        # Custom execution logic
        self.fire_event("goal_start", goal=self.goal)

        for round_num in range(1, self.max_rounds + 1):
            context = self.build_context()
            response = self.call_llm(context)
            results = self.execute_tool_calls(response.tool_calls)

            self.fire_event("round_end", round_number=round_num)

            if self.check_completion(results):
                break

        self.fire_event("goal_complete", success=True)
```

## Architecture Deep Dive

### Agent Hierarchy

```
Orchestrator (orchestrator_agent.py)
├─ Behaviors:
│  ├─ DelegationBehavior (delegate_task tool)
│  ├─ HierarchicalContextBehavior
│  └─ LoopDetectionBehavior
│
├─ Delegates to Architect (architect_agent.py)
│  └─ Behaviors:
│     ├─ ArchitectToolsBehavior (write_task_list, write_architecture_doc)
│     ├─ ArchitectContextBehavior
│     └─ SubAgentModeBehavior
│
└─ Delegates to TaskExecutor (task_executor_agent.py)
   └─ Behaviors:
      ├─ SubAgentModeBehavior (mark_complete, completion nudging)
      ├─ FileToolsBehavior
      ├─ CommandToolsBehavior
      ├─ ServerToolsBehavior
      ├─ WorkspaceTaskNotesBehavior
      └─ LoopDetectionBehavior
```

### Context Management Strategies

Different behaviors provide different context management:

| Behavior | Strategy | When to Use |
|----------|----------|-------------|
| **CompactWhenNearFullBehavior** | Append until 75% full, then LLM-compact | General purpose, moderate tasks |
| **HierarchicalContextBehavior** | Keep last N exchanges, clear on subtask | Orchestrator with task switching |
| **SubAgentContextBehavior** | Append all messages, 128K limit | Delegated execution, needs full history |
| **ArchitectContextBehavior** | Optimized for long design discussions | Architecture planning phase |

### Workspace Isolation

Each goal gets an isolated workspace:

```
.agent_workspaces/
├─ create-calculator-package/
│  ├─ calculator/
│  │  ├─ __init__.py
│  │  └─ operations.py
│  ├─ tests/
│  │  └─ test_calculator.py
│  └─ workspace_task_notes.md  # Persistent context
│
└─ create-flask-api-for-books/
   ├─ app.py
   ├─ models.py
   ├─ routes.py
   ├─ tests/
   └─ workspace_task_notes.md
```

**Benefits:**
- No context pollution from other projects
- Clean slate for each goal
- Easy to resume work
- Human-inspectable output

## Performance Characteristics

Based on evaluation across 40 tasks (L3-L6 complexity):

| Metric | qwen3:8b (Default) | Notes |
|--------|-------------------|-------|
| **Success Rate** | 50% | After completion nudging: 60-65% expected |
| **Avg Time/Task** | 77.7s | 1.8x faster than gpt-oss:20b |
| **Context Window** | 128K tokens | Sufficient for complex workflows |
| **Model Size** | 5.2GB | 60% smaller than gpt-oss:20b |
| **Per-round Speed** | 28.5s | Slower per-round but fewer rounds needed |

**Task complexity vs success rate:**
- L3 (simple packages): 20% → 40-50% (after fixes)
- L4 (moderate packages): 40% → 60-70%
- L5 (REST APIs): 60% → 70-80%
- L6 (Full apps): 80% → 90-95%
- L7 (Complex systems): 100% (1/1)

**Key finding:** "One-shot strategy" - qwen3:8b completes tasks in 1-2 rounds vs 6-12 for larger models, offsetting slower per-round inference.

## Command Reference

### Main Entry Point

```bash
# Run with default settings
python orchestrator_main.py "Your goal here"

# Resume from workspace
python orchestrator_main.py --workspace .agent_workspaces/previous-goal "Continue"

# Override model
OLLAMA_MODEL=qwen3:14b python orchestrator_main.py "Your goal"
```

### Testing

```bash
# Run unit tests
pytest -q

# Run agent on test goal
python orchestrator_main.py "Create a test package"

# Run evaluations
python tests/evaluation_scripts/run_model_comparison_eval.py
```

### Linting

```bash
# Check code
ruff check .

# Auto-fix
ruff check --fix .
```

## Configuration Files

| File | Purpose |
|------|---------|
| `agent_config.yaml` | Global defaults (model, timeouts, limits) |
| `orchestrator_config.yaml` | Orchestrator-specific behaviors and settings |
| `architect_config.yaml` | Architect-specific behaviors and settings |
| `task_executor_config.yaml` | TaskExecutor-specific behaviors and settings |
| `agents.yaml` | Agent registry and delegation relationships |

## Troubleshooting

### Agent hits max rounds without completing

**Symptom:** Task exits with "Max rounds (12) reached" but work looks complete

**Fix:** Completion nudging should catch this. Check config:

```yaml
# task_executor_config.yaml
behaviors:
  - type: SubAgentModeBehavior
    params:
      enable_completion_nudging: true  # Should be true
      min_rounds_before_nudge: 3
```

### LLM call hangs

**Symptom:** Agent starts round but never responds

**Fix:** Check timeout settings:

```yaml
# agent_config.yaml
llm:
  timeout:
    inactivity_timeout: 30  # Max seconds without activity
    max_total_time: 180     # Max seconds per call (recommended)
```

### Context window exceeded

**Symptom:** Error about context length

**Fix:** Adjust max_tokens for compaction:

```yaml
# agent_config.yaml
behavior_defaults:
  CompactWhenNearFullBehavior:
    max_tokens: 96000  # 75% of 128K (leave headroom)
```

### Command not allowed

**Symptom:** "Command not in whitelist" error

**Fix:** Add command to whitelist:

```yaml
# task_executor_config.yaml
behaviors:
  - type: CommandToolsBehavior
    params:
      whitelist: ["python", "pytest", "ruff", "pip", "npm", "git"]
```

Or edit `jetbox_commands_whitelist` file in root.

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete architecture reference for AI assistants
- **[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)** - Detailed architecture documentation
- **[BEHAVIORS_DOCUMENTATION.md](BEHAVIORS_DOCUMENTATION.md)** - All behaviors reference
- **[CONFIG_SYSTEM.md](CONFIG_SYSTEM.md)** - Configuration system guide
- **[STATUS_DISPLAY.md](STATUS_DISPLAY.md)** - Progress visualization system
- **[docs/implementation_notes/](docs/implementation_notes/)** - Implementation details and decisions
- **[evaluation_results/](evaluation_results/)** - Model evaluation results and analysis

## Design Philosophy

1. **Local-first**: All processing happens locally via Ollama. No API keys, no cloud services.

2. **Crash-resilient**: Agent expects to be interrupted. All state persisted to plaintext files. Fast rehydration from logs.

3. **Human-inspectable**: No databases. Everything is files you can read with `cat` or open in an editor.

4. **Composable**: Behaviors are self-contained, single-responsibility modules. Mix and match freely.

5. **Config-driven**: Change agent behavior via YAML, not code. Iterate without rewriting logic.

6. **Verify-first**: Always probe actual state (file existence, test results) before deciding what to do. No hallucination.

7. **Backward-chaining**: Plan from desired end state (tests pass) back to current state.

## License

MIT

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Credits

Built with:
- [Ollama](https://ollama.ai/) - Local LLM inference
- [qwen3:8b](https://ollama.ai/library/qwen3) - Default model (Alibaba Cloud)
- Python 3.11+
