# Jetbox Architecture

## Core Philosophy

**Composition + Configuration + Crash-Resilience**

1. **Composition Over Inheritance** - BaseAgent + YAML-loaded behaviors (no subclassing)
2. **Configuration-Driven** - All customization in YAML, not code
3. **Crash-Resilient** - Plaintext state, append-only logs, resumable
4. **Single Responsibility** - Each behavior/file does one thing

## Naming Conventions

**Files**: `{noun}_{suffix}.py`
- `_tools.py` - Tool-providing behaviors
- `_behavior.py` - Non-tool behaviors
- `_utils.py` - Utility functions
- `_manager.py` - State management
- Core files have no suffix (agent.py, base_agent.py)

**Classes**:
- Behaviors: `{Feature}Behavior` (WriteFileToolsBehavior)
- Configs: `{Section}Config` (LLMConfig)
- Managers: `{Feature}Manager` (WorkspaceManager)

**Methods**:
- Lifecycle: `on_{event}()` (on_goal_start, on_round_start, on_goal_complete)
- Getters: `get_{property}()` (get_tools, get_context)
- Tools: `tool_{name}()` (tool_write_file)
- Private: `_{method}()` (underscore prefix)

## Architecture

### BaseAgent (`base_agent.py`)
Universal orchestrator for all agents:
- Loads behaviors from YAML config
- Manages lifecycle (setup → rounds → completion)
- Calls LLM with context from behaviors
- Dispatches tool calls to behaviors
- Does NOT implement tools or context strategies (behaviors do this)

### Behaviors (`behaviors/*.py`)
Independent, composable modules:
- **Tool Behaviors**: WriteFileToolsBehavior, CommandToolsBehavior, ServerToolsBehavior
- **Context Behaviors**: CompactWhenNearFullBehavior
- **Utility Behaviors**: LoopDetectionBehavior, WorkspaceTaskNotesBehavior

**Key Properties**:
- Zero dependencies between behaviors
- Loaded via YAML, not imports
- Communicate via events: `on_goal_start()`, `on_initial_context()`, `on_round_start()`, `on_llm_response()`, `on_tool_call()`, `on_round_end()`, `on_goal_complete()`

### Configuration System (`agent_config.py`)
Three-level hierarchy: **code defaults → config files → agent config**

Files:
- `config/llm_config.yaml` - Model, temperature, timeouts
- `config/agent_runtime.yaml` - Rounds, escalation, hierarchy
- `config/agents/{agent}.yaml` - Behaviors, system prompt
- `config/behavior_defaults.yaml` - Default behavior params

Functions:
- `load_llm_config()` - Load LLM settings
- `load_runtime_config()` - Load runtime settings
- `load_agent_config(name)` - Load agent-specific config
- `AgentConfig.load()` - Merge all configs

### LLM Communication (`llm_utils.py`)
Interface with Ollama:
- `chat_with_inactivity_timeout()` - Streaming with timeout protection
- Preserves thinking tokens from thinking-capable models
- Circuit breaker for consecutive failures
- Auto-restart Ollama when threshold exceeded

### Workspace Isolation (`workspace_manager.py`)
Creates isolated directories per goal:
- Auto-creates: `.agent_workspaces/{goal-slug}/`
- All file operations workspace-relative
- Tracks created files
- Prevents context distraction

### CLI Entry (`agent.py`)
Parse arguments and launch agents:
- Clears OLLAMA_MODEL env var (forces config file usage)
- Loads team configurations
- Creates agents with behaviors
- Handles keyboard interrupts

## Lifecycle Flow

```
1. Init: Load configs → Create workspace → Load behaviors
2. Goal: Set goal → on_goal_start() → on_initial_context()
3. Loop: [on_round_start() → LLM call → on_tool_call() × N → on_round_end()] × rounds
4. Complete: mark_complete()/mark_failed() → on_goal_complete()
```

## Tool Dispatch

```
LLM returns: {"name": "write_file", "arguments": {...}}
    ↓
BaseAgent looks up: tool_registry["write_file"] → WriteFileToolsBehavior
    ↓
Behavior validates + executes → Returns result
    ↓
Result added to context for next LLM call
```

## Configuration Cascade

```
CODE DEFAULTS
    ↓ (merged by)
agent_runtime.yaml
    ↓ (merged by)
llm_config.yaml
    ↓ (merged by)
agents/{agent}.yaml
    ↓ (fallbacks from)
behavior_defaults.yaml
    ↓
Final AgentConfig
```

## Adding a Behavior

1. Create `behaviors/my_feature_behavior.py`:
```python
from behaviors.base import AgentBehavior

class MyFeatureBehavior(AgentBehavior):
    def get_tools(self):
        return [...]  # Tool definitions

    def dispatch_tool_call(self, tool_name, arguments):
        # Handle tool calls
        pass

    def on_round_start(self, agent, round_number, context):
        # Inject context each round
        return context
```

2. Register in `behaviors/__init__.py`

3. Add to agent config:
```yaml
behaviors:
  - type: MyFeatureBehavior
    params:
      my_setting: value
```

## Directory Structure

```
jetbox/
├── agent.py                      # CLI entry point
├── base_agent.py                 # Universal agent base
├── agent_config.py               # Config loading
├── llm_utils.py                  # Ollama communication
├── workspace_manager.py          # Workspace isolation
├── jetbox_notes.py               # Persistent context summaries
├── jetbox_commands_whitelist     # Allowed bash commands
│
├── config/                       # Configuration files
│   ├── llm_config.yaml          # Model, temp, timeouts
│   ├── agent_runtime.yaml       # Rounds, escalation
│   ├── behavior_defaults.yaml   # Behavior params
│   ├── agents/                  # Per-agent configs
│   │   ├── task_executor.yaml
│   │   └── orchestrator.yaml
│   └── teams/                   # Multi-agent teams
│       └── default.yaml
│
├── behaviors/                    # Composable behaviors
│   ├── base.py                  # Base behavior class
│   ├── write_file_tools.py      # File operations
│   ├── read_file_tools.py
│   ├── directory_tools.py
│   ├── command_tools.py         # Bash execution
│   ├── server_tools.py          # Server management
│   ├── compact_when_near_full.py # Context compaction
│   ├── loop_detection.py        # Infinite loop detection
│   ├── workspace_task_notes.py  # Persistent summaries
│   ├── status_display.py        # Progress tracking
│   ├── context_inspector.py     # Context debugging
│   ├── delegation.py            # Multi-agent delegation
│   └── ...
│
├── tests/                        # Test suite
│   ├── test_mathx.py            # Sample package tests
│   ├── evaluation_scripts/      # Benchmark scripts
│   │   ├── run_l5_l7_x5_eval.py
│   │   └── rerun_l5_l7_eval.py
│   ├── evaluation_suite.py      # Task definitions
│   ├── evaluation_suite_extended.py # Extended tasks
│   └── flexible_validation.py   # Task validators
│
├── evaluation_results/          # Benchmark logs & analysis
│   ├── POST_FIX_SUMMARY.md
│   ├── context_analysis_*/      # Context inspection dumps
│   └── *.log                    # Eval run logs
│
├── debug_scripts/               # Dev debugging tools
│   ├── debug_agent_execution.py
│   └── quick_l1_test.py
│
├── tools/                       # Utility scripts
│   └── report_generator.py     # Eval report generator
│
├── docs/                        # Documentation
│   ├── analysis/                # Design analysis docs
│   └── jetbox_notes/            # Notes system docs
│
├── archive/                     # Deprecated code (reference only)
│
└── .agent_workspaces/           # Isolated workspaces
    └── {goal-slug}/
        ├── .agent_context/      # State & history
        │   ├── state.json
        │   ├── history.jsonl
        │   └── stats.json
        ├── workspace_task_notes.md # Persistent summaries
        └── {generated files}
```

## Key Files Reference

**Core**:
- `base_agent.py` - How agents work
- `agent_config.py` - How configs load
- `behaviors/base.py` - How behaviors work
- `config/llm_config.yaml` - Current model & settings

**Behaviors**:
- `behaviors/__init__.py` - Behavior registry
- `behaviors/write_file_tools.py` - File I/O
- `behaviors/command_tools.py` - Bash execution
- `behaviors/compact_when_near_full.py` - Context management

**Testing**:
- `tests/evaluation_suite_extended.py` - 38 tasks across 7 levels
- `tests/evaluation_scripts/` - Benchmark runners
- `evaluation_results/` - Historical results

## Common Tasks

**Change model**: Edit `config/llm_config.yaml` → `model: "qwen3-coder:30b"`

**Add behavior**: Create in `behaviors/` → Register in `__init__.py` → Add to agent YAML

**Adjust rounds**: Edit `config/agent_runtime.yaml` → `rounds.max_per_subtask`

**Debug agent**: Check `.agent_context/state.json` and `agent_ledger.log`

**Run tests**: `pytest -q` or `python tests/evaluation_scripts/run_l5_l7_x5_eval.py`

## Design Patterns

**Behavior Pattern**: Behaviors are strategies loaded from YAML, not hardcoded

**Event-Driven**: BaseAgent emits events, behaviors listen independently

**Registry**: Maps tool names → behaviors for dynamic dispatch

**Crash-Resilient**: JSON state + append-only logs = always resumable

## Anti-Patterns (Don't Do This)

❌ Import behaviors from other behaviors (use events)

❌ Hardcode agent logic in BaseAgent (use behaviors)

❌ Create agent subclasses for small changes (use YAML config)

❌ Bypass config system (use YAML, not env vars)

❌ Store state in memory only (persist to JSON)

## Summary

Jetbox = **BaseAgent** (orchestrator) + **Behaviors** (composable modules) + **YAML Config** (customization)

Everything is:
- Modular (behaviors are independent)
- Observable (plaintext state, append-only logs)
- Maintainable (one file = one purpose)
- Crash-resilient (always resumable from logs)
