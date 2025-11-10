# BaseAgent Refactor: Integration Strategy

## TL;DR

**BaseAgent remains the central orchestrator** that composes the extracted modules. All sub-modules are called FROM base_agent.py, not somewhere else. The public API stays unchanged.

---

## Integration Pattern: Composition

```python
# src/base_agent.py (simplified)
from src.agent_state import AgentState, StatePersistence
from src.tool_dispatch import ToolDispatcher
from src.behavior_loader import BehaviorLoader
from src.agent_events import EventSystem
from src.agent_lifecycle import AgentLifecycle

class BaseAgent:
    """
    Central orchestrator that composes specialized modules.

    Public API remains unchanged - all methods still callable from BaseAgent.
    Implementation delegated to composed modules.
    """

    def __init__(self, name: str, workspace: Path, config_file: str, ...):
        # === Core state ===
        self.name = name
        self.workspace = workspace
        self.goal = None

        # === Composed modules ===
        self.state_manager = StatePersistence(self.workspace)
        self.tool_dispatcher = ToolDispatcher(self)
        self.behavior_loader = BehaviorLoader(self)
        self.event_system = EventSystem(self)
        self.lifecycle = AgentLifecycle(self)

        # === Load initial state ===
        self.state = self.state_manager.load_state() or AgentState(name=name, ...)

        # === Load behaviors ===
        self.behaviors = []
        self.tool_registry = {}
        self.behavior_loader.load_from_config(config_file)

    # ==========================================
    # PUBLIC API - Methods users/behaviors call
    # ==========================================

    def set_goal(self, goal: str):
        """Public API - unchanged."""
        self.goal = goal
        self.event_system.trigger_goal_start(goal)

    def run(self, max_rounds: int = 12) -> dict:
        """Public API - delegates to lifecycle module."""
        return self.lifecycle.run(max_rounds)

    def dispatch_tool(self, tool_call: dict) -> dict:
        """Public API - delegates to tool dispatcher."""
        return self.tool_dispatcher.dispatch(tool_call)

    def add_message(self, message: dict):
        """Public API - updates state."""
        self.state.messages.append(message)

    def persist_state(self):
        """Public API - delegates to state manager."""
        self.state_manager.persist(self.state)

    def load_state(self):
        """Public API - delegates to state manager."""
        self.state = self.state_manager.load_state()

    def get_tools(self) -> list[dict]:
        """Public API - delegates to tool dispatcher."""
        return self.tool_dispatcher.get_all_tools()

    def build_context(self) -> list[dict]:
        """Public API - builds context with help from event system."""
        context = [{"role": "system", "content": self.get_system_prompt()}]
        context = self.event_system.inject_context(context)
        context.extend(self.state.messages)
        return context

    # ==========================================
    # INTERNAL - Methods for module coordination
    # ==========================================

    def _get_behavior_by_name(self, name: str):
        """Internal helper - still in BaseAgent."""
        return next((b for b in self.behaviors if b.get_name() == name), None)
```

---

## Module Responsibilities

### 1. src/agent_state.py

**What it does:**
- Defines AgentState dataclass
- Handles state serialization/deserialization
- Manages state file I/O

**How it's called:**
```python
# From BaseAgent.__init__
self.state_manager = StatePersistence(self.workspace)
self.state = self.state_manager.load_state()

# From BaseAgent.persist_state()
self.state_manager.persist(self.state)

# From BaseAgent.load_state()
self.state = self.state_manager.load_state()
```

**Does NOT:**
- Make decisions about when to persist (BaseAgent decides)
- Know about behaviors or tools
- Directly modify BaseAgent state

### 2. src/tool_dispatch.py

**What it does:**
- Maintains tool_registry (tool_name → behavior)
- Validates tool parameters
- Dispatches tool calls to behaviors
- Generates tool documentation

**How it's called:**
```python
# From BaseAgent.__init__
self.tool_dispatcher = ToolDispatcher(self)

# From BaseAgent.dispatch_tool()
result = self.tool_dispatcher.dispatch(tool_call)

# From BaseAgent.get_tools()
tools = self.tool_dispatcher.get_all_tools()

# From behavior registration
self.tool_dispatcher.register_tool("write_file", behavior)
```

**Key methods:**
```python
class ToolDispatcher:
    def __init__(self, agent: BaseAgent):
        self.agent = agent  # Reference back to BaseAgent
        self.tool_registry = {}

    def register_tool(self, tool_name: str, behavior):
        """Called when behaviors are loaded."""
        self.tool_registry[tool_name] = behavior

    def dispatch(self, tool_call: dict) -> dict:
        """Called from BaseAgent.dispatch_tool()."""
        tool_name = tool_call["function"]["name"]
        behavior = self.tool_registry.get(tool_name)

        if not behavior:
            return {"error": f"Unknown tool: {tool_name}"}

        # Validate parameters
        validated = self._validate_parameters(tool_call, behavior)
        if "error" in validated:
            return validated

        # Dispatch to behavior
        return behavior.dispatch_tool_call(tool_name, validated["arguments"])

    def get_all_tools(self) -> list[dict]:
        """Collect tools from all behaviors."""
        tools = []
        for behavior in self.agent.behaviors:
            tools.extend(behavior.get_tools())
        return tools
```

### 3. src/behavior_loader.py

**What it does:**
- Loads behavior config from YAML
- Imports behavior classes dynamically
- Initializes behaviors with params
- Registers behaviors with BaseAgent

**How it's called:**
```python
# From BaseAgent.__init__
self.behavior_loader = BehaviorLoader(self)
self.behavior_loader.load_from_config(config_file)

# Behavior loader populates:
# - self.behaviors (list)
# - self.tool_registry (via tool_dispatcher)
```

**Key methods:**
```python
class BehaviorLoader:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def load_from_config(self, config_file: str):
        """Main entry point - called from BaseAgent.__init__."""
        config = self._load_yaml(config_file)

        for behavior_spec in config.get("behaviors", []):
            behavior = self._create_behavior(behavior_spec)
            self.agent.behaviors.append(behavior)

            # Register tools
            for tool in behavior.get_tools():
                self.agent.tool_dispatcher.register_tool(
                    tool["function"]["name"],
                    behavior
                )

    def _create_behavior(self, spec: dict):
        """Creates and initializes behavior instance."""
        behavior_type = spec["type"]
        params = spec.get("params", {})

        # Dynamic import
        behavior_class = self._import_behavior_class(behavior_type)

        # Initialize
        behavior = behavior_class(**params)
        behavior.agent = self.agent  # Give behavior reference to agent

        return behavior
```

### 4. src/agent_events.py

**What it does:**
- Triggers lifecycle events to behaviors
- Manages event propagation
- Handles context injection from behaviors

**How it's called:**
```python
# From BaseAgent.__init__
self.event_system = EventSystem(self)

# From BaseAgent.set_goal()
self.event_system.trigger_goal_start(goal)

# From BaseAgent.build_context()
context = self.event_system.inject_context(context)

# From AgentLifecycle (during run loop)
self.agent.event_system.trigger_round_start(round_num, context)
```

**Key methods:**
```python
class EventSystem:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def trigger_goal_start(self, goal: str):
        """Called from BaseAgent.set_goal()."""
        for behavior in self.agent.behaviors:
            if hasattr(behavior, 'on_goal_start'):
                behavior.on_goal_start(self.agent, goal)

    def trigger_round_start(self, round_num: int, context: list) -> list:
        """Called during run loop - behaviors can modify context."""
        for behavior in self.agent.behaviors:
            if hasattr(behavior, 'on_round_start'):
                context = behavior.on_round_start(self.agent, round_num, context)
        return context

    def inject_context(self, context: list) -> list:
        """Called from BaseAgent.build_context()."""
        for behavior in self.agent.behaviors:
            if hasattr(behavior, 'on_initial_context'):
                context = behavior.on_initial_context(self.agent, context)
        return context
```

### 5. src/agent_lifecycle.py

**What it does:**
- Implements the main run() loop
- Coordinates setup, rounds, and teardown
- Calls LLM and processes responses

**How it's called:**
```python
# From BaseAgent.run()
return self.lifecycle.run(max_rounds)

# Lifecycle coordinates:
# - self.agent.build_context()
# - self.agent.call_llm()
# - self.agent.dispatch_tool()
# - self.agent.event_system.trigger_*()
```

**Key methods:**
```python
class AgentLifecycle:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def run(self, max_rounds: int = 12) -> dict:
        """Main entry point - called from BaseAgent.run()."""
        # Setup
        setup_result = self._setup_run(max_rounds)

        # Main loop
        for round_num in range(1, max_rounds + 1):
            result = self._run_round(round_num)

            if result["status"] in ["goal_complete", "goal_failed"]:
                break

        # Teardown
        return self._teardown(result)

    def _run_round(self, round_num: int) -> dict:
        """Single round execution."""
        # 1. Build context
        context = self.agent.build_context()

        # 2. Trigger round start (behaviors inject context)
        context = self.agent.event_system.trigger_round_start(round_num, context)

        # 3. Call LLM
        response = self.agent.call_llm(context)

        # 4. Process tool calls
        if "tool_calls" in response:
            for tool_call in response["tool_calls"]:
                result = self.agent.dispatch_tool(tool_call)
                self.agent.add_message({"role": "tool", "content": result})

        # 5. Trigger round end
        self.agent.event_system.trigger_round_end(round_num)

        # 6. Check completion
        return self._check_completion(response)
```

---

## Call Flow Example: run()

```
User calls: agent.run()
    ↓
BaseAgent.run()
    ↓ delegates to
AgentLifecycle.run()
    ↓ calls back to
BaseAgent.build_context()
    ↓ calls
EventSystem.inject_context()  # behaviors inject context
    ↓ returns to
AgentLifecycle._run_round()
    ↓ calls
BaseAgent.call_llm()  # LLM interaction
    ↓ returns response
AgentLifecycle processes response
    ↓ calls
BaseAgent.dispatch_tool()
    ↓ delegates to
ToolDispatcher.dispatch()
    ↓ calls
Behavior.dispatch_tool_call()  # behavior executes tool
    ↓ returns result
AgentLifecycle continues loop
    ↓ calls
EventSystem.trigger_round_end()
    ↓ eventually returns to
User receives: final result
```

---

## Why This Pattern?

### 1. **Backward Compatibility**
```python
# Old code still works:
agent = BaseAgent(...)
agent.set_goal("Build calculator")
result = agent.run()

# Public API unchanged
agent.persist_state()
tools = agent.get_tools()
```

### 2. **Single Responsibility**
- BaseAgent: Orchestration and coordination
- AgentState: State management
- ToolDispatcher: Tool routing
- BehaviorLoader: Behavior initialization
- EventSystem: Event propagation
- AgentLifecycle: Run loop logic

Each module does ONE thing well.

### 3. **Testability**
```python
# Test modules independently
def test_tool_dispatcher():
    dispatcher = ToolDispatcher(mock_agent)
    result = dispatcher.dispatch({"name": "write_file", ...})
    assert result["success"]

# Test BaseAgent with mocked modules
def test_base_agent():
    agent = BaseAgent(...)
    agent.lifecycle = MockLifecycle()
    agent.run()
```

### 4. **Maintainability**
- Changes to run loop? Edit agent_lifecycle.py only
- Changes to tool dispatch? Edit tool_dispatch.py only
- Changes to state persistence? Edit agent_state.py only

No need to search through 2,745 lines!

### 5. **Discoverability**
```
Need to understand run loop? → agent_lifecycle.py (400 lines)
Need to understand tool dispatch? → tool_dispatch.py (300 lines)
Need to understand behavior loading? → behavior_loader.py (600 lines)
```

Much easier than one 2,745-line file.

---

## What Changes for Users?

### For Agent Users (Python API):
**Nothing.** All public methods stay in BaseAgent.

```python
# Before refactor:
agent = BaseAgent(...)
agent.set_goal("...")
result = agent.run()

# After refactor:
agent = BaseAgent(...)  # Same!
agent.set_goal("...")   # Same!
result = agent.run()    # Same!
```

### For Behavior Developers:
**Almost nothing.** Behaviors still interact with BaseAgent.

```python
class MyBehavior(AgentBehavior):
    def on_round_start(self, agent, round_num, context):
        # Agent is still BaseAgent
        agent.add_message(...)  # Still works
        agent.workspace  # Still accessible
        return context
```

### For Core Developers:
**Better structure.** Navigate to specific modules instead of one huge file.

```python
# Need to modify tool dispatch logic?
# Before: Search through base_agent.py (2,745 lines)
# After:  Open tool_dispatch.py (300 lines)

# Need to modify run loop?
# Before: Search through base_agent.py (2,745 lines)
# After:  Open agent_lifecycle.py (400 lines)
```

---

## Migration Safety

### Phase 1: Extract without breaking
```python
# Week 1: Extract agent_state.py
class BaseAgent:
    def __init__(self, ...):
        self.state_manager = StatePersistence(...)
        self.state = self.state_manager.load_state()  # New

    def persist_state(self):
        self.state_manager.persist(self.state)  # Delegates

    # Old implementation commented out but kept for reference
    # def persist_state(self):
    #     with open(self.state_file, 'w') as f:
    #         json.dump(self.state.to_dict(), f)
```

### Phase 2: Verify tests still pass
```bash
pytest -xvs tests/test_base_agent.py
pytest -xvs tests/unit/test_agent_state.py  # New tests
```

### Phase 3: Remove old code
```python
# After tests pass, remove commented code
class BaseAgent:
    def persist_state(self):
        self.state_manager.persist(self.state)
```

### Phase 4: Repeat for each module

---

## Alternative Considered: Inheritance

```python
# Alternative: Multiple inheritance (NOT RECOMMENDED)
class BaseAgent(
    StatePersistence,
    ToolDispatcher,
    BehaviorLoader,
    EventSystem,
    AgentLifecycle
):
    pass
```

**Why we DON'T do this:**
- ❌ Violates "composition over inheritance" principle
- ❌ Method resolution order (MRO) conflicts
- ❌ Harder to test (can't mock individual components)
- ❌ Breaks single responsibility (everything is in self)

**Why composition is better:**
- ✅ Clear ownership (self.tool_dispatcher owns dispatch logic)
- ✅ Easy to test (mock self.tool_dispatcher)
- ✅ Explicit dependencies (self.lifecycle needs self.event_system)
- ✅ Follows Jetbox philosophy

---

## Summary

**BaseAgent remains the conductor:**
- Composes specialized modules as attributes
- Delegates to modules for specific tasks
- Coordinates module interactions
- Maintains public API

**Modules are helpers:**
- Do one thing well
- Called by BaseAgent
- Reference back to BaseAgent when needed
- Don't call each other directly

**Result:**
- Same functionality
- Same API
- Better structure
- Easier maintenance

**The key insight:** We're not moving logic AWAY from BaseAgent, we're extracting logic INTO modules that BaseAgent USES. BaseAgent stays in charge.
