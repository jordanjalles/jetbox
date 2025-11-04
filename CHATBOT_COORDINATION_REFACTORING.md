# ChatbotBehavior Coordination Refactoring

## Problem

The orchestrator agent had custom logic to coordinate with ChatbotBehavior for multi-task chat mode. This logic was orchestrator-specific, preventing other agents from using chat mode even if they had ChatbotBehavior added.

**Key Issue**: ChatbotBehavior coordination was hardcoded in `orchestrator_agent.py`, not available to other agents.

## Solution

**Move ChatbotBehavior coordination to `base_agent.py` so ANY agent can become a chatbot by adding the behavior.**

### Changes Made

#### 1. base_agent.py - Generic Chat Coordination

Added generic chat mode detection and coordination in `run_agent()`:

```python
@classmethod
def run_agent(cls, agent: BaseAgent, args: dict[str, Any]) -> None:
    """Execute agent with automatic ChatbotBehavior detection."""

    # Detect ChatbotBehavior
    chatbot_behavior = None
    for behavior in agent.behaviors:
        if behavior.get_name() == "chatbot":
            chatbot_behavior = behavior
            break

    # Use multi-task chat mode if ChatbotBehavior present
    if chatbot_behavior:
        cls._run_multi_task_chat_mode(agent, chatbot_behavior, ...)
    elif initial_message:
        # Single-goal execution (no ChatbotBehavior)
        agent.run()
    else:
        # No ChatbotBehavior and no goal - can't do anything
        print("Interactive mode not supported without ChatbotBehavior.")
```

Added `_run_multi_task_chat_mode()` helper:
- Defines execute_task callback
- Calls agent.run_task_round_loop() for each task
- Integrates with ChatbotBehavior.run_multi_task_chat_loop()
- Supports agent-specific hooks (pre_task_hook, cleanup_hook)

#### 2. orchestrator_agent.py - Simplified with Hooks

**Before**: 201 lines with custom run_agent() override
**After**: 130 lines with simple hooks

Removed:
- Entire `run_agent()` override (68 lines)
- `execute_task()` method (24 lines)

Added:
- `pre_task_hook()` - Called before each task (cleanup server requests)
- `cleanup_hook()` - Called at end of execution (stop servers)

### Results

#### Size Reduction
- **Original**: 246 lines
- **After infrastructure refactor**: 201 lines
- **After hook refactor**: **130 lines**
- **Total reduction**: 116 lines (47% smaller)

#### Architecture Improvements

**BEFORE**:
- ChatbotBehavior coordination only in orchestrator
- Other agents couldn't use chat mode
- Duplicated execution logic in orchestrator

**AFTER**:
- ChatbotBehavior coordination in base_agent (generic)
- ANY agent can use chat mode by adding ChatbotBehavior
- Agents customize via hooks (pre_task_hook, cleanup_hook)
- Orchestrator is minimal wrapper with hooks

### Benefits

✅ **Composability**: Add ChatbotBehavior to any agent for chat mode
✅ **DRY**: No duplicated execution logic across agents
✅ **Separation of Concerns**: Base infrastructure vs agent-specific hooks
✅ **Extensibility**: Other agents can add pre_task/cleanup hooks as needed

### Testing

All sanity checks passed (5/5):
- ✅ TaskExecutor instantiation
- ✅ Architect instantiation
- ✅ Orchestrator with ChatbotBehavior
- ✅ Orchestrator without ChatbotBehavior
- ✅ BaseAgent chat coordination detection

### Example: Adding Chat Mode to TaskExecutor

Before this refactoring, TaskExecutor couldn't do multi-task chat mode. Now it works automatically:

```python
# TaskExecutor with ChatbotBehavior (already in config)
agent = TaskExecutorAgent(workspace=".", goal=None)

# Run with chat mode (no --once flag)
python task_executor_agent.py

# Automatically enters multi-task chat mode!
# User can provide multiple tasks in one session
```

### Example: Adding Chat Mode to Custom Agent

```python
class MyCustomAgent(BaseAgent):
    def __init__(self, workspace, goal=None):
        super().__init__(
            name="my_agent",
            workspace=workspace,
            config_file="my_agent_config.yaml"  # Include ChatbotBehavior
        )

    def pre_task_hook(self):
        """Optional: called before each task."""
        print("Starting new task...")

    def cleanup_hook(self):
        """Optional: called at end of execution."""
        print("Cleaning up resources...")

# That's it! Multi-task chat mode works automatically.
```

## Conclusion

ChatbotBehavior coordination is now a **generic base_agent feature**, not orchestrator-specific. Any agent can become a chatbot by adding ChatbotBehavior to its config.

This follows the principle: **"the coordination of chatbotbehavior loop should happen between chatbotbehavior and base agent. Any agent should become a chatbot if we add the behavior."**
