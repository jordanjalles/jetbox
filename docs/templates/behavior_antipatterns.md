# Behavior Anti-Patterns

This document shows common mistakes when creating behaviors.

## ❌ Anti-Pattern 1: Cross-Behavior Dependencies

**Bad:**
```python
from behaviors.read_file_tools import ReadFileToolsBehavior

class MyBehavior(AgentBehavior):
    def __init__(self):
        self.file_tools = ReadFileToolsBehavior()  # ❌

    def my_method(self):
        self.file_tools.dispatch_tool(agent, "read_file", {...})  # ❌
```

**Why it's bad:** Behaviors must be independent. This creates coupling.

**Good:**
```python
class MyBehavior(AgentBehavior):
    def dispatch_tool(self, agent, tool_name, args):
        # Let the agent handle file operations
        # Agent will dispatch to appropriate behavior
        pass
```

**Or:** If your behavior needs file operations, it should provide its own file tools.

## ❌ Anti-Pattern 2: Multiple Responsibilities

**Bad:**
```python
class FileAndCommandBehavior(AgentBehavior):
    def get_tools(self):
        return [
            {"function": {"name": "read_file", ...}},    # ❌ File tool
            {"function": {"name": "run_command", ...}}   # ❌ Command tool
        ]
```

**Why it's bad:** Violates single responsibility. Should be two behaviors.

**Good:**
```python
# Split into two behaviors
class FileToolsBehavior(AgentBehavior):
    # Only file operations

class CommandToolsBehavior(AgentBehavior):
    # Only command execution
```

## ❌ Anti-Pattern 3: Hardcoded Agent Knowledge

**Bad:**
```python
class MyBehavior(AgentBehavior):
    def on_round_start(self, agent, round_number, context):
        # Check agent class name
        if agent.__class__.__name__ == 'OrchestratorAgent':  # ❌
            # Special logic for orchestrator only
            pass
```

**Why it's bad:** Behaviors should be agent-agnostic.

**Good:**
```python
class MyBehavior(AgentBehavior):
    def on_round_start(self, agent, round_number, context):
        # Work the same for all agents
        # Let agent config determine which agents use this behavior
        # If specific to one agent, name it accordingly (e.g., OrchestratorSpecificBehavior)
        pass
```

## ❌ Anti-Pattern 4: State Mutation Without Cleanup

**Bad:**
```python
class MyBehavior(AgentBehavior):
    def on_goal_start(self, agent, goal):
        self.temp_files = []
        self.create_temp_files()  # ❌ Never cleaned up
```

**Why it's bad:** Resource leaks.

**Good:**
```python
class MyBehavior(AgentBehavior):
    def on_goal_start(self, agent, goal):
        self.temp_files = []
        self.create_temp_files()

    def on_goal_complete(self, agent, success, summary):
        # Cleanup
        for f in self.temp_files:
            f.unlink()
        self.temp_files = []
```

## ❌ Anti-Pattern 5: Brittle Context Parsing

**Bad:**
```python
def on_round_start(self, agent, round_number, context):
    # Assumes specific context structure
    system_msg = context[0]  # ❌ May not be system message
    goal_msg = context[1]    # ❌ May not be goal
```

**Why it's bad:** Context structure varies by agent and behavior composition.

**Good:**
```python
def on_round_start(self, agent, round_number, context):
    # Use helper that handles structure differences
    return self.inject_user_message_after_system(
        context,
        f"Additional info: {agent.goal}"
    )

def on_initial_context(self, agent, context):
    # For static content, inject once
    goal = agent.goal if hasattr(agent, 'goal') else ''
    return self.inject_user_message_after_system(context, f"GOAL: {goal}")
```

## ❌ Anti-Pattern 6: Conversational System Prompts

**Bad (for agent config):**
```yaml
system_prompt: |
  Hi! I'm a helpful assistant. I'm here to help you with your coding tasks!
  Just tell me what you'd like to do and I'll try my best!
```

**Why it's bad:** Too conversational, not tool-focused.

**Good:**
```yaml
system_prompt: |
  You are a coding agent that executes tasks using tools.

  Guidelines:
  - ALWAYS use tools - never just respond with text
  - Use write_file to create files
  - Use run_bash to execute commands
  - Be concise and focused on the goal
```
