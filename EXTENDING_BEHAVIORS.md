# Extending Jetbox with Custom Behaviors

**Version**: 1.0
**Last Updated**: 2025-01-01

## Introduction

This guide walks you through creating custom behaviors for Jetbox agents. Behaviors are the building blocks that give agents their capabilities—from context management to tool provision to event handling.

### What Are Behaviors?

Behaviors are self-contained modules that:
- **Inject context** into LLM prompts (`enhance_context`)
- **Provide tools** for the agent to use (`get_tools`, `dispatch_tool`)
- **Handle events** from the agent lifecycle (`on_*` methods)
- **Add instructions** to the system prompt (`get_instructions`)

### Why Composability Matters

Jetbox's behavior system is built on **composability**: each behavior does ONE thing and works independently. This means:

- ✅ You can mix and match any behaviors without conflicts
- ✅ Adding/removing behaviors doesn't break others
- ✅ Behaviors can be tested in isolation
- ✅ No hidden dependencies between behaviors
- ✅ Configuration via YAML, not hardcoded

**Example**: TaskExecutor combines 8 separate behaviors (SubAgent, Compact, Files, Commands, Servers, Loop, Notes, Status) that all work together without knowing about each other.

---

## Architecture Principles

Before creating a behavior, understand these core principles:

### 1. Single Responsibility

**Each behavior does ONE thing and does it well.**

**GOOD**:
```python
class FileToolsBehavior(AgentBehavior):
    """Provides file operations ONLY."""
    def get_tools(self):
        return [write_file, read_file, list_dir]
```

**BAD**:
```python
class FileAndCommandBehavior(AgentBehavior):
    """Provides file ops AND command execution."""  # Too much!
    def get_tools(self):
        return [write_file, read_file, run_bash, run_pytest]
```

### 2. Composability

**Behaviors work independently and in any combination.**

**GOOD**:
```python
class WorkspaceTaskNotesBehavior(AgentBehavior):
    """Loads notes. Doesn't care about other behaviors."""
    def enhance_context(self, context, **kwargs):
        notes = self._load_notes()
        if notes:
            context.insert(1, {"role": "user", "content": notes})
        return context
```

**BAD**:
```python
class NotesWithCompactionBehavior(AgentBehavior):
    """Loads notes AND compacts context."""  # Should be 2 behaviors!
    def enhance_context(self, context, **kwargs):
        notes = self._load_notes()
        context = self._compact_if_full(context)  # Violates composability
        return context
```

### 3. No Hidden Dependencies

**No behavior embeds functionality from another.**

**GOOD**:
```python
# SubAgentContextBehavior: injects goal header ONLY
class SubAgentContextBehavior(AgentBehavior):
    def enhance_context(self, context, **kwargs):
        # ONLY injects delegated goal header
        context.insert(1, {"role": "user", "content": "DELEGATED GOAL: ..."})
        return context

# WorkspaceTaskNotesBehavior: loads notes ONLY
class WorkspaceTaskNotesBehavior(AgentBehavior):
    def enhance_context(self, context, **kwargs):
        # ONLY loads and injects notes
        notes = self._load_notes()
        if notes:
            context.insert(1, {"role": "user", "content": notes})
        return context
```

**BAD**:
```python
# SubAgentContextBehavior that also loads notes internally
class SubAgentContextBehavior(AgentBehavior):
    def enhance_context(self, context, **kwargs):
        context.insert(1, {"role": "user", "content": "DELEGATED GOAL: ..."})
        # BAD: Loading notes is a separate concern!
        notes = jetbox_notes.load_jetbox_notes()
        if notes:
            context.insert(2, {"role": "user", "content": notes})
        return context
```

### 4. Config-Driven

**Behaviors are configured via YAML, not hardcoded.**

**GOOD**:
```yaml
# task_executor_config.yaml
behaviors:
  - type: LoopDetectionBehavior
    params:
      max_repeats: 5
```

**BAD**:
```python
# Hardcoded in agent __init__
class TaskExecutorAgent:
    def __init__(self):
        self.loop_detector = LoopDetectionBehavior(max_repeats=5)  # Hardcoded!
```

### 5. Event-Driven

**Behaviors respond to lifecycle events independently.**

**GOOD**:
```python
class LoopDetectionBehavior(AgentBehavior):
    def on_tool_call(self, tool_name, args, result, **kwargs):
        # Records action independently
        self.action_history.append((tool_name, args))
```

**BAD**:
```python
class LoopDetectionBehavior(AgentBehavior):
    def on_tool_call(self, tool_name, args, result, **kwargs):
        # BAD: Assumes StatusDisplayBehavior exists
        status_display = kwargs["agent"].get_behavior("status_display")
        status_display.update_metrics(...)  # Violates independence
```

### 6. Clear Interfaces

**All behaviors implement standardized methods.**

**Required**:
- `get_name() -> str` - Unique identifier

**Optional**:
- `get_tools() -> list[dict]` - Tool definitions
- `dispatch_tool(tool_name, args, **kwargs) -> dict` - Tool handler
- `enhance_context(context, **kwargs) -> list[dict]` - Context injection
- `get_instructions() -> str` - System prompt additions
- `on_goal_start(goal, **kwargs)` - Goal start event
- `on_tool_call(tool_name, args, result, **kwargs)` - Tool call event
- `on_round_end(round_number, **kwargs)` - Round end event
- `on_timeout(elapsed_seconds, **kwargs)` - Timeout event
- `on_goal_complete(success, **kwargs)` - Goal complete event

---

## Behavior Lifecycle

Understanding when behaviors are called helps you design effective behaviors.

### Agent Initialization

```python
# 1. Agent created
agent = TaskExecutorAgent(workspace=".", goal="Create calculator")

# 2. Config loaded
config = load_config("task_executor_config.yaml")

# 3. Behaviors instantiated
for behavior_spec in config["behaviors"]:
    behavior = create_behavior(behavior_spec["type"], behavior_spec["params"])
    agent.add_behavior(behavior)

# 4. Tools registered
all_tools = []
for behavior in agent.behaviors:
    all_tools.extend(behavior.get_tools())
```

### Agent Execution

```python
# 5. Goal starts
for behavior in agent.behaviors:
    behavior.on_goal_start(goal=agent.goal, workspace_manager=agent.workspace)

# 6. Agent loop
while not done:
    # Build context
    context = [{"role": "system", "content": system_prompt}]
    for behavior in agent.behaviors:
        context = behavior.enhance_context(context, agent=agent, ...)

    # Call LLM
    response = llm.generate(context, tools=all_tools)

    # Execute tools
    for tool_call in response.tool_calls:
        # Find behavior that owns this tool
        behavior = find_behavior_for_tool(tool_call.name)
        result = behavior.dispatch_tool(tool_call.name, tool_call.args, agent=agent, ...)

        # Notify all behaviors
        for b in agent.behaviors:
            b.on_tool_call(tool_call.name, tool_call.args, result, agent=agent, ...)

    # Round end
    for behavior in agent.behaviors:
        behavior.on_round_end(round_number=round_num, agent=agent, ...)

# 7. Goal completes
for behavior in agent.behaviors:
    behavior.on_goal_complete(success=True, goal=agent.goal, ...)
```

### Event Timing

```
Agent Start
    ↓
on_goal_start()  ← All behaviors notified
    ↓
┌─ Agent Loop ─────────────────┐
│   enhance_context()  ← Build │  Repeat until
│   LLM Call                    │  goal complete
│   dispatch_tool()  ← Execute │
│   on_tool_call()  ← Notify   │
│   on_round_end()  ← Notify   │
└──────────────────────────────┘
    ↓
on_goal_complete()  ← All behaviors notified
    ↓
Agent End
```

---

## Creating a Custom Behavior

### Step 1: Define Your Behavior's Purpose

Ask yourself:
- What ONE thing does this behavior do?
- Does it provide tools, enhance context, or handle events?
- Can it work independently?
- Does it depend on other behaviors? (If yes, rethink the design)

**Example**: "I want to add code analysis tools"
- Purpose: Provide code quality analysis tools
- Type: Tool behavior
- Dependencies: None (just analyzes code passed as arguments)

### Step 2: Create Behavior File

```bash
# Create file in behaviors/ directory
touch behaviors/code_analysis.py
```

### Step 3: Implement Behavior Class

```python
# behaviors/code_analysis.py
"""
CodeAnalysisBehavior - Static code analysis tools.

Provides tools for analyzing code quality, complexity, and patterns.
"""

from typing import Any
from behaviors.base import AgentBehavior


class CodeAnalysisBehavior(AgentBehavior):
    """
    Behavior that provides code analysis tools.

    Supports multiple languages and analysis types.
    """

    def __init__(self, languages: list[str] = None, **kwargs):
        """
        Initialize code analysis behavior.

        Args:
            languages: List of languages to support (default: ["python"])
            **kwargs: Additional config for forward compatibility
        """
        self.languages = languages or ["python"]

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "code_analysis"

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return code analysis tool definitions.

        Returns:
            List of tool definitions for code analysis
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_code_complexity",
                    "description": "Analyze code complexity metrics (cyclomatic, cognitive, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to analyze"
                            },
                            "language": {
                                "type": "string",
                                "description": f"Programming language",
                                "enum": self.languages
                            }
                        },
                        "required": ["code", "language"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_code_smells",
                    "description": "Detect common code smells and anti-patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to analyze"
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language",
                                "enum": self.languages
                            }
                        },
                        "required": ["code", "language"]
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Dispatch code analysis tool calls.

        Args:
            tool_name: Tool to execute
            args: Tool arguments
            **kwargs: Additional context (agent, workspace, etc.)

        Returns:
            Analysis result
        """
        if tool_name == "analyze_code_complexity":
            return self._analyze_complexity(args["code"], args["language"])
        elif tool_name == "detect_code_smells":
            return self._detect_smells(args["code"], args["language"])
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _analyze_complexity(self, code: str, language: str) -> dict[str, Any]:
        """
        Analyze code complexity.

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            Complexity metrics
        """
        # Simplified placeholder implementation
        lines = code.split("\n")
        num_functions = code.count("def ") if language == "python" else 0
        cyclomatic_complexity = 1 + code.count("if ") + code.count("for ") + code.count("while ")

        return {
            "success": True,
            "metrics": {
                "lines_of_code": len(lines),
                "num_functions": num_functions,
                "cyclomatic_complexity": cyclomatic_complexity,
                "complexity_grade": "A" if cyclomatic_complexity < 10 else "B"
            }
        }

    def _detect_smells(self, code: str, language: str) -> dict[str, Any]:
        """
        Detect code smells.

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            List of detected code smells
        """
        smells = []

        # Simple heuristics for Python
        if language == "python":
            if len(code.split("\n")) > 50:
                smells.append({
                    "type": "long_method",
                    "severity": "warning",
                    "message": "Method/function is too long (>50 lines)"
                })

            if code.count("global ") > 0:
                smells.append({
                    "type": "global_variable",
                    "severity": "warning",
                    "message": "Use of global variables detected"
                })

        return {
            "success": True,
            "smells": smells,
            "smell_count": len(smells)
        }

    def get_instructions(self) -> str:
        """
        Return code analysis instructions.

        Returns:
            Instructions for using code analysis tools
        """
        return f"""
CODE ANALYSIS TOOLS:
You have access to code analysis tools for: {', '.join(self.languages)}

Available tools:
- analyze_code_complexity: Get complexity metrics
- detect_code_smells: Find anti-patterns and code smells

Use these tools to:
- Assess code quality
- Identify refactoring opportunities
- Provide feedback on code complexity
"""
```

### Step 4: Register Behavior

Add your behavior to `behaviors/__init__.py`:

```python
# behaviors/__init__.py
from behaviors.code_analysis import CodeAnalysisBehavior

__all__ = [
    # ... existing behaviors
    "CodeAnalysisBehavior",
]
```

### Step 5: Configure Behavior

Create or update a config file:

```yaml
# code_reviewer_config.yaml
behaviors:
  # Context management
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 16000

  # File reading (but not writing - read-only reviewer)
  - type: FileToolsBehavior
    params: {}

  # Your custom behavior
  - type: CodeAnalysisBehavior
    params:
      languages: ["python", "javascript", "typescript"]

  # Utility
  - type: LoopDetectionBehavior
    params:
      max_repeats: 3
```

### Step 6: Test Your Behavior

See "Testing Your Behavior" section below.

---

## Testing Your Behavior

### Isolation Testing

Test your behavior independently:

```python
# tests/test_code_analysis_behavior.py
import pytest
from behaviors.code_analysis import CodeAnalysisBehavior


class TestCodeAnalysisBehaviorIsolation:
    """Test CodeAnalysisBehavior works independently."""

    def test_instantiation(self):
        """Behavior can be instantiated without other behaviors."""
        behavior = CodeAnalysisBehavior(languages=["python", "javascript"])
        assert behavior.get_name() == "code_analysis"

    def test_tools_registered(self):
        """Behavior provides tools."""
        behavior = CodeAnalysisBehavior()
        tools = behavior.get_tools()
        assert len(tools) == 2
        assert any(t["function"]["name"] == "analyze_code_complexity" for t in tools)
        assert any(t["function"]["name"] == "detect_code_smells" for t in tools)

    def test_complexity_analysis(self):
        """analyze_code_complexity tool works."""
        behavior = CodeAnalysisBehavior()
        code = """
def example():
    if x:
        if y:
            return True
    return False
"""
        result = behavior.dispatch_tool("analyze_code_complexity", {
            "code": code,
            "language": "python"
        })

        assert result["success"] == True
        assert "metrics" in result
        assert result["metrics"]["cyclomatic_complexity"] > 1

    def test_smell_detection(self):
        """detect_code_smells tool works."""
        behavior = CodeAnalysisBehavior()
        code = "global x\nx = 5"  # Uses global variable

        result = behavior.dispatch_tool("detect_code_smells", {
            "code": code,
            "language": "python"
        })

        assert result["success"] == True
        assert result["smell_count"] > 0
        assert any(s["type"] == "global_variable" for s in result["smells"])
```

### Composition Testing

Test your behavior with others:

```python
class TestCodeAnalysisBehaviorComposition:
    """Test CodeAnalysisBehavior composes with other behaviors."""

    def test_with_file_tools(self):
        """CodeAnalysisBehavior works with FileToolsBehavior."""
        from behaviors.file_tools import FileToolsBehavior

        file_behavior = FileToolsBehavior()
        analysis_behavior = CodeAnalysisBehavior()

        # Get all tools
        all_tools = file_behavior.get_tools() + analysis_behavior.get_tools()

        # No tool name conflicts
        tool_names = [t["function"]["name"] for t in all_tools]
        assert len(tool_names) == len(set(tool_names))  # All unique

    def test_with_context_behavior(self):
        """CodeAnalysisBehavior works with context behaviors."""
        from behaviors.compact_when_near_full import CompactWhenNearFullBehavior

        compact = CompactWhenNearFullBehavior(max_tokens=8000)
        analysis = CodeAnalysisBehavior()

        # Build context
        context = [{"role": "system", "content": "You are a code reviewer"}]
        context = compact.enhance_context(context)

        # No conflicts (analysis doesn't enhance context)
        assert len(context) == 1

    def test_order_independence(self):
        """Behavior order doesn't matter."""
        from behaviors.file_tools import FileToolsBehavior
        from behaviors.loop_detection import LoopDetectionBehavior

        # Order 1
        behaviors1 = [
            CodeAnalysisBehavior(),
            FileToolsBehavior(),
            LoopDetectionBehavior()
        ]

        # Order 2
        behaviors2 = [
            LoopDetectionBehavior(),
            CodeAnalysisBehavior(),
            FileToolsBehavior()
        ]

        # Both orders work
        tools1 = [t for b in behaviors1 for t in b.get_tools()]
        tools2 = [t for b in behaviors2 for t in b.get_tools()]
        assert len(tools1) == len(tools2)
```

### Integration Testing

Test with a real agent:

```python
class TestCodeAnalysisBehaviorIntegration:
    """Test CodeAnalysisBehavior in a real agent."""

    def test_in_agent(self, tmp_path):
        """CodeAnalysisBehavior works in BaseAgent."""
        from base_agent import BaseAgent

        # Create agent with custom config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
behaviors:
  - type: CompactWhenNearFullBehavior
    params:
      max_tokens: 8000
  - type: CodeAnalysisBehavior
    params:
      languages: ["python"]
""")

        agent = BaseAgent(
            name="code_reviewer",
            workspace=tmp_path,
            config_file=str(config_path)
        )

        # Verify behavior loaded
        behavior_names = [b.get_name() for b in agent.behaviors]
        assert "code_analysis" in behavior_names

        # Verify tools available
        tool_names = [t["function"]["name"] for t in agent.get_all_tools()]
        assert "analyze_code_complexity" in tool_names
```

---

## Common Patterns

### Pattern 1: Tool-Only Behavior

Provides tools without context enhancement or events.

```python
class SimpleBehavior(AgentBehavior):
    def get_name(self):
        return "simple"

    def get_tools(self):
        return [...]  # Tool definitions

    def dispatch_tool(self, tool_name, args, **kwargs):
        # Handle tools
        return {"result": "..."}
```

**Examples**: `FileToolsBehavior`, `CommandToolsBehavior`, `ServerToolsBehavior`

### Pattern 2: Context-Only Behavior

Enhances context without providing tools.

```python
class ContextBehavior(AgentBehavior):
    def get_name(self):
        return "context"

    def enhance_context(self, context, **kwargs):
        # Modify context
        context.insert(1, {"role": "user", "content": "..."})
        return context
```

**Examples**: `SubAgentContextBehavior`, `CompactWhenNearFullBehavior`

### Pattern 3: Event-Only Behavior

Handles events without tools or context.

```python
class EventBehavior(AgentBehavior):
    def get_name(self):
        return "events"

    def on_tool_call(self, tool_name, args, result, **kwargs):
        # Track or log events
        self.log(f"Tool called: {tool_name}")
```

**Examples**: `LoopDetectionBehavior`, `StatusDisplayBehavior`

### Pattern 4: Hybrid Behavior

Combines multiple capabilities.

```python
class HybridBehavior(AgentBehavior):
    def get_name(self):
        return "hybrid"

    def get_tools(self):
        return [...]  # Provide tools

    def dispatch_tool(self, tool_name, args, **kwargs):
        return {"result": "..."}

    def enhance_context(self, context, **kwargs):
        return context  # Also enhance context

    def on_tool_call(self, tool_name, args, result, **kwargs):
        pass  # Also handle events
```

**Examples**: `WorkspaceTaskNotesBehavior` (tools + context + events)

---

## Anti-Patterns

### Anti-Pattern 1: Embedded Functionality

**DON'T** embed functionality from other behaviors:

```python
# BAD
class BadBehavior(AgentBehavior):
    def enhance_context(self, context, **kwargs):
        # BAD: Doing compaction internally
        if len(context) > 10:
            context = self._compact_context(context)

        # BAD: Loading notes internally
        notes = jetbox_notes.load_jetbox_notes()
        context.insert(1, {"role": "user", "content": notes})

        return context
```

**DO** keep behaviors separate:

```python
# GOOD
class ContextHeaderBehavior(AgentBehavior):
    """ONLY injects header."""
    def enhance_context(self, context, **kwargs):
        context.insert(1, {"role": "user", "content": "HEADER: ..."})
        return context

# Use CompactWhenNearFullBehavior separately
# Use WorkspaceTaskNotesBehavior separately
```

### Anti-Pattern 2: Behavior Dependencies

**DON'T** assume other behaviors exist:

```python
# BAD
class BadBehavior(AgentBehavior):
    def on_tool_call(self, tool_name, args, result, **kwargs):
        # BAD: Assumes StatusDisplayBehavior exists
        agent = kwargs["agent"]
        status_behavior = agent.get_behavior("status_display")
        status_behavior.update(...)  # Will crash if not present
```

**DO** work independently:

```python
# GOOD
class GoodBehavior(AgentBehavior):
    def on_tool_call(self, tool_name, args, result, **kwargs):
        # GOOD: Does its own work independently
        self.action_count += 1
        self.last_tool = tool_name
```

### Anti-Pattern 3: Global State

**DON'T** use global variables:

```python
# BAD
action_history = []  # Global!

class BadBehavior(AgentBehavior):
    def on_tool_call(self, tool_name, args, result, **kwargs):
        action_history.append(tool_name)  # BAD: Global state
```

**DO** use instance variables:

```python
# GOOD
class GoodBehavior(AgentBehavior):
    def __init__(self):
        self.action_history = []  # Instance variable

    def on_tool_call(self, tool_name, args, result, **kwargs):
        self.action_history.append(tool_name)  # GOOD
```

### Anti-Pattern 4: Tool Name Conflicts

**DON'T** duplicate tool names:

```python
# BAD: FileToolsBehavior already provides write_file
class BadBehavior(AgentBehavior):
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_file",  # CONFLICT!
                    "description": "...",
                }
            }
        ]
```

**DO** use unique tool names:

```python
# GOOD
class GoodBehavior(AgentBehavior):
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_structured_file",  # Unique
                    "description": "...",
                }
            }
        ]
```

---

## Checklist: Pre-Integration Validation

Before integrating your behavior, validate it against this checklist:

### Design Checklist

- [ ] **Single Responsibility**: Behavior does ONE thing only
- [ ] **No Hidden Dependencies**: Doesn't embed functionality from other behaviors
- [ ] **Self-Contained**: Can be instantiated and used independently
- [ ] **No Global State**: Uses instance variables only
- [ ] **Unique Tool Names**: No conflicts with existing behaviors

### Implementation Checklist

- [ ] **Implements get_name()**: Returns unique identifier
- [ ] **Accepts **kwargs**: All methods accept `**kwargs` for forward compatibility
- [ ] **Handles Missing Context**: Gracefully handles missing kwargs
- [ ] **Returns Correct Types**: Methods return expected types (list, dict, str, etc.)
- [ ] **No Side Effects**: Doesn't modify kwargs or shared state

### Testing Checklist

- [ ] **Isolation Test**: Behavior works without other behaviors
- [ ] **Composition Test**: Works with 2-3 other behaviors
- [ ] **Tool Test**: All tools can be called successfully
- [ ] **Context Test**: enhance_context() doesn't break context structure
- [ ] **Event Test**: Event handlers don't crash

### Documentation Checklist

- [ ] **Docstrings**: Class and all public methods documented
- [ ] **Config Example**: YAML example provided
- [ ] **Usage Instructions**: get_instructions() returns helpful guide
- [ ] **Added to __init__.py**: Behavior exported from behaviors module

---

## Getting Help

### Resources

- **Behavior Documentation**: `BEHAVIORS_DOCUMENTATION.md` - Complete reference
- **Migration Guide**: `MIGRATION_GUIDE.md` - Examples of behavior composition
- **Main Documentation**: `CLAUDE.md` - System overview
- **Example Behaviors**: `behaviors/` directory - Real implementations

### Common Questions

**Q: My behavior depends on another behavior. What should I do?**

A: Rethink your design. Behaviors should be independent. If you need functionality from another behavior, extract the shared logic into a utility module that both behaviors can use.

**Q: How do I pass data between behaviors?**

A: Use kwargs. The agent passes context through kwargs to all behaviors. Example: `workspace_manager`, `agent`, `context_manager`.

**Q: Can I modify kwargs in my behavior?**

A: No. Treat kwargs as read-only. If you need to pass data, use the return value or store it in instance variables.

**Q: What if my behavior needs configuration?**

A: Add parameters to `__init__()` and accept them from the YAML config's `params` section.

**Q: How do I test event handlers?**

A: Mock the event call directly:

```python
behavior = MyBehavior()
behavior.on_tool_call(
    tool_name="test_tool",
    args={"arg": "value"},
    result={"success": True},
    agent=mock_agent
)
# Assert behavior handled event correctly
```

---

## Summary

### Key Takeaways

1. **Behaviors are composable**: Each does ONE thing, works independently
2. **No hidden dependencies**: Don't embed functionality from other behaviors
3. **Config-driven**: YAML configuration, not hardcoded
4. **Event-driven**: Respond to lifecycle events without knowing other behaviors
5. **Test thoroughly**: Isolation + composition + integration tests

### Next Steps

1. Review existing behaviors in `behaviors/` directory
2. Design your behavior following the six principles
3. Implement and test your behavior
4. Validate against the pre-integration checklist
5. Add to your agent config and test

### Remember

The behavior system's power comes from **composability**. Keep behaviors small, focused, and independent. When in doubt, split into multiple behaviors.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-01
**Author**: Jetbox Team
