# Template Usage Guide

Complete guide for using Jetbox behavior and agent templates to create new components.

## Overview

The Jetbox self-extensibility system provides templates to help you create:
- **Behaviors**: Composable capabilities that agents can use
- **Agent Configurations**: YAML files defining agent roles and behavior sets
- **Tests**: Comprehensive test suites for behaviors

This guide shows how to use each template and validate your generated code.

---

## Quick Start

### Creating a New Behavior

1. **Choose the right template:**
   - Simple behavior with no tools? → `behavior_minimal_template.py`
   - Behavior with tools? → `behavior_with_tools_template.py`
   - Context enhancement? → `behavior_context_enhancement_template.py`

2. **Copy and fill in placeholders:**
   ```bash
   cp docs/templates/behavior_with_tools_template.py behaviors/my_new_behavior.py
   ```

3. **Replace placeholders** (marked with `{PLACEHOLDER}`):
   - `{BEHAVIOR_NAME}`: Snake case identifier (e.g., "my_tool")
   - `{BEHAVIOR_CLASS_NAME}`: PascalCase class name (e.g., "MyTool")
   - `{TOOL_NAME}`, `{PARAM_NAME}`, etc.

4. **Validate your code:**
   ```python
   from utils.behavior_validator import validate_python_syntax, validate_behavior_independence

   with open("behaviors/my_new_behavior.py") as f:
       code = f.read()

   print(validate_python_syntax(code))
   print(validate_behavior_independence(code))
   ```

5. **Write tests** using `behavior_test_template.py`

6. **Run tests:**
   ```bash
   pytest tests/test_my_new_behavior.py -v
   ```

---

## Template Reference

### 1. Minimal Behavior Template

**File:** `docs/templates/behavior_minimal_template.py`

**When to use:**
- Simple behavior with no tools
- Context enhancement only
- Event handler behavior

**Placeholders:**
- `{BEHAVIOR_NAME}`: Identifier returned by `get_name()`
- `{BEHAVIOR_CLASS_NAME}`: Class name (e.g., MyBehavior)
- `{ONE_SENTENCE_DESCRIPTION}`: Brief description
- `{FEATURE_1}`, `{FEATURE_2}`: Key features
- `{DETAILED_DESCRIPTION}`: Full docstring
- `{WHAT_IT_PROVIDES}`: What capabilities it adds
- `{WHAT_IT_DOES_NOT_DO}`: What it explicitly doesn't do

**Example:**
```python
"""
PerformanceMonitorBehavior - Tracks agent performance metrics

Features:
- Measures LLM response times
- Counts tool calls
- Tracks success rates

This is a MINIMAL behavior showing the simplest possible implementation.
"""

from typing import Any
from behaviors.base import AgentBehavior


class PerformanceMonitorBehavior(AgentBehavior):
    """
    Tracks performance metrics during agent execution.

    This behavior provides: Performance tracking and statistics
    This behavior does NOT: Modify agent behavior or provide tools
    """

    def __init__(self, **kwargs):
        """Initialize performance monitor behavior."""
        self.metrics = {
            "tool_calls": 0,
            "errors": 0
        }

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "performance_monitor"

    def on_tool_call(self, agent, tool_name, args, result):
        """Track tool calls."""
        self.metrics["tool_calls"] += 1
        if "error" in result:
            self.metrics["errors"] += 1
```

---

### 2. Tool-Providing Behavior Template

**File:** `docs/templates/behavior_with_tools_template.py`

**When to use:**
- Behavior provides tools for LLM to call
- Tools perform actions (file ops, API calls, etc.)

**Placeholders:**
- All from minimal template, plus:
- `{TOOL_NAME}`: Name of the tool
- `{TOOL_DESCRIPTION}`: What the tool does
- `{TOOL_CATEGORY}`: Category (e.g., "file operation")
- `{PARAM_NAME}`: Parameter name
- `{PARAM_TYPE}`: Parameter type (string, integer, etc.)
- `{PARAM_DESCRIPTION}`: Parameter description
- `{REQUIRED_PARAM}`: Required parameter name
- `{DEFAULT_VALUE}`: Default value for parameter

**Key Patterns:**

1. **Agent-first dispatch signature:**
   ```python
   def dispatch_tool(self, agent, tool_name, args):
       # Access agent.workspace, agent.state directly
   ```

2. **Parameter invention tolerance:**
   ```python
   supported = {"path", "content"}
   unsupported = set(args.keys()) - supported
   if unsupported:
       print(f"[{self.get_name()}] Ignoring parameters: {unsupported}")
   ```

3. **Workspace access:**
   ```python
   workspace = agent.workspace if hasattr(agent, 'workspace') else None
   ```

**Example:**
```python
"""
JsonValidatorBehavior - Validates JSON against schemas

Provides tools:
- validate_json: Validate JSON data against a schema
"""

from typing import Any, TYPE_CHECKING
from behaviors.base import AgentBehavior
import json
import jsonschema

if TYPE_CHECKING:
    from behaviors.base_agent import BaseAgent


class JsonValidatorBehavior(AgentBehavior):
    """Provides JSON validation tools for agent use."""

    def __init__(self, workspace_manager=None, **kwargs):
        self.workspace_manager = workspace_manager

    def get_name(self) -> str:
        return "json_validator"

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "validate_json",
                    "description": "Validate JSON data against a schema",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "JSON string to validate"
                            },
                            "schema": {
                                "type": "string",
                                "description": "JSON schema string"
                            }
                        },
                        "required": ["data", "schema"]
                    }
                }
            }
        ]

    def dispatch_tool(self, agent: "BaseAgent", tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "validate_json":
            return self._execute_validate_json(agent, args)
        else:
            return super().dispatch_tool(agent, tool_name, args)

    def _execute_validate_json(self, agent: "BaseAgent", args: dict[str, Any]) -> dict[str, Any]:
        try:
            data_str = args.get("data")
            schema_str = args.get("schema")

            # Warn about unsupported parameters
            supported = {"data", "schema"}
            unsupported = set(args.keys()) - supported
            if unsupported:
                print(f"[{self.get_name()}] Ignoring parameters: {unsupported}")

            # Parse JSON
            data = json.loads(data_str)
            schema = json.loads(schema_str)

            # Validate
            jsonschema.validate(data, schema)

            return {"result": "Valid JSON", "success": True}

        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {str(e)}"}
        except jsonschema.ValidationError as e:
            return {"error": f"Validation failed: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}
```

---

### 3. Context Enhancement Template

**File:** `docs/templates/behavior_context_enhancement_template.py`

**When to use:**
- Behavior modifies context sent to LLM
- Injects warnings, guidance, or information
- Adds dynamic prompts based on state

**Placeholders:**
- `{WHAT_INFORMATION}`: What information is injected
- `{CONTEXT_HEADER}`: Header for injected message
- `{DYNAMIC_WARNING}`: Warning message

**Key Concepts:**

1. **on_initial_context (called ONCE):**
   - Use for static content that doesn't change
   - Examples: Goal description, tool documentation, loaded notes

2. **on_round_start (called EVERY round):**
   - Use for dynamic content that changes
   - Examples: Loop warnings, progress updates, changing guidance

**Example:**
```python
"""
DeadlineTrackerBehavior - Tracks task deadlines and warns agent

Enhances context by: Injecting deadline warnings
"""

from typing import Any, TYPE_CHECKING
from behaviors.base import AgentBehavior
from datetime import datetime

if TYPE_CHECKING:
    from behaviors.base_agent import BaseAgent


class DeadlineTrackerBehavior(AgentBehavior):
    """Behavior that enhances context with deadline information."""

    def __init__(self, deadline=None, **kwargs):
        self.deadline = deadline
        self.state = {"warned": False}

    def get_name(self) -> str:
        return "deadline_tracker"

    def on_initial_context(self, agent: "BaseAgent", context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject deadline information into initial context (called ONCE)."""
        if self.deadline:
            message = f"DEADLINE: Task must be completed by {self.deadline}"
            return self.inject_user_message_after_system(context, message)
        return context

    def on_round_start(self, agent: "BaseAgent", round_number: int, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject dynamic deadline warnings (called EVERY round)."""
        if self.deadline:
            now = datetime.now()
            if now > self.deadline and not self.state["warned"]:
                warning = "⚠️ DEADLINE PASSED! Complete task immediately."
                context = self.inject_user_message_after_system(context, warning)
                self.state["warned"] = True

        return context
```

---

### 4. Behavior Test Template

**File:** `docs/templates/behavior_test_template.py`

**When to use:**
- Testing any new behavior
- Ensuring behavior follows Jetbox patterns

**Placeholders:**
- `{BEHAVIOR_CLASS_NAME}`: Behavior class name
- `{BEHAVIOR_MODULE}`: Module name (e.g., "json_validator")
- `{BEHAVIOR_NAME}`: Identifier from get_name()
- `{TOOL_NAME}`, `{PARAM_NAME}`, `{TEST_VALUE}`: Test data

**Key Tests to Include:**

1. **test_get_name()** - Verify identifier
2. **test_initialization()** - Ensure no init errors
3. **test_tool_schema()** - Validate tool definitions
4. **test_tool_dispatch_success()** - Test happy path
5. **test_tool_dispatch_unknown_tool()** - Test error handling
6. **test_initial_context_injection()** - Test static context
7. **test_round_start_context_injection()** - Test dynamic context

**Example:**
```python
"""
Tests for JsonValidatorBehavior.
"""

import pytest
from unittest.mock import Mock
from behaviors.json_validator import JsonValidatorBehavior


class TestJsonValidatorBehavior:
    def test_get_name(self):
        behavior = JsonValidatorBehavior()
        assert behavior.get_name() == "json_validator"

    def test_initialization(self):
        behavior = JsonValidatorBehavior()
        assert behavior is not None

    def test_tool_schema(self):
        behavior = JsonValidatorBehavior()
        tools = behavior.get_tools()

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "validate_json"

    def test_tool_dispatch_success(self):
        behavior = JsonValidatorBehavior()
        mock_agent = Mock()

        result = behavior.dispatch_tool(
            agent=mock_agent,
            tool_name="validate_json",
            args={
                "data": '{"name": "test"}',
                "schema": '{"type": "object", "properties": {"name": {"type": "string"}}}'
            }
        )

        assert result["success"] is True

    def test_tool_dispatch_invalid_json(self):
        behavior = JsonValidatorBehavior()
        mock_agent = Mock()

        result = behavior.dispatch_tool(
            agent=mock_agent,
            tool_name="validate_json",
            args={
                "data": "not valid json",
                "schema": "{}"
            }
        )

        assert "error" in result
```

---

### 5. Agent Configuration Template

**File:** `docs/templates/agent_config_template.yaml`

**When to use:**
- Creating a new agent type
- Composing behaviors for specific purpose

**Placeholders:**
- `{AGENT_NAME}`: Agent name (e.g., "DataAnalyst")
- `{ROLE_DESCRIPTION}`: Short role description
- `{WHAT_IT_DOES}`, `{WHEN_TO_USE_IT}`, `{KEY_CAPABILITIES}`, `{USE_CASES}`: Blurb components
- `{DELEGATION_TOOL_NAME}`: Tool name for calling this agent
- `{TOOL_DESCRIPTION}`: Tool description
- `{PARAM_NAME}`, `{PARAM_TYPE}`, `{PARAM_DESCRIPTION}`: Parameter specs
- `{AGENT_DESCRIPTION}`: First line of system prompt
- `{GUIDELINE_1}`, `{GUIDELINE_2}`: Guidelines for agent
- `{CUSTOM_BEHAVIOR}`: Any custom behaviors

**Example:**
```yaml
# DataAnalyst Agent Configuration

role: "Data analysis specialist"

blurb: |
  DataAnalyst specializes in analyzing datasets and generating insights.
  Handles data cleaning, visualization, and statistical analysis.
  Can work with CSV, JSON, and other common data formats.
  Best for exploratory data analysis and reporting tasks.

delegation_tool:
  name: "delegate_to_data_analyst"
  description: "Delegate data analysis task to DataAnalyst"
  parameters:
    dataset_path:
      type: string
      description: "Path to dataset file"
      required: true
    analysis_type:
      type: string
      description: "Type of analysis (summary, visualization, correlation, etc.)"
      required: true

system_prompt: |
  You are a data analysis agent specialized in working with datasets.

  Guidelines:
  - ALWAYS use tools - never just respond with text
  - Use read_file to load datasets
  - Use run_bash to run pandas/numpy scripts
  - Create visualizations when appropriate
  - Report findings in clear, structured format

behaviors:
  - type: ChatbotBehavior
  - type: CompactWhenNearFullBehavior
  - type: DirectoryToolsBehavior
  - type: ReadFileToolsBehavior
  - type: WriteFileToolsBehavior
  - type: CommandToolsBehavior
  - type: LoopDetectionBehavior
  - type: WorkspaceTaskNotesBehavior
```

---

## Validation Workflow

### Step-by-Step Validation

1. **Validate Python Syntax:**
   ```python
   from utils import validate_python_syntax

   result = validate_python_syntax(code)
   if not result["valid"]:
       print(f"Syntax error: {result['error']}")
   ```

2. **Validate Behavior Independence:**
   ```python
   from utils import validate_behavior_independence

   result = validate_behavior_independence(code)
   if not result["valid"]:
       print(f"Cross-imports detected: {result['violations']}")
   ```

3. **Validate Tool Schemas:**
   ```python
   from utils import validate_tool_schema

   # Extract tool from behavior
   tools = behavior.get_tools()
   for tool in tools:
       result = validate_tool_schema(tool)
       if not result["valid"]:
           print(f"Invalid tool schema: {result['error']}")
   ```

4. **Validate Class Structure:**
   ```python
   from utils import validate_behavior_class_structure

   result = validate_behavior_class_structure(code)
   if not result["valid"]:
       print(f"Structure error: {result['error']}")
   ```

### Agent Config Validation

1. **Validate YAML Syntax:**
   ```python
   from utils import validate_yaml_syntax

   result = validate_yaml_syntax(file_path="my_agent_config.yaml")
   if not result["valid"]:
       print(f"YAML error: {result['error']}")
   config = result["data"]
   ```

2. **Validate Config Structure:**
   ```python
   from utils import validate_agent_config_structure

   result = validate_agent_config_structure(config)
   if not result["valid"]:
       print(f"Config error: {result['error']}")
   if "warnings" in result:
       for warning in result["warnings"]:
           print(f"Warning: {warning}")
   ```

3. **Validate Behavior References:**
   ```python
   from utils import validate_behavior_references

   result = validate_behavior_references(config)
   if not result["valid"]:
       print(f"Unknown behaviors: {result['unknown_behaviors']}")
   ```

4. **Validate Delegation DAG:**
   ```python
   from utils import validate_agent_dag

   # Load agents.yaml
   result = validate_agent_dag(agents_config)
   if not result["valid"]:
       print(f"Delegation cycle: {result['cycle']}")
   ```

---

## Best Practices

### Behavior Design

1. **Single Responsibility:**
   - Each behavior does ONE thing
   - If you need multiple capabilities, create multiple behaviors

2. **No Dependencies:**
   - Never import other behaviors
   - Use agent composition instead

3. **Workspace Awareness:**
   - Always access `agent.workspace` for file paths
   - Use `agent.workspace_manager.resolve_path()` when available

4. **Parameter Tolerance:**
   - Warn about unsupported parameters, don't crash
   - LLMs often invent parameters

5. **Error Handling:**
   - Return `{"error": "message"}` for failures
   - Return `{"result": data, "success": True}` for success

### Agent Configuration

1. **Tool-Focused Prompts:**
   - Emphasize tool usage
   - Avoid conversational tone
   - Be specific about guidelines

2. **Minimal Behavior Sets:**
   - Only include behaviors you need
   - More behaviors = more complexity

3. **Clear Delegation Tools:**
   - Name: `delegate_to_<agent>`
   - Description: Clear purpose
   - Parameters: Well-documented

### Testing

1. **Comprehensive Coverage:**
   - Test all public methods
   - Test error cases
   - Test edge cases

2. **Independence Verification:**
   - Use AST checking for imports
   - Verify no hardcoded references

3. **Integration Tests:**
   - Test behavior with real agent
   - Test tool execution end-to-end

---

## Common Pitfalls

### ❌ Cross-Behavior Dependencies
```python
# BAD
from behaviors.read_file_tools import ReadFileToolsBehavior

class MyBehavior(AgentBehavior):
    def __init__(self):
        self.file_tools = ReadFileToolsBehavior()
```

### ❌ Multiple Responsibilities
```python
# BAD
def get_tools(self):
    return [
        {"function": {"name": "read_file", ...}},
        {"function": {"name": "run_command", ...}}
    ]
```

### ❌ Brittle Context Parsing
```python
# BAD
def on_round_start(self, agent, round_number, context):
    goal_msg = context[1]  # Assumes structure
```

### ✅ Good Patterns
```python
# GOOD - Use helper
def on_round_start(self, agent, round_number, context):
    return self.inject_user_message_after_system(context, "Message")
```

---

## Examples

See:
- **Existing behaviors** in `behaviors/` for real-world examples
- **Anti-patterns doc** at `docs/templates/behavior_antipatterns.md`
- **Test suite** in `tests/test_behavior_independence.py`

---

## Questions?

- Check SELF_EXTENSIBILITY_PLAN.md for architecture details
- Review existing behaviors for patterns
- Run validators to catch issues early
- Write tests as you build

Happy extending!
