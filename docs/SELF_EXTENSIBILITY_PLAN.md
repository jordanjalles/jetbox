# Self-Extensibility Architecture Plan

**Status**: Phase 1 Complete ✅ (6/6 deliverables)
**Created**: 2025-11-06
**Last Updated**: 2025-11-07
**Author**: Claude (via ultrathink)

## Vision

Enable Jetbox agents to autonomously create new behaviors and agent configurations, making the system self-extensible while maintaining safety, composability, and elegance.

**End-to-end scenario:**
```
User → Orchestrator: "I need JSON schema validation for API responses"
Orchestrator → MetaProgrammer: delegate_meta_task(...)
MetaProgrammer:
  1. Reads CreateBehaviorBehavior spec
  2. Generates behaviors/json_schema_validator.py
  3. Generates tests/test_json_schema_validator.py
  4. Validates syntax, composability, independence
  5. Runs tests in sandbox
  6. Presents to user for review
  7. On approval: installs to production
Orchestrator → TaskExecutor: "Use new validator behavior to validate this API"
```

---

## Part 1: Anatomy of Excellence

### What Makes a Behavior "Jetbox-Native"?

Through deep study of the codebase, I've identified the DNA of composable behaviors:

#### 1. **Single Responsibility**
Every behavior does ONE thing, clearly stated in its docstring.

**Examples from codebase:**
- `DirectoryToolsBehavior`: Provides `list_dir` tool (that's it)
- `ReadFileToolsBehavior`: Provides `read_file` tool (that's it)
- `CompactWhenNearFullBehavior`: Manages context compaction (that's it)

**Anti-pattern:**
```python
# BAD: FileAndCommandBehavior that does both file ops AND command execution
# Violates single responsibility - should be two behaviors
```

#### 2. **Zero Dependencies**
No behavior imports or embeds another behavior's functionality.

**Verified via test:** `test_behavior_independence.py` checks for cross-imports

**Anti-pattern:**
```python
# BAD: Importing another behavior
from behaviors.file_tools import ReadFileToolsBehavior

class MyBehavior(AgentBehavior):
    def __init__(self):
        self.file_tools = ReadFileToolsBehavior()  # ❌ DEPENDENCY
```

**Correct pattern:**
```python
# GOOD: Agent composes multiple behaviors
behaviors = [
    ReadFileToolsBehavior(),
    MyNewBehavior()  # Works alongside, not dependent on
]
```

#### 3. **Lifecycle Hook Contract**

Behaviors can override these methods (all optional except `get_name`):

```python
class AgentBehavior(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """REQUIRED: Return unique identifier"""

    def get_tools(self) -> list[dict]:
        """Optional: OpenAI function call schemas"""

    def dispatch_tool(self, agent: "BaseAgent", tool_name: str, args: dict) -> dict:
        """Optional: Handle tool execution (agent-first signature)"""

    def get_instructions(self) -> str:
        """Optional: Add to system prompt"""

    # Lifecycle events (all optional, chronological order)
    def on_goal_start(self, agent: "BaseAgent", goal: str): pass
    def on_initial_context(self, agent: "BaseAgent", context: list[dict]) -> list[dict]:
        """Called ONCE for first-time context setup"""
        return context
    def on_round_start(self, agent: "BaseAgent", round_number: int, context: list[dict]) -> list[dict]:
        """Called EVERY round for dynamic context modifications"""
        return context
    def on_llm_response(self, agent: "BaseAgent", response: dict): pass
    def on_tool_call(self, agent: "BaseAgent", tool_name: str, args: dict, result: dict): pass
    def on_round_end(self, agent: "BaseAgent", round_number: int): pass
    def on_timeout(self, agent: "BaseAgent", elapsed_seconds: float): pass
    def on_goal_complete(self, agent: "BaseAgent", success: bool, summary: str): pass
```

**When to use each hook:**
- **Static context injection**: Use `on_initial_context()` (called ONCE - for goal, tools, notes)
- **Dynamic context injection**: Use `on_round_start()` (called EVERY round - for warnings, prompts)
- **Tools**: Use `get_tools()` + `dispatch_tool()` with agent-first signature
- **Tracking**: Use event handlers (`on_tool_call`, `on_round_end`)
- **Initialization**: Use `on_goal_start()`
- **Cleanup**: Use `on_goal_complete()`

**Key principle: Agent-first signatures** - All lifecycle methods receive `agent` as first parameter,
providing direct access to `agent.workspace`, `agent.state`, etc. No more `**kwargs` soup!

#### 4. **Parameter Invention Tolerance**

Tools gracefully handle parameter invention by LLMs:

```python
def dispatch_tool(self, agent, tool_name, args):
    if tool_name == "my_tool":
        # Warn about unsupported params but don't crash
        supported = {"path", "content"}
        unsupported = set(args.keys()) - supported
        if unsupported:
            print(f"[{self.get_name()}] Ignoring parameters: {unsupported}")

        # Continue with supported params
        return self._do_work(args.get("path"), args.get("content"))
```

**From codebase:** `DirectoryToolsBehavior` demonstrates this pattern.

#### 5. **Workspace Awareness**

Behaviors access workspace directly through the `agent` parameter:

```python
def dispatch_tool(self, agent, tool_name, args):
    if tool_name == "read_file":
        # Access workspace directly from agent
        workspace = agent.workspace

        # Resolve paths through workspace_manager
        if hasattr(agent, 'workspace_manager') and agent.workspace_manager:
            resolved_path = agent.workspace_manager.resolve_path(args["path"])
        else:
            resolved_path = workspace / args["path"]

        return self._read_file(resolved_path)
```

**Key insight:** Agent-first signature provides direct access to workspace state without `**kwargs` soup.

---

## Part 2: Agent Configuration Patterns

### Anatomy of Agent Configs

#### Structure (from task_executor_config.yaml):

```yaml
# Agent metadata
role: "Code task executor"  # Human-readable role

# Blurb for parent agents (delegation)
blurb: |
  TaskExecutor handles focused implementation work...
  Best for implementation work, bug fixes, feature development.

# Delegation tool schema (how others call this agent)
delegation_tool:
  name: "delegate_to_executor"
  description: "Delegate a coding task to TaskExecutor"
  parameters:
    task_description:
      type: string
      description: "Clear description of the task"
      required: true

# System prompt (tool-focused instructions)
system_prompt: |
  You are a local coding agent that helps build software.

  Guidelines:
  - ALWAYS use tools - never just respond with text
  - Be concise and focused on completing the goal

  # Tool documentation is dynamically generated

# Behaviors to load (composable capabilities)
behaviors:
  - type: ChatbotBehavior
  - type: CompactWhenNearFullBehavior
  - type: DirectoryToolsBehavior
  - type: ReadFileToolsBehavior
  - type: WriteFileToolsBehavior
  - type: CommandToolsBehavior
  - type: ServerToolsBehavior
  - type: LoopDetectionBehavior
  - type: WorkspaceTaskNotesBehavior
```

#### Delegation Relationships (agents.yaml):

```yaml
agents:
  orchestrator:
    class: OrchestratorAgent
    can_delegate_to:
      - architect       # Can consult for design
      - task_executor   # Can delegate implementation

  architect:
    class: ArchitectAgent
    can_delegate_to: []  # Terminal consultant

  task_executor:
    class: TaskExecutorAgent
    can_delegate_to: []  # Terminal executor
```

**Key insight:** Delegation is a directed acyclic graph (DAG). Cycles = infinite loops.

---

## Part 3: Testing Patterns

### Behavior Tests

**Pattern:** Isolated unit tests verifying single responsibility.

```python
class TestMyBehavior:
    def test_get_name(self):
        behavior = MyBehavior()
        assert behavior.get_name() == "my_behavior"

    def test_tool_schema(self):
        behavior = MyBehavior()
        tools = behavior.get_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "my_tool"

    def test_tool_dispatch(self):
        behavior = MyBehavior()
        mock_agent = Mock()
        result = behavior.dispatch_tool(mock_agent, "my_tool", {"arg": "value"})
        assert "result" in result

    def test_no_cross_behavior_imports(self):
        # Use AST to check source file doesn't import other behaviors
        source = inspect.getsource(MyBehavior)
        tree = ast.parse(source)
        # ... verify no imports of other behaviors
```

**From codebase:** `test_behavior_independence.py` (line 41-80)

### Agent Integration Tests

```python
def test_agent_with_new_behavior(temp_workspace):
    agent = TaskExecutorAgent(
        workspace=temp_workspace,
        goal="Test goal",
        config_file="task_executor_config.yaml",
        max_rounds=10,
        timeout=120
    )

    # Verify behavior loaded
    behavior_names = [b.get_name() for b in agent.behaviors]
    assert "my_behavior" in behavior_names

    # Run and verify result
    result = agent.run()
    assert result.success
```

**From codebase:** `test_individual_agents_with_behaviors.py` (line 38-86)

---

## Part 4: The Self-Extensibility System

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Request                         │
│   "I need a behavior that validates JSON schemas"       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  OrchestratorAgent                       │
│  - Receives request                                      │
│  - Determines: meta-programming task                     │
│  - Delegates to MetaProgrammerAgent                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               MetaProgrammerAgent                        │
│  Behaviors:                                              │
│    - CreateBehaviorBehavior                              │
│    - CreateAgentBehavior                                 │
│    - ValidationBehavior                                  │
│    - SandboxTestBehavior                                 │
│  Workflow:                                               │
│    1. Read spec/template                                 │
│    2. Generate code                                      │
│    3. Validate (syntax, composability, tests)            │
│    4. Sandbox test                                       │
│    5. Human review (if enabled)                          │
│    6. Install to production                              │
└─────────────────────────────────────────────────────────┘
```

### Components to Build

#### 1. **CreateBehaviorBehavior**

**Purpose:** Enables agents to generate new behavior modules.

**Tools provided:**
```python
{
    "name": "create_behavior",
    "description": "Generate a new behavior module",
    "parameters": {
        "behavior_name": str,       # e.g., "json_schema_validator"
        "description": str,         # What it does
        "tools": list[dict],        # Tool schemas to provide
        "lifecycle_hooks": dict,    # Which hooks to implement
        "safety_mode": str          # "strict" | "review" | "auto"
    }
}
```

**Workflow:**
1. Load template from `docs/templates/behavior_template.py`
2. Generate code using LLM with template + requirements
3. Validate generated code (syntax, composability, independence)
4. Generate test file using `docs/templates/behavior_test_template.py`
5. Run tests in sandbox workspace
6. If `safety_mode == "strict"` or `"review"`: present to user
7. On approval: install to `behaviors/` and `tests/`

**Safety checks:**
- ✓ Python syntax valid
- ✓ Inherits from `AgentBehavior`
- ✓ Implements required `get_name()`
- ✓ No cross-behavior imports (via AST check)
- ✓ Tool schemas well-formed (OpenAI spec)
- ✓ Tests pass in sandbox

#### 2. **CreateAgentBehavior**

**Purpose:** Enables agents to generate new agent configurations.

**Tools provided:**
```python
{
    "name": "create_agent",
    "description": "Generate a new agent configuration",
    "parameters": {
        "agent_name": str,          # e.g., "data_analyst"
        "role": str,                # Human-readable role
        "system_prompt": str,       # Instructions for agent
        "behaviors": list[str],     # Behavior types to load
        "can_delegate_to": list[str],  # Other agents (optional)
        "safety_mode": str          # "strict" | "review" | "auto"
    }
}
```

**Workflow:**
1. Load template from `docs/templates/agent_config_template.yaml`
2. Generate config using LLM with template + requirements
3. Validate config (YAML syntax, behavior references valid, no cycles)
4. Generate agent class from template (if custom logic needed)
5. Update `agents.yaml` with delegation relationships
6. If `safety_mode == "strict"` or `"review"`: present to user
7. On approval: install to root directory

**Safety checks:**
- ✓ YAML syntax valid
- ✓ All referenced behaviors exist
- ✓ Delegation relationships don't create cycles (DAG check)
- ✓ System prompt is tool-focused (not conversational)
- ✓ Blurb is concise (3-5 sentences)
- ✓ Agent class can instantiate successfully

#### 3. **ValidationBehavior**

**Purpose:** Provides validation tools for generated code.

**Tools:**
- `validate_python_syntax(code: str) -> dict`
- `validate_behavior_independence(file_path: str) -> dict`
- `validate_tool_schema(tool: dict) -> dict`
- `validate_agent_dag(agents_config: dict) -> dict`

**Used by:** CreateBehaviorBehavior and CreateAgentBehavior

#### 4. **SandboxTestBehavior**

**Purpose:** Runs generated code in isolated sandbox environment.

**Tools:**
- `run_behavior_tests(behavior_name: str, test_file: str) -> dict`
- `run_agent_sanity_check(agent_name: str) -> dict`

**Sandbox characteristics:**
- Isolated workspace (temp directory)
- Limited tool access (no network, restricted filesystem)
- Short timeout (30 seconds)
- Monitors resource usage

#### 5. **MetaProgrammerAgent**

**New agent configuration:** `meta_programmer_config.yaml`

**Role:** "Meta-programmer for creating behaviors and agents"

**Behaviors:**
```yaml
behaviors:
  - type: ChatbotBehavior
  - type: CompactWhenNearFullBehavior
  - type: CreateBehaviorBehavior
  - type: CreateAgentBehavior
  - type: ValidationBehavior
  - type: SandboxTestBehavior
  - type: ReadFileToolsBehavior    # Read templates
  - type: WriteFileToolsBehavior   # Write generated code
  - type: DirectoryToolsBehavior   # Navigate codebase
  - type: CommandToolsBehavior     # Run tests
```

**System prompt highlights:**
```yaml
system_prompt: |
  You are a meta-programmer agent that creates new behaviors and agents
  for the Jetbox system.

  CRITICAL PRINCIPLES:
  1. Single Responsibility - Each behavior does ONE thing
  2. Zero Dependencies - No cross-behavior imports
  3. Composability - Behaviors work independently
  4. Safety First - Always validate before installing

  When creating behaviors:
  - Use behavior templates as starting point
  - Implement only necessary lifecycle hooks
  - Provide clear tool schemas
  - Write comprehensive tests

  When creating agents:
  - Write tool-focused system prompts
  - Choose minimal behavior set needed
  - Avoid delegation cycles
  - Provide clear role and blurb
```

**Delegation:**
```yaml
# In agents.yaml
meta_programmer:
  class: MetaProgrammerAgent
  can_delegate_to:
    - task_executor  # For running complex validations

orchestrator:
  can_delegate_to:
    - architect
    - task_executor
    - meta_programmer  # NEW: Can delegate meta-programming
```

---

## Part 5: Templates & Specifications

### Template Philosophy

**Key insight:** The best spec is a worked example, not abstract rules.

Templates encode patterns through:
1. **Annotated minimal examples** - Show the simplest valid implementation
2. **Annotated full-featured examples** - Show all available hooks
3. **Anti-patterns** - Show what NOT to do
4. **Inline comments** - Explain the "why" behind decisions

### Template: Minimal Behavior

**File:** `docs/templates/behavior_minimal_template.py`

```python
"""
{BEHAVIOR_NAME}Behavior - {ONE_SENTENCE_DESCRIPTION}

Features:
- {FEATURE_1}
- {FEATURE_2}

This is a MINIMAL behavior showing the simplest possible implementation.
"""

from typing import Any
from behaviors.base import AgentBehavior


class {BEHAVIOR_CLASS_NAME}Behavior(AgentBehavior):
    """
    {DETAILED_DESCRIPTION}

    This behavior provides: {WHAT_IT_PROVIDES}
    This behavior does NOT: {WHAT_IT_DOES_NOT_DO}
    """

    def __init__(self, **kwargs):
        """
        Initialize {BEHAVIOR_NAME} behavior.

        Args:
            **kwargs: Additional parameters (for flexibility)
        """
        # Initialize any state here
        pass

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "{BEHAVIOR_NAME}"

    # Override ONLY the hooks you need below
    # Delete unused methods - keep it minimal
```

### Template: Tool-Providing Behavior

**File:** `docs/templates/behavior_with_tools_template.py`

```python
"""
{BEHAVIOR_NAME}Behavior - {ONE_SENTENCE_DESCRIPTION}

Provides tools:
- {TOOL_1_NAME}: {TOOL_1_DESCRIPTION}
- {TOOL_2_NAME}: {TOOL_2_DESCRIPTION}
"""

from typing import Any
from behaviors.base import AgentBehavior


class {BEHAVIOR_CLASS_NAME}Behavior(AgentBehavior):
    """
    Provides {TOOL_CATEGORY} tools for agent use.
    """

    def __init__(self, workspace_manager=None, **kwargs):
        """
        Initialize behavior.

        Args:
            workspace_manager: Optional WorkspaceManager for path resolution
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager

    def get_name(self) -> str:
        return "{BEHAVIOR_NAME}"

    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions in OpenAI function call format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "{TOOL_NAME}",
                    "description": "{TOOL_DESCRIPTION}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "{PARAM_NAME}": {
                                "type": "{PARAM_TYPE}",
                                "description": "{PARAM_DESCRIPTION}"
                            }
                        },
                        "required": ["{REQUIRED_PARAM}"]
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle tool execution.

        Args:
            agent: Agent instance (access agent.workspace, agent.state, etc.)
            tool_name: Tool being called
            args: Tool arguments from LLM

        Returns:
            Tool result dict ({"result": ..., "success": True} or {"error": ...})
        """
        if tool_name == "{TOOL_NAME}":
            return self._execute_{TOOL_NAME}(agent, args)
        else:
            # Fall through to parent for unknown tools
            return super().dispatch_tool(agent, tool_name, args)

    def _execute_{TOOL_NAME}(
        self,
        agent: Any,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute {TOOL_NAME} tool.

        Args:
            agent: Agent instance (for workspace access)
            args: Tool arguments

        Returns:
            Result dict
        """
        try:
            # Get parameters with defaults
            param1 = args.get("{PARAM_NAME}", "{DEFAULT_VALUE}")

            # Warn about unsupported parameters (parameter invention tolerance)
            supported = {"{PARAM_NAME}", "{OTHER_PARAM}"}
            unsupported = set(args.keys()) - supported
            if unsupported:
                print(f"[{self.get_name()}] Ignoring parameters: {unsupported}")

            # Access workspace if needed
            workspace = agent.workspace if hasattr(agent, 'workspace') else None

            # DO THE WORK HERE
            result = self._do_work(param1, workspace)

            return {"result": result, "success": True}

        except Exception as e:
            return {"error": str(e)}

    def _do_work(self, param1, workspace):
        """Core logic separated for testability."""
        # Implement actual functionality
        pass
```

### Template: Context-Enhancing Behavior

**File:** `docs/templates/behavior_context_enhancement_template.py`

```python
"""
{BEHAVIOR_NAME}Behavior - {ONE_SENTENCE_DESCRIPTION}

Enhances context by: {WHAT_IT_INJECTS}
"""

from typing import Any
from behaviors.base import AgentBehavior


class {BEHAVIOR_CLASS_NAME}Behavior(AgentBehavior):
    """
    Behavior that enhances context with {WHAT_INFORMATION}.
    """

    def __init__(self, **kwargs):
        self.state = {}  # Track any state needed

    def get_name(self) -> str:
        return "{BEHAVIOR_NAME}"

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject {WHAT_INFORMATION} into initial context (called ONCE).

        Use this for static content that doesn't change between rounds.

        Args:
            agent: Agent instance (access agent.goal, agent.workspace, etc.)
            context: Initial context (system prompt only)

        Returns:
            Modified context with injected information
        """
        # Extract info from agent
        goal = agent.goal if hasattr(agent, 'goal') else ''

        # Build message to inject
        message = f"{CONTEXT_HEADER}: {goal}"

        # Use helper to inject after system prompt
        return self.inject_user_message_after_system(context, message)

    def on_round_start(
        self,
        agent: Any,
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject dynamic {WHAT_INFORMATION} into context (called EVERY round).

        Use this for dynamic content that changes between rounds (warnings, prompts).

        Args:
            agent: Agent instance
            round_number: Current round number
            context: Current context

        Returns:
            Modified context
        """
        # Example: Inject dynamic warning based on state
        if self.state.get('should_warn'):
            warning = "⚠️ {DYNAMIC_WARNING}"
            context = self.inject_user_message_after_system(context, warning)

        return context
```

### Template: Behavior Test

**File:** `docs/templates/behavior_test_template.py`

```python
"""
Tests for {BEHAVIOR_CLASS_NAME}Behavior.

Tests:
- Behavior identifier
- Tool schemas (if applicable)
- Tool dispatch (if applicable)
- Context enhancement (if applicable)
- Event handlers (if applicable)
- No cross-behavior dependencies
"""

import pytest
from unittest.mock import Mock
from behaviors.{BEHAVIOR_MODULE} import {BEHAVIOR_CLASS_NAME}Behavior


class Test{BEHAVIOR_CLASS_NAME}Behavior:
    """Test suite for {BEHAVIOR_CLASS_NAME}Behavior."""

    def test_get_name(self):
        """Behavior returns correct identifier."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        assert behavior.get_name() == "{BEHAVIOR_NAME}"

    def test_initialization(self):
        """Behavior initializes without errors."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        assert behavior is not None

    # ADD TESTS BELOW BASED ON WHAT THE BEHAVIOR DOES

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_tool_schema(self):
        """Tool schemas are well-formed."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        tools = behavior.get_tools()

        assert len(tools) > 0
        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_tool_dispatch_success(self):
        """Tool dispatch returns expected result."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        mock_agent = Mock()

        result = behavior.dispatch_tool(
            agent=mock_agent,
            tool_name="{TOOL_NAME}",
            args={"{PARAM_NAME}": "{TEST_VALUE}"}
        )

        assert "result" in result or "success" in result

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_tool_dispatch_unknown_tool(self):
        """Unknown tools raise NotImplementedError."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        mock_agent = Mock()

        with pytest.raises(NotImplementedError):
            behavior.dispatch_tool(
                agent=mock_agent,
                tool_name="unknown_tool",
                args={}
            )

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_initial_context_injection(self):
        """on_initial_context injects expected information (called ONCE)."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        mock_agent = Mock()
        mock_agent.goal = "Test goal"

        context = [
            {"role": "system", "content": "System prompt"}
        ]

        enhanced = behavior.on_initial_context(agent=mock_agent, context=context)

        # Verify injection
        assert len(enhanced) > len(context)
        assert any("Test goal" in msg.get("content", "") for msg in enhanced)

    @pytest.mark.skipif(True, reason="Template placeholder")
    def test_round_start_context_injection(self):
        """on_round_start injects dynamic information (called EVERY round)."""
        behavior = {BEHAVIOR_CLASS_NAME}Behavior()
        mock_agent = Mock()

        context = [
            {"role": "system", "content": "System prompt"}
        ]

        enhanced = behavior.on_round_start(
            agent=mock_agent,
            round_number=1,
            context=context
        )

        # Verify injection or no modification depending on behavior
        assert isinstance(enhanced, list)
```

### Template: Agent Configuration

**File:** `docs/templates/agent_config_template.yaml`

```yaml
# {AGENT_NAME} Agent Configuration
#
# This file defines the {AGENT_NAME} agent's role, system prompt,
# and behavior composition.

# Agent metadata
role: "{ROLE_DESCRIPTION}"

# Blurb for parent agents (3-5 sentences describing specialty/purpose)
blurb: |
  {AGENT_NAME} {WHAT_IT_DOES}.
  {WHEN_TO_USE_IT}.
  {KEY_CAPABILITIES}.
  Best for {USE_CASES}.

# Delegation tool configuration (how other agents call this agent)
delegation_tool:
  name: "{DELEGATION_TOOL_NAME}"
  description: "{TOOL_DESCRIPTION}"
  parameters:
    {PARAM_NAME}:
      type: {PARAM_TYPE}
      description: "{PARAM_DESCRIPTION}"
      required: {true/false}

# System prompt (tool-focused instructions)
system_prompt: |
  You are {AGENT_DESCRIPTION}.

  Guidelines:
  - ALWAYS use tools - never just respond with text
  - {GUIDELINE_1}
  - {GUIDELINE_2}

  # Tool documentation is dynamically generated based on behaviors

# Behaviors to load (composable capabilities)
behaviors:
  # Execution mode
  - type: ChatbotBehavior

  # Context management
  - type: CompactWhenNearFullBehavior

  # Tool behaviors (choose appropriate ones)
  - type: DirectoryToolsBehavior   # Directory navigation
  - type: ReadFileToolsBehavior    # File reading
  - type: WriteFileToolsBehavior   # File writing
  - type: CommandToolsBehavior     # Command execution

  # Utility behaviors
  - type: LoopDetectionBehavior
  - type: WorkspaceTaskNotesBehavior

  # Custom behaviors (if any)
  # - type: {CUSTOM_BEHAVIOR}
```

### Template: Anti-Patterns Document

**File:** `docs/templates/behavior_antipatterns.md`

```markdown
# Behavior Anti-Patterns

This document shows common mistakes when creating behaviors.

## ❌ Anti-Pattern 1: Cross-Behavior Dependencies

**Bad:**
```python
from behaviors.file_tools import ReadFileToolsBehavior

class MyBehavior(AgentBehavior):
    def __init__(self):
        self.file_tools = ReadFileToolsBehavior()  # ❌

    def my_method(self):
        self.file_tools.dispatch_tool("read_file", {...})  # ❌
```

**Why it's bad:** Behaviors must be independent. This creates coupling.

**Good:**
```python
class MyBehavior(AgentBehavior):
    def dispatch_tool(self, tool_name, args, **kwargs):
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
  Just tell me what you'd like to do and I'll try my best! 😊
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
```

---

## Part 6: Safety Mechanisms

### Multi-Tier Safety Model

Self-modification is risky. We implement defense-in-depth:

#### Tier 1: Validation (Automated)

**Syntax validation:**
```python
def validate_python_syntax(code: str) -> dict:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return {"valid": True}
    except SyntaxError as e:
        return {"valid": False, "error": str(e)}
```

**Composability validation:**
```python
def validate_behavior_independence(file_path: str) -> dict:
    """Check for cross-behavior imports."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith('behaviors.'):
                    # Check if importing another behavior (not base)
                    module = alias.name.split('.')[-1]
                    if module != 'base' and module in KNOWN_BEHAVIORS:
                        return {
                            "valid": False,
                            "error": f"Cross-behavior import: {alias.name}"
                        }

    return {"valid": True}
```

**Tool schema validation:**
```python
def validate_tool_schema(tool: dict) -> dict:
    """Validate tool follows OpenAI function calling spec."""
    required_keys = ["type", "function"]
    function_keys = ["name", "description", "parameters"]

    if not all(k in tool for k in required_keys):
        return {"valid": False, "error": "Missing required keys"}

    if not all(k in tool["function"] for k in function_keys):
        return {"valid": False, "error": "Invalid function schema"}

    # Validate parameters structure
    params = tool["function"]["parameters"]
    if params.get("type") != "object":
        return {"valid": False, "error": "Parameters must be object type"}

    return {"valid": True}
```

**DAG validation (no delegation cycles):**
```python
def validate_agent_dag(agents_config: dict) -> dict:
    """Ensure delegation relationships form a DAG (no cycles)."""
    # Build adjacency list
    graph = {}
    for agent_name, config in agents_config["agents"].items():
        graph[agent_name] = config.get("can_delegate_to", [])

    # DFS cycle detection
    def has_cycle(node, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    visited = set()
    for node in graph:
        if node not in visited:
            if has_cycle(node, visited, set()):
                return {"valid": False, "error": f"Cycle detected involving {node}"}

    return {"valid": True}
```

#### Tier 2: Sandbox Testing (Automated)

**Isolated test environment:**
```python
def run_behavior_tests_in_sandbox(
    behavior_name: str,
    test_file: str,
    timeout: int = 30
) -> dict:
    """
    Run behavior tests in isolated sandbox.

    Sandbox characteristics:
    - Temporary workspace
    - Limited tool access
    - Resource monitoring
    - Network isolation (if possible)
    - Short timeout
    """
    sandbox_dir = tempfile.mkdtemp(prefix=f"sandbox_{behavior_name}_")

    try:
        # Copy behavior and test to sandbox
        shutil.copy(f"behaviors/{behavior_name}.py", sandbox_dir)
        shutil.copy(test_file, sandbox_dir)

        # Run pytest in sandbox with timeout
        result = subprocess.run(
            ["pytest", "-v", os.path.basename(test_file)],
            cwd=sandbox_dir,
            timeout=timeout,
            capture_output=True,
            text=True
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Tests timed out (30s limit)"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        # Cleanup sandbox
        shutil.rmtree(sandbox_dir, ignore_errors=True)
```

#### Tier 3: Human Review (Optional)

**Safety modes:**

1. **Auto mode** (fastest, lowest safety):
   - Generate → Validate → Sandbox test → Install
   - Use for: Well-defined, low-risk additions
   - Skip: Human review

2. **Review mode** (balanced):
   - Generate → Validate → Sandbox test → Present to user → Wait for approval
   - Use for: Most cases
   - User sees: Generated code, validation results, test results
   - User can: Approve, reject, or request modifications

3. **Strict mode** (slowest, highest safety):
   - Generate → Validate → Sandbox test → Full audit → Present to user
   - Additional checks:
     - Security audit (check for eval(), exec(), subprocess with user input)
     - Resource usage analysis (memory leaks, file handles)
     - Code quality metrics (complexity, test coverage)
   - Use for: Production systems, public deployment

**Review interface:**
```python
def present_for_review(
    generated_code: str,
    validation_results: dict,
    test_results: dict
) -> bool:
    """
    Present generated code to user for approval.

    Returns:
        True if approved, False if rejected
    """
    print("="*80)
    print("Generated Code Review Required")
    print("="*80)

    print("\n📝 Generated Code:")
    print(generated_code)

    print("\n✅ Validation Results:")
    for check, result in validation_results.items():
        status = "✓" if result["valid"] else "✗"
        print(f"  {status} {check}: {result.get('error', 'PASS')}")

    print("\n🧪 Test Results:")
    if test_results["success"]:
        print("  ✓ All tests passed")
    else:
        print(f"  ✗ Tests failed: {test_results.get('error')}")

    print("\n" + "="*80)
    response = input("Approve for installation? (yes/no): ").strip().lower()
    return response in ["yes", "y"]
```

#### Tier 4: Rollback Capability

**Backup before install:**
```python
def install_with_rollback(
    source_file: str,
    dest_file: str,
    backup_dir: Path = Path(".agent_generated/backups")
) -> bool:
    """
    Install file with rollback capability.

    Creates backup of existing file (if any) before overwriting.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)

    dest_path = Path(dest_file)

    # Backup existing file
    if dest_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{dest_path.name}.{timestamp}.backup"
        shutil.copy(dest_path, backup_file)
        print(f"[install] Backed up existing file to: {backup_file}")

    # Install new file
    try:
        shutil.copy(source_file, dest_path)
        print(f"[install] Installed: {dest_path}")
        return True
    except Exception as e:
        print(f"[install] Error: {e}")
        # Restore from backup if install failed
        if dest_path.exists():
            shutil.copy(backup_file, dest_path)
            print(f"[install] Restored from backup")
        return False
```

**Manual rollback:**
```bash
# User can manually rollback
ls .agent_generated/backups/
cp .agent_generated/backups/my_behavior.py.20251106_120000.backup behaviors/my_behavior.py
```

#### Tier 5: Dry-Run Mode

**Generate but don't install:**
```python
def create_behavior_dryrun(
    behavior_name: str,
    description: str,
    tools: list[dict],
    **kwargs
) -> dict:
    """
    Generate behavior code but don't install.

    Saves to staging area: .agent_generated/staging/
    User can review and manually install.
    """
    staging_dir = Path(".agent_generated/staging")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Generate code
    code = generate_behavior_code(behavior_name, description, tools)

    # Validate
    validation = validate_behavior_code(code)

    # Generate tests
    test_code = generate_behavior_tests(behavior_name, tools)

    # Save to staging
    behavior_file = staging_dir / f"{behavior_name}.py"
    test_file = staging_dir / f"test_{behavior_name}.py"

    behavior_file.write_text(code)
    test_file.write_text(test_code)

    return {
        "success": True,
        "message": f"Generated (not installed). Review in: {staging_dir}",
        "files": {
            "behavior": str(behavior_file),
            "test": str(test_file)
        },
        "validation": validation,
        "next_steps": [
            f"1. Review: cat {behavior_file}",
            f"2. Test: pytest {test_file}",
            f"3. Install: mv {behavior_file} behaviors/",
            f"4. Install test: mv {test_file} tests/"
        ]
    }
```

---

## Part 7: Implementation Roadmap

### Phase 1: Foundations (Week 1)

**Deliverables:**
1. Template files:
   - `docs/templates/behavior_minimal_template.py`
   - `docs/templates/behavior_with_tools_template.py`
   - `docs/templates/behavior_context_enhancement_template.py`
   - `docs/templates/behavior_test_template.py`
   - `docs/templates/agent_config_template.yaml`
   - `docs/templates/behavior_antipatterns.md`

2. Validation utilities:
   - `utils/behavior_validator.py`:
     - `validate_python_syntax()`
     - `validate_behavior_independence()`
     - `validate_tool_schema()`
   - `utils/agent_validator.py`:
     - `validate_agent_dag()`
     - `validate_yaml_syntax()`

3. Documentation:
   - This plan document
   - Template usage guide

**Success criteria:**
- All templates exist and are well-commented
- Validation utilities have unit tests
- Documentation is clear and comprehensive

### Phase 2: Validation & Sandbox (Week 2)

**Deliverables:**
1. `ValidationBehavior`:
   - Implements validation tools
   - Integrates with validator utilities
   - Provides clear error messages

2. `SandboxTestBehavior`:
   - Creates isolated test environments
   - Runs tests with timeout
   - Reports results clearly

3. Staging infrastructure:
   - `.agent_generated/staging/` directory
   - `.agent_generated/backups/` directory
   - Installation utilities with rollback

**Success criteria:**
- Can validate arbitrary Python behavior code
- Can run tests in sandbox without affecting system
- Installation with rollback works correctly

### Phase 3: CreateBehaviorBehavior (Week 3)

**Deliverables:**
1. `CreateBehaviorBehavior` implementation:
   - Tool: `create_behavior()`
   - Workflow: Generate → Validate → Test → Review → Install
   - Safety modes: auto, review, strict, dryrun

2. LLM integration:
   - Prompt engineering for code generation
   - Template-based generation
   - Error recovery (retry with feedback)

3. Testing:
   - Unit tests for behavior
   - Integration test: Generate simple behavior end-to-end
   - Validation that generated code passes independence checks

**Success criteria:**
- Can generate a simple tool-providing behavior
- Generated code passes all validations
- Generated tests pass in sandbox
- Manual review shows high-quality code

### Phase 4: CreateAgentBehavior (Week 4)

**Deliverables:**
1. `CreateAgentBehavior` implementation:
   - Tool: `create_agent()`
   - Agent config generation
   - DAG validation for delegation
   - Integration with agents.yaml

2. Agent instantiation testing:
   - Verify generated agent can instantiate
   - Verify behaviors load correctly
   - Verify tools are available

3. Testing:
   - Unit tests for behavior
   - Integration test: Generate simple agent end-to-end

**Success criteria:**
- Can generate agent config that loads successfully
- Can update agents.yaml without creating cycles
- Generated agent can execute simple tasks

### Phase 5: MetaProgrammerAgent (Week 5)

**Deliverables:**
1. `MetaProgrammerAgent` configuration:
   - `meta_programmer_config.yaml`
   - System prompt with behavior creation guidelines
   - Behavior composition

2. Agent class (if needed):
   - `meta_programmer_agent.py`
   - Custom dispatch logic (if any)

3. Integration with orchestrator:
   - Update `orchestrator_config.yaml` with delegation tool
   - Update `agents.yaml` relationships

**Success criteria:**
- Orchestrator can delegate to MetaProgrammer
- MetaProgrammer can create behaviors via natural language
- MetaProgrammer follows safety protocols

### Phase 6: End-to-End Testing & Refinement (Week 6)

**Deliverables:**
1. Comprehensive test suite:
   - Test full workflow: User request → Generated behavior → Installation
   - Test all safety modes
   - Test error cases (invalid code, failing tests, etc.)

2. Documentation:
   - User guide for creating behaviors
   - Developer guide for extending templates
   - Safety best practices

3. Examples:
   - Example: Create JSON schema validator behavior
   - Example: Create data analyst agent
   - Example: Create custom API client behavior

**Success criteria:**
- Can execute full user scenario from vision section
- Generated code quality matches hand-written code
- Safety mechanisms prevent common mistakes
- Documentation is clear and complete

---

## Part 8: Success Metrics

### Code Quality Metrics

**Generated behaviors must meet these standards:**

1. **Composability Score: 100%**
   - ✓ No cross-behavior imports
   - ✓ No hardcoded agent knowledge
   - ✓ Works in any behavior composition

2. **Test Coverage: >80%**
   - All tool methods tested
   - Edge cases covered
   - Error conditions handled

3. **Documentation Score: >90%**
   - Docstrings for all public methods
   - Type hints for all parameters
   - Clear module-level documentation

4. **Maintainability: Pass**
   - Cyclomatic complexity < 10 per method
   - No duplicate code
   - Clear variable names

### Safety Metrics

**Safety mechanisms must achieve:**

1. **Validation Accuracy: 100%**
   - No false negatives (invalid code marked valid)
   - False positives acceptable (valid code marked invalid) - prefer safe

2. **Sandbox Isolation: 100%**
   - No sandbox escape possible
   - No contamination of production environment

3. **Human Review Rate:**
   - Auto mode: 0% (no review)
   - Review mode: 100% (user sees all changes)
   - Strict mode: 100% + audit log

### Performance Metrics

**Generation performance:**

1. **Speed:**
   - Simple behavior: <60s end-to-end
   - Complex behavior: <180s end-to-end
   - Agent config: <30s end-to-end

2. **Success Rate:**
   - First attempt: >60%
   - After retry: >90%
   - Human intervention: <5%

### User Experience Metrics

**Usability:**

1. **Clarity:**
   - Error messages actionable (what to fix)
   - Review interface intuitive
   - Next steps always clear

2. **Confidence:**
   - Validation gives confidence in safety
   - Test results show functionality
   - Rollback available if needed

---

## Part 9: Risk Mitigation

### Risk 1: Generated Code Breaks System

**Mitigation:**
- Validation catches syntax errors (Tier 1)
- Sandbox testing catches runtime errors (Tier 2)
- Rollback capability allows recovery (Tier 4)
- Review mode allows human oversight (Tier 3)

**Residual risk: Low**

### Risk 2: Security Vulnerabilities in Generated Code

**Mitigation:**
- Strict mode includes security audit
- Sandbox prevents dangerous operations
- Templates encode security best practices
- Human review can catch issues

**Residual risk: Medium** (requires security expertise in review)

**Recommendation:** Add automated security scanning in strict mode:
- Check for: `eval()`, `exec()`, `__import__()`, `compile()`
- Check for: Unsafe deserialization (pickle, yaml.unsafe_load)
- Check for: Command injection vectors

### Risk 3: Delegation Cycles Create Infinite Loops

**Mitigation:**
- DAG validation prevents cycles (Tier 1)
- Agents.yaml validation runs before install
- Rollback available if cycle introduced

**Residual risk: Very Low**

### Risk 4: Generated Behaviors Violate Composability

**Mitigation:**
- Independence validation checks imports (Tier 1)
- Templates encode composability principles
- Anti-patterns documentation
- Review mode allows human check

**Residual risk: Low**

### Risk 5: Resource Exhaustion (Memory Leaks, File Handles)

**Mitigation:**
- Sandbox has timeout (30s)
- Templates show cleanup patterns
- Strict mode can add resource monitoring

**Residual risk: Medium**

**Recommendation:** Add resource monitoring to strict mode:
- Track file handles opened/closed
- Monitor memory usage during sandbox test
- Warn if resource growth detected

### Risk 6: LLM Generates Poor Quality Code

**Mitigation:**
- Templates provide strong examples
- Validation catches many issues
- Retry with feedback improves output
- Human review can reject poor code

**Residual risk: Medium**

**Recommendation:**
- Iterate on prompts to improve quality
- Consider fine-tuning on high-quality behaviors
- Build quality scoring (complexity, style, patterns)

---

## Part 10: Future Enhancements

### Enhancement 1: Behavior Marketplace

**Vision:** Shared repository of community-created behaviors.

**Features:**
- Publish behaviors to central registry
- Search/browse behaviors by category
- Install behaviors from marketplace
- Rating/review system
- Automatic security scanning

**Implementation:**
- `behaviors/marketplace/` directory structure
- Behavior manifest with metadata
- Installation tools
- Verification system

### Enhancement 2: Behavior Composition Analysis

**Vision:** Detect which behaviors work well together.

**Features:**
- Analyze behavior combinations in successful runs
- Recommend behaviors for new agent types
- Warn about conflicting behaviors
- Suggest optimal behavior ordering

**Implementation:**
- Track agent performance by behavior set
- Statistical analysis of combinations
- Recommendation engine

### Enhancement 3: Automated Behavior Evolution

**Vision:** Behaviors improve themselves over time.

**Features:**
- Track behavior performance (errors, usage)
- Identify improvement opportunities
- Generate patches automatically
- Test and deploy improvements

**Implementation:**
- Performance telemetry
- Error pattern analysis
- Automated patch generation
- A/B testing framework

### Enhancement 4: Visual Behavior Builder

**Vision:** GUI for creating behaviors without coding.

**Features:**
- Drag-and-drop tool designer
- Visual workflow editor
- Parameter configuration forms
- One-click generation

**Implementation:**
- Web-based UI
- Visual → code translation
- Integration with CreateBehaviorBehavior

### Enhancement 5: Behavior Versioning

**Vision:** Track behavior versions and dependencies.

**Features:**
- Semantic versioning for behaviors
- Dependency management (behavior X requires Y v2.0+)
- Migration tools for version upgrades
- Compatibility checking

**Implementation:**
- Version metadata in behaviors
- Dependency resolver
- Migration scripts
- Compatibility matrix

---

## Implementation Status

**Last Updated**: 2025-11-07

### Phase 1 (Foundations) - ✅ COMPLETE

All deliverables completed and tested:

1. **Templates** - ✅ DONE
   - `docs/templates/behavior_minimal_template.py`
   - `docs/templates/behavior_with_tools_template.py`
   - `docs/templates/behavior_context_enhancement_template.py`
   - `docs/templates/behavior_test_template.py`
   - `docs/templates/agent_config_template.yaml`
   - `docs/templates/behavior_antipatterns.md`

2. **Validation Utilities** - ✅ DONE
   - `utils/behavior_validator.py`:
     - `validate_python_syntax()` ✅
     - `validate_behavior_independence()` ✅
     - `validate_tool_schema()` ✅
     - `validate_behavior_class()` ✅
   - `utils/agent_validator.py`:
     - `validate_agent_dag()` ✅
     - `validate_yaml_syntax()` ✅
     - `validate_agent_config()` ✅

3. **ValidationBehavior** - ✅ DONE
   - Wraps all validation utilities as tools
   - Supports `config_file` parameter for direct file validation
   - Returns unwrapped results for consistency

4. **SandboxTestBehavior** - ✅ DONE
   - Creates isolated test environments
   - Runs pytest with timeout
   - Reports structured results

5. **CreateBehaviorBehavior** - ✅ DONE (commit: 5fb8fb9)
   - Full behavior generation workflow
   - LLM-based code generation
   - Test generation with proper mocking
   - Validation and sandbox testing
   - `context_enhancement` parameter for guided generation
   - Safety modes: auto, review, strict, dryrun

6. **CreateAgentBehavior** - ✅ DONE (commit: 04ef202)
   - Full YAML agent config generation
   - `_run_agent_generation_workflow()` implemented
   - `_generate_agent_config()` builds proper YAML structure
   - `_validate_agent_config()` validates generated configs
   - Supports full delegation_tool specification
   - Saves to `.agent_generated/staging/`
   - Returns `{"success": True, "config_file": path}`

### Test Results

**Category 1: Simple Behaviors (Tools Only)**
- ✅ Test 1.1: HTTPRequestBehavior - PASSING
- ✅ Test 1.2: JSONToolsBehavior - PASSING
- ✅ Test 1.3: EnvironmentBehavior - PASSING

**Category 2: Complex Behaviors (State + Lifecycle)**
- ✅ Test 2.1: GitOperationsBehavior - PASSING
- ✅ Test 2.2: DockerBehavior - PASSING (with `context_enhancement`)
- 🔄 Test 2.3: CachingBehavior - IN PROGRESS

**Category 3: Agent Generation**
- ✅ Test 3.2: DocGeneratorAgent - PASSING
- 🔄 Test 3.3: TestGeneratorAgent - IN PROGRESS

### Next Steps

1. **Complete remaining tests**:
   - Test 2.3 (CachingBehavior)
   - Test 3.3 (TestGeneratorAgent)

2. **Phase 2 onwards**:
   - MetaProgrammerAgent configuration
   - End-to-end integration testing
   - Safety mechanism refinement
   - Production deployment procedures

---

## Conclusion

This plan provides a complete blueprint for self-extensibility in Jetbox:

**What we're building:**
- CreateBehaviorBehavior: Enables generating new behaviors
- CreateAgentBehavior: Enables generating new agents
- MetaProgrammerAgent: Orchestrates meta-programming tasks
- Safety mechanisms: Multi-tier validation and review
- Templates: Encode patterns and principles

**Why it's elegant:**
- Behaviors remain composable (no special cases)
- Safety is layered (defense in depth)
- Templates teach by example (not abstract rules)
- System can evolve without human coding

**Why it's safe:**
- Validation catches errors before execution
- Sandbox isolates testing from production
- Human review for high-risk changes
- Rollback available for mistakes

**The magic:**
A behavior that creates behaviors. An agent that creates agents. The system learns to extend itself, following the same composability principles that made it elegant in the first place.

**Next step:** Approve this plan → Begin Phase 1 implementation.

---

*"The best way to predict the future is to invent it. But the best way to invent the future is to create systems that invent themselves."*
