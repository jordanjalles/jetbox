# E2E Test: Bugs Found and Fixes Applied

## Overview

During deep E2E testing of the self-extensibility system (SearchToolsBehavior + AllAgent generation), multiple critical bugs were discovered in LLM-generated code. This document summarizes all issues found and fixes applied.

## Test Scenario

**Goal**: Generate SearchToolsBehavior and AllAgent to test full self-extensibility workflow

**SearchToolsBehavior Specification**:
- Provides `search_tools` tool for dynamic tool discovery
- Removes all tools except itself from initial context
- Adds only relevant tools when queried
- Context enhancement via lifecycle hooks

**AllAgent Specification**:
- Universal agent with all 25 behaviors loaded
- Uses SearchToolsBehavior for dynamic tool filtering

## Bugs Found

### Bug 1: YAML Multi-Line Description Indentation

**File**: `behaviors/create_agent.py:298`

**Symptom**: Generated AllAgent.yaml failed to parse with error:
```
could not find expected ':'
  in ".agent_generated/staging/AllAgent.yaml", line 11, column 1
```

**Root Cause**: Multi-line descriptions in `blurb:` block not properly indented

**Generated Code (BROKEN)**:
```yaml
blurb: |
  Line 1
Line 2  # <-- Not indented, breaks YAML parser
```

**Fix**: Added indentation for all description lines
```python
# Before:
yaml_content = f"""
blurb: |
  {description}
"""

# After:
description_indented = '\n'.join('  ' + line for line in description.split('\n'))
yaml_content = f"""
blurb: |
{description_indented}
"""
```

**Commit**: `a8f95e4`

---

### Bug 2: super() Double-Dispatch in dispatch_tool (CRITICAL)

**File**: `behaviors/SearchToolsBehavior.py:269` (LLM-generated)

**Also Found In**:
- `behaviors/validation.py:269` (existing code)
- `behaviors/templates/behavior_simple_template.py:109`
- `behaviors/templates/behavior_with_tools_template.py:124`

**Symptom**: `super().dispatch_tool()` called for unknown tools

**Generated Code (BROKEN)**:
```python
def dispatch_tool(self, agent, tool_name, args):
    if tool_name == "my_tool":
        return {"result": "ok"}
    else:
        return super().dispatch_tool(agent, tool_name, args)  # BUG!
```

**Root Cause**:
- AgentBehavior.dispatch_tool() is a final implementation that doesn't chain
- Calling super() causes tools to be dispatched twice
- LLM learned this pattern from bad templates

**Impact**: Double-dispatch causes:
- Tools execute twice with same arguments
- Potential infinite loops
- Incorrect result propagation

**Correct Pattern**:
```python
def dispatch_tool(self, agent, tool_name, args):
    if tool_name == "my_tool":
        return {"result": "ok"}
    else:
        # Unknown tool - return error
        # IMPORTANT: Do NOT call super().dispatch_tool() as it causes double-dispatch
        return {"error": f"Unknown tool: {tool_name}"}
```

**Fixes Applied**:
1. Fixed SearchToolsBehavior.py (generated code)
2. Fixed ValidationBehavior.py (existing code)
3. Updated behavior_simple_template.py (template)
4. Updated behavior_with_tools_template.py (template)
5. Added validation rule to catch future occurrences

**Commit**: `bc69c76`

---

### Bug 3: Stale Tools Cache

**File**: `behaviors/SearchToolsBehavior.py:110-122`

**Symptom**: SearchToolsBehavior cached tools at initialization, never refreshed

**Generated Code (BROKEN)**:
```python
def __init__(self, workspace_manager=None, **kwargs):
    self.workspace_manager = workspace_manager
    self._all_tools: List[Dict[str, Any]] = []  # Cached once

def dispatch_tool(self, agent, tool_name, args):
    if tool_name == "search_tools":
        # Use cached tools
        if not self._all_tools:
            self._all_tools = agent.get_tools()
        for tool in self._all_tools:  # Stale cache!
            ...
```

**Root Cause**: Cached tools at first search, never updated

**Impact**:
- Would miss dynamically added tools
- Breaks future extensibility where tools can be added after initialization

**Fix**: Query agent.get_tools() every time search_tools is invoked
```python
def dispatch_tool(self, agent, tool_name, args):
    if tool_name == "search_tools":
        # ALWAYS query current tools (don't rely on cache)
        # This ensures we see tools added after initial context
        current_tools = agent.get_tools()
        for tool in current_tools:
            ...
```

**Commit**: `bc69c76`

---

### Bug 4: Wrong Tool Structure Access

**File**: `behaviors/SearchToolsBehavior.py:117-119`

**Symptom**: Incorrect access pattern for OpenAI function format

**Generated Code (BROKEN)**:
```python
for tool in current_tools:
    name = tool.get("name", "")  # WRONG: name is nested
    description = tool.get("description", "")  # WRONG: description is nested
```

**Root Cause**: Tools use OpenAI function format with nested structure:
```python
{
    "type": "function",
    "function": {
        "name": "tool_name",
        "description": "Tool description",
        ...
    }
}
```

**Fix**: Access via function wrapper
```python
for tool in current_tools:
    func = tool.get("function", {})  # Get nested function object
    name = func.get("name", "")
    description = func.get("description", "")
```

**Commit**: `bc69c76`

---

## Validation Enhancements

### New Validation Rule: validate_no_super_in_dispatch

**File**: `utils/behavior_validator.py:265-331`

**Purpose**: Automatically detect super() calls in dispatch_tool methods

**Implementation**: AST-based detection
```python
def validate_no_super_in_dispatch(code: str) -> dict[str, Any]:
    """
    Validate that dispatch_tool does not call super().dispatch_tool().

    Returns:
        dict with keys:
        - valid (bool): True if no super() calls
        - error (str): Error message if found
        - line (int): Line number of problematic call
    """
    tree = ast.parse(code)

    # Find dispatch_tool method in AgentBehavior subclasses
    # Check for super().dispatch_tool() pattern
    # Return validation error if found
```

**Integration**:
- Added to ValidationBehavior as new tool
- Wired into CreateBehaviorBehavior validation pipeline
- Now runs automatically on all generated behaviors

**Test Results**:
```python
# Bad code with super():
{'valid': False, 'error': 'dispatch_tool must not call super()...', 'line': 7}

# Good code with error return:
{'valid': True}
```

**Commit**: `692a61a`

---

## Configuration Changes

### MetaProgrammer Config: Process-Focused, Not Tool-Focused

**File**: `config/agents/meta_programmer.yaml`

**Issue**: System prompt listed specific tool names, creating potential conflicts

**Problem**:
- Tools are dynamically injected by behaviors via lifecycle hooks
- Listing specific tools in system prompts creates brittleness
- Adding/removing behaviors would require config updates

**Changes Made**:
1. Removed "AVAILABLE TOOLS" section (lines 179-206)
2. Replaced with "TOOL USAGE PHILOSOPHY" section
3. Changed "Tool-focused prompts" to "Process-focused prompts"
4. Guidance now describes capabilities, not tool names

**Before**:
```yaml
## AVAILABLE TOOLS

**Meta-programming tools** (from CreateBehaviorBehavior):
- create_behavior: Generate new behavior
- install_behavior: Install behavior
...
```

**After**:
```yaml
## TOOL USAGE PHILOSOPHY

Your available tools are dynamically provided by behaviors.
Focus on the **workflow and principles**, not specific tool names.

Key capabilities you have:
- **Generate code**: Create behaviors and agents
- **Validate code**: Check quality and compliance
- **Test code**: Run tests in sandbox
- **Read/write files**: Navigate codebase
- **Execute commands**: Run validation and tests
```

**Rationale**:
- Agent configs describe process/principles
- Behaviors provide tools dynamically
- More maintainable and composable

**Commit**: `502b32c`

---

## Summary Statistics

**Total Bugs Found**: 4 critical bugs
- 1 YAML generation bug
- 1 double-dispatch bug (architectural)
- 1 stale cache bug (future-proofing)
- 1 data structure access bug

**Files Fixed**: 7 files
- behaviors/SearchToolsBehavior.py (generated)
- behaviors/validation.py (existing)
- behaviors/create_behavior.py (generator)
- behaviors/create_agent.py (generator)
- behaviors/templates/behavior_simple_template.py
- behaviors/templates/behavior_with_tools_template.py
- config/agents/meta_programmer.yaml

**Validation Enhancements**: 1 new validation rule
- validate_no_super_in_dispatch() with AST-based detection
- Integrated into CreateBehaviorBehavior pipeline

**Commits**: 3 commits
- `a8f95e4`: Fix YAML indentation
- `bc69c76`: Fix all SearchToolsBehavior bugs and templates
- `692a61a`: Add validation rule for super() calls
- `502b32c`: Make MetaProgrammer config process-focused

---

## Lessons Learned

### 1. LLM-Generated Code Needs Review

**Finding**: Automated validation caught syntax errors but not architectural bugs

**Impact**: User code review essential for catching semantic bugs

**Action**: Added validate_no_super_in_dispatch() to catch architectural issues

### 2. Templates Are Critical

**Finding**: Bad patterns in templates propagate to all generated code

**Impact**: super() bug appeared because templates had incorrect pattern

**Action**: Updated all templates and added validation to prevent recurrence

### 3. Future-Proofing Matters

**Finding**: Stale tools cache would break future dynamic tool loading

**Impact**: Code works today but breaks when requirements evolve

**Action**: Always query current state, don't cache what might change

### 4. Tool Structure Knowledge Required

**Finding**: LLM assumed flat tool structure, OpenAI uses nested format

**Impact**: search_tools would fail to find any tools

**Action**: Document tool structure in templates and examples

### 5. Configuration Philosophy

**Finding**: Listing specific tools in system prompts creates brittleness

**Impact**: Config changes needed whenever behaviors change

**Action**: Focus configs on process/principles, let behaviors provide tools

---

## Validation Pipeline Status

### Current Validation Checks

Generated behaviors are now validated for:
1. ✅ Python syntax (AST parsing)
2. ✅ Cross-behavior independence (no imports from other behaviors)
3. ✅ Valid tool schemas (OpenAI function format)
4. ✅ Proper behavior class structure (inherits from AgentBehavior)
5. ✅ **No super() calls in dispatch_tool** (NEW)

### Testing Pipeline

Generated code goes through:
1. Syntax validation
2. Independence validation
3. Class structure validation
4. Super() dispatch validation
5. Sandbox testing (isolated pytest execution)
6. Manual review (safety_mode='review')
7. Installation to production

---

## Next Steps

### Immediate
- [x] Fix SearchToolsBehavior bugs
- [x] Update templates
- [x] Add validation rule
- [x] Update MetaProgrammer config
- [x] Document all fixes

### Future Improvements
- [ ] Add more AST-based validation rules
- [ ] Detect other antipatterns (e.g., mutable default arguments)
- [ ] Validate lifecycle hook signatures
- [ ] Check for parameter invention tolerance patterns
- [ ] Validate context enhancement implementations

---

## Test Results

### SearchToolsBehavior Generation
- ✅ Behavior code generated (173 lines)
- ✅ Test code generated
- ✅ Validation passed (after fixes)
- ✅ Lifecycle hook present: on_initial_context
- ✅ Tool method present: search_tools
- ✅ Installation completed

### AllAgent Generation
- ✅ Config generated (.agent_generated/staging/AllAgent.yaml)
- ✅ YAML structure valid (after indentation fix)
- ✅ 25 behaviors loaded
- ❌ SearchToolsBehavior not in config (naming mismatch - SearchtoolsbehaviorBehavior)

**Note**: AllAgent included "SearchtoolsbehaviorBehavior" instead of "SearchToolsBehavior" due to behavior discovery using glob patterns that normalize names. This is a separate issue to be addressed.

---

**Generated**: 2025-11-07T01:30:00
**Test**: search_tools_e2e_fixed.log
**Commits**: a8f95e4, bc69c76, 692a61a, 502b32c
