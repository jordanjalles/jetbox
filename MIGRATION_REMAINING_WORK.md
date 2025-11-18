# Remaining Tool Decorator Migrations

## Quick Reference: 10 behaviors, 31 tools remaining

All behaviors below have imports and docstrings already updated. Just need to remove boilerplate and convert tool methods.

---

## 1. execution_mode.py (1 tool) - SIMPLE

**Tool:** `select_execution_mode`

**Pattern:**
```python
# Remove get_tools() at line ~434
# Remove dispatch_tool() at line ~462

# Convert to:
@tool(description="Change execution mode between verify_first, exploratory, or autonomous")
def select_execution_mode(self, mode: str) -> dict[str, Any]:
    """Args: mode: One of 'verify_first', 'exploratory', 'autonomous'"""
    # Implementation already exists in current method
```

---

## 2. create_agent.py (1 tool) - SIMPLE

**Tool:** `create_agent_config`

**Pattern:** Similar to execution_mode, single tool with simple implementation

---

## 3. create_behavior.py (1 tool) - SIMPLE

**Tool:** `create_behavior_file`

**Pattern:** Single tool, may have complex implementation but straightforward conversion

---

## 4. sandbox_test.py (2 tools) - SIMPLE

**Tools:**
1. `create_test_file`
2. `run_sandbox_test`

**Pattern:** Two tools, both straightforward

---

## 5. chatbot.py (3 tools) - MEDIUM

**Tools:**
1. `set_goal`
2. `clarify_with_user`
3. `activate_chat_mode`

**Notes:** May have user interaction logic to preserve

---

## 6. home_assistant.py (5 tools) - MEDIUM

**Tools:**
1. `ha_list_devices`
2. `ha_get_state`
3. `ha_call_service`
4. `ha_list_automations`
5. `ha_trigger_automation`

**Notes:** API integration behavior, has _api_request helper to keep

---

## 7. architect_tools.py (5 tools) - MEDIUM

**Tools:**
1. `create_architecture_doc`
2. `update_architecture_doc`
3. `create_design_decision`
4. `create_component_spec`
5. `list_architecture_artifacts`

**Notes:** File-based tool, workspace-aware

---

## 8. task_management.py (4 tools) - MEDIUM

**Tools:**
1. `mark_subtask_complete`
2. `mark_blocked`
3. `update_task_progress`
4. `add_subtask`

**Notes:** State management, has helper methods to preserve

---

## 9. workspace_management.py (4 tools) - MEDIUM

**Tools:**
1. `create_workspace_subdirectory`
2. `get_workspace_path`
3. `list_workspace_files`
4. `workspace_tree`

**Notes:** Workspace operations, path resolution logic

---

## 10. validation.py (8 tools) - COMPLEX

**Tools:**
1. `validate_python_syntax`
2. `validate_json`
3. `validate_yaml`
4. `validate_markdown`
5. `validate_file_exists`
6. `validate_directory_structure`
7. `run_pytest`
8. `run_ruff`

**Notes:** Most complex, 8 different validation tools with various implementations

---

## Migration Steps for Each

1. Open the behavior file
2. Locate `get_tools()` method - note tool names and descriptions
3. Locate `dispatch_tool()` method - note how tools are dispatched
4. Find implementation methods (usually _tool_name format)
5. Remove get_tools() entirely
6. Remove dispatch_tool() entirely
7. Convert each _tool_name to tool_name with @tool decorator
8. Add type hints to all parameters
9. Add Google-style docstring with Args section
10. Replace agent parameter access with self.agent
11. Test the behavior loads and generates tools

## Quick Test Command

After each migration:
```python
from behaviors.BEHAVIOR_NAME import BehaviorClassName
b = BehaviorClassName()
tools = b.get_tools()
print(f"{b.get_name()}: {len(tools)} tools")
for t in tools:
    print(f"  - {t['function']['name']}")
```

## Estimated Time

- Simple (1-2 tools): 10-15 min each = ~1 hour for 4 behaviors
- Medium (3-5 tools): 20-30 min each = ~2.5 hours for 5 behaviors
- Complex (8 tools): 45 min = ~0.75 hours for 1 behavior

**Total:** ~4-5 hours of focused work

## Git Commit Strategy

Batch commits by complexity:
1. Commit all 4 simple behaviors together
2. Commit medium behaviors in pairs
3. Commit validation.py separately (complex)

---

## Files Ready for Migration

All these already have `from behaviors.tool_decorator import tool`:
- ✅ behaviors/execution_mode.py
- ✅ behaviors/create_agent.py
- ✅ behaviors/create_behavior.py
- ✅ behaviors/sandbox_test.py
- ✅ behaviors/chatbot.py
- ✅ behaviors/home_assistant.py
- ✅ behaviors/architect_tools.py
- ✅ behaviors/task_management.py
- ✅ behaviors/workspace_management.py
- ✅ behaviors/validation.py
