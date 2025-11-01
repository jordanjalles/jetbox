# Dynamic Tool Documentation - Before/After Comparison

Date: 2025-11-01

## Overview

This document shows the before/after comparison of system prompt generation with hardcoded vs. dynamic tool documentation.

## Task Executor Config

### BEFORE (Hardcoded in Config)

```yaml
system_prompt: |
  You are a local coding agent that helps build software projects.

  Guidelines:
  - ALWAYS use tools - never just respond with text
  - ...

  Core tools available:
  - write_file(path, content, append=False, encoding="utf-8", line_end=None, overwrite=True): Write/overwrite files
  - read_file(path, encoding="utf-8", max_size=1000000): Read files (up to 1MB by default)
  - list_dir(path): List directory contents
  - run_bash(command, timeout=60): Run ANY shell command with full bash features

  Common operations:
  - write_file("file.py", "import sys\n\nprint('hello')")  # Write file
  - ...
```

**Problems**:
- Tool documentation hardcoded in config
- Must manually update when behaviors change
- Duplication between config and behavior tool definitions
- Adding/removing behaviors requires config edits

### AFTER (Dynamic Generation)

```yaml
system_prompt: |
  You are a local coding agent that helps build software projects.

  Guidelines:
  - ALWAYS use tools - never just respond with text
  - ...

  # Tool documentation is dynamically generated based on loaded behaviors
```

**Implementation**:

1. **base_agent.py** - `generate_tool_documentation()` method:
```python
def generate_tool_documentation(self) -> str:
    """Generate tool documentation from loaded behaviors."""
    if not self.behaviors:
        return ""

    tool_docs = []
    for behavior in self.behaviors:
        tools = behavior.get_tools()
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            # Build parameter signature
            param_strs = []
            for param_name, param_spec in params.items():
                param_type = param_spec.get("type", "any")
                if param_name in required:
                    param_strs.append(f"{param_name}: {param_type}")
                else:
                    default = param_spec.get("default")
                    if default is not None:
                        param_strs.append(f"{param_name}: {param_type} = {default}")
                    else:
                        param_strs.append(f"{param_name}?: {param_type}")

            param_sig = ", ".join(param_strs) if param_strs else ""
            tool_docs.append(f"  - {name}({param_sig}): {desc}")

    if tool_docs:
        return "\n\nAvailable tools:\n" + "\n".join(tool_docs)
    return ""
```

2. **task_executor_agent.py** - `get_system_prompt()` update:
```python
def get_system_prompt(self) -> str:
    base_prompt = self.config_system_prompt if self.config_system_prompt else config.llm.system_prompt
    parts = [base_prompt]

    if self.use_behaviors:
        behavior_instructions = self.get_behavior_instructions()
        if behavior_instructions:
            parts.append(behavior_instructions)

        # NEW: Add dynamic tool documentation
        tool_docs = self.generate_tool_documentation()
        if tool_docs:
            parts.append(tool_docs)

    return "\n".join(parts)
```

**Benefits**:
- ✓ Tool documentation automatically generated from loaded behaviors
- ✓ Adding/removing behaviors automatically updates tool list
- ✓ Single source of truth (behavior tool definitions)
- ✓ No manual config updates needed
- ✓ Consistent formatting across all agents

## Architect Config

### BEFORE (Hardcoded in Config)

```yaml
system_prompt: |
  ## Available Tools

  - **write_architecture_doc(title, content)**: Write high-level architecture document
  - **write_module_spec(module_name, responsibility, interfaces, dependencies, technologies, implementation_notes)**: Write detailed module specification
  - **write_task_list(tasks)**: Write structured task breakdown (JSON) for orchestrator
  - **list_architecture_docs()**: List existing architecture documents
  - **read_architecture_doc(file_path)**: Read existing architecture document

  ## Output Format
  ...
```

### AFTER (Dynamic Generation)

```yaml
system_prompt: |
  # Tool documentation is dynamically generated based on loaded behaviors

  ## Output Format
  ...
```

**Same benefits as TaskExecutor**.

## Orchestrator Config

### BEFORE

Orchestrator config was already minimal and used the behavior system, but didn't have dynamic tool docs.

### AFTER

Now includes dynamic tool documentation generation via `generate_tool_documentation()`.

## Sample Generated Output

For TaskExecutor with FileToolsBehavior, CommandToolsBehavior, ServerToolsBehavior:

```
Available tools:
  - write_file(path: string, content: string, append?: boolean, encoding?: string, line_end?: string, overwrite?: boolean): Write or append content to a file
  - read_file(path: string, encoding?: string, max_size?: number): Read file contents (up to max_size bytes)
  - list_dir(path: string): List directory contents (files and subdirectories)
  - run_bash(command: string, timeout?: number): Execute bash command with timeout
  - start_server(port: number, command: string, name?: string): Start a background server process
  - stop_server(name: string): Stop a running server by name
  - check_server(name: string): Check if a server is running
  - list_servers(): List all running servers
```

**Format**:
- Function name with typed parameter signature
- Required params: `param: type`
- Optional params: `param?: type`
- Params with defaults: `param: type = default`
- Description from tool definition

## Testing

All tests pass:
```
tests/test_core_integration_final.py::TestDynamicToolDocumentation::test_generate_tool_documentation PASSED
tests/test_core_integration_final.py::TestDynamicToolDocumentation::test_system_prompt_includes_tool_docs PASSED
```

## Migration Impact

**Config files updated**:
1. `/workspace/task_executor_config.yaml` - Removed ~15 lines of hardcoded tool docs
2. `/workspace/architect_config.yaml` - Removed ~5 lines of hardcoded tool docs
3. `/workspace/orchestrator_config.yaml` - No changes needed (already minimal)

**Code files updated**:
1. `/workspace/base_agent.py` - Added `generate_tool_documentation()` method
2. `/workspace/task_executor_agent.py` - Updated `get_system_prompt()` to use dynamic docs
3. `/workspace/architect_agent.py` - Updated `get_system_prompt()` to use dynamic docs
4. `/workspace/orchestrator_agent.py` - Updated `get_system_prompt()` to use dynamic docs

**No breaking changes** - Old configs still work if they have hardcoded tools (will just have duplicates until cleaned up).

## Conclusion

Dynamic tool documentation generation successfully implemented. All agents now generate tool lists automatically from loaded behaviors, eliminating hardcoded duplication and manual maintenance.
