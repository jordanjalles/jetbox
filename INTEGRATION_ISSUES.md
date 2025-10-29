# Integration Issues Found

## Critical Issues (Need Fixing)

### 1. Tool Definitions Duplicated
**File:** `task_executor_agent.py`
**Line:** 93-184

**Issue:** `get_tools()` manually defines all tools instead of calling `tools.get_tool_definitions()`.

**Impact:**
- Tool definitions must be maintained in TWO places
- If a tool is added to `tools.py`, must remember to add it to `task_executor_agent.py`
- Descriptions can drift out of sync

**Current State:**
- `tools.py` defines 11 tools (including grep_file and server tools)
- `task_executor_agent.py.get_tools()` only defines 6 tools
- Missing from get_tools(): grep_file, start_server, stop_server, check_server, list_servers

**Solution:**
```python
def get_tools(self) -> list[dict[str, Any]]:
    """Return tools available to TaskExecutor."""
    return tools.get_tool_definitions()
```

**Risk:** LOW - server tools and grep_file are in dispatch map but not advertised to LLM


### 2. orchestrator_status.py Not Integrated
**File:** `orchestrator_status.py` (entire file)

**Issue:** File exists with `OrchestratorStatusDisplay` class but is never imported or used.

**Evidence:**
```bash
$ grep -r "OrchestratorStatus" --include="*.py" .
orchestrator_status.py:class OrchestratorStatusDisplay:
```

**Impact:**
- Dead code in repository
- Orchestrator has no status display (unlike TaskExecutor)
- Architecture diagram claims it exists but it's not used

**Solution:**
- Either integrate into orchestrator_agent.py
- Or delete if not needed


### 3. prompt_loader.py Not Used
**File:** `prompt_loader.py`

**Issue:** Module exists with `load_prompts()` function but is never imported.

**Current State:**
- prompts.yaml exists
- prompt_loader.py can load prompts from YAML
- But no code calls it

**Impact:**
- prompts.yaml is unused
- Agent prompts are hardcoded in agent_config.yaml instead

**Solution:**
- Delete prompt_loader.py and prompts.yaml if not needed
- Or integrate into agent_config.py to load prompts from YAML


## Medium Issues (Consider Fixing)

### 4. grep_file Tool Missing from TaskExecutor
**File:** `task_executor_agent.py`
**Line:** 81-184

**Issue:** `grep_file` tool is defined in tools.py and in dispatch map, but not in get_tools() list.

**Impact:**
- LLM doesn't know grep_file exists
- Tool works if called, but LLM won't call it

**Solution:**
- Add grep_file to get_tools() return value (or use tools.get_tool_definitions())


### 5. Server Tools Missing from TaskExecutor Docstring
**File:** `task_executor_agent.py`
**Line:** 81-92

**Issue:** Docstring lists 6 tools but get_tools() could return 11.

**Current docstring:**
```python
"""
Tools:
- write_file: Write content to a file
- read_file: Read file contents
- list_dir: List directory contents
- run_cmd: Execute shell commands (whitelisted)
- mark_subtask_complete: Mark current subtask as done
- decompose_task: Break task into subtasks
"""
```

**Missing from docstring:**
- grep_file
- start_server, stop_server, check_server, list_servers

**Impact:**
- Documentation out of sync with reality
- Developers might not know these tools exist


## Low Priority Issues (Minor)

### 6. Unused Utility Scripts
**Files:** `diag_speed.py`, `dsl.py`, `sitecustomize.py`

**Issue:** Scripts exist but are never imported.

**Analysis:**
- `diag_speed.py` - Standalone utility for diagnosing Ollama speed
- `dsl.py` - Experiment with domain-specific language (incomplete)
- `sitecustomize.py` - Python startup customization

**Impact:** None - these are utilities/experiments, not library code

**Solution:** Document as utilities or move to tools/ directory


### 7. agent_legacy.py Imports Not All in New Agent
**Missing imports:**
- `argparse` - CLI parsing (agent.py handles this now)
- `threading` - Used for server management (still works via server_manager)
- `traceback` - Error handling (could add for better error messages)
- `io` - String buffer operations (not needed)

**Impact:** LOW - most are legacy CLI features


## Fixed Issues (Already Resolved)

### ✅ completion_detector Integration
**Status:** FIXED in commit d348878

**Was:** completion_detector existed but wasn't called in task_executor_agent.py
**Now:** analyze_llm_response() called after each LLM response
**Impact:** Agent now properly signals task completion


## Summary Statistics

- **Critical issues:** 3 (tool duplication, orchestrator_status, prompt_loader)
- **Medium issues:** 2 (grep_file missing, docstring out of sync)
- **Low priority:** 2 (unused utilities, legacy imports)
- **Fixed:** 1 (completion_detector)

## Recommended Actions

1. **HIGH PRIORITY:** Replace manual tool definitions with `tools.get_tool_definitions()`
2. **HIGH PRIORITY:** Delete or integrate `orchestrator_status.py`
3. **MEDIUM:** Delete or integrate `prompt_loader.py` and `prompts.yaml`
4. **LOW:** Update docstrings to match actual tool availability
5. **LOW:** Document utility scripts or move to separate directory
