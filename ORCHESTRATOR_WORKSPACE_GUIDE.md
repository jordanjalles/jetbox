# Orchestrator Workspace Management Guide

## Overview

The Orchestrator now understands workspace behavior and can correctly route updates to existing projects instead of creating new workspaces every time.

## The Problem (Before)

**User:** "Create a calculator"
- Orchestrator delegates → Creates `.agent_workspace/create-a-calculator/`

**User:** "Add square root support to the calculator"
- Orchestrator delegates → Creates NEW workspace `.agent_workspace/add-square-root-support-to-the-calculator/`
- TaskExecutor can't find original calculator files ❌

## The Solution (Now)

**User:** "Create a calculator"
- Orchestrator delegates → Creates `.agent_workspace/create-a-calculator/`

**User:** "Add square root support to the calculator"
- Orchestrator calls `list_workspaces` → Finds `create-a-calculator/`
- Orchestrator delegates with `workspace` parameter
- TaskExecutor works in existing workspace ✅

## How It Works

### 1. Workspace Parameter

The `delegate_to_executor` tool now has an optional `workspace` parameter:

```python
delegate_to_executor(
    task_description="Add square root function",
    workspace="/path/to/.agent_workspace/create-a-calculator/"
)
```

When provided, TaskExecutor runs with `--workspace` flag:
```bash
python agent.py --workspace /path/to/workspace "task description"
```

### 2. List Workspaces Tool

Orchestrator can list all existing workspaces:

```python
list_workspaces()
```

Returns:
```
Found 14 workspace(s):

- create-a-calculator/
  Path: /workspace/.agent_workspace/create-a-calculator/
  Files (3): calculator.py, test_calculator.py, README.md

- create-a-web-app/
  Path: /workspace/.agent_workspace/create-a-web-app/
  Files (5): index.html, script.js, style.css ... and 2 more
```

### 3. Intelligent System Prompt

Orchestrator's system prompt now includes workspace decision tree:

```
WORKSPACE DECISION TREE:
1. User says "create/make/build NEW thing" → Omit workspace (creates new)
2. User says "update/modify/add to EXISTING thing" → Call list_workspaces, then specify workspace
3. User says "fix bug in X" or "improve X" → Find X's workspace and specify it
4. Not sure? → Ask user or call list_workspaces to check
```

## Usage Examples

### Example 1: Creating and Updating

```
User: Create a simple todo app in Python

Orchestrator: → Delegating to TaskExecutor...
[Creates .agent_workspace/create-a-simple-todo-app-in-python/]

TaskExecutor: ✅ Created todo.py with add/list/remove functions
Orchestrator: Task completed! Files at .agent_workspace/create-a-simple-todo-app-in-python/

---

User: Add a priority system to the todo app

Orchestrator: Checking existing workspaces...
[Calls list_workspaces, finds "create-a-simple-todo-app-in-python"]

Orchestrator: → Delegating to TaskExecutor (existing workspace)...
[Delegates with workspace parameter]

TaskExecutor: ✅ Updated todo.py with priority support
Orchestrator: Task completed! Updated files in same workspace.
```

### Example 2: Ambiguous Request

```
User: Fix the bug in the calculator

Orchestrator: I see multiple calculator workspaces. Which one?
  1. create-a-calculator/
  2. create-an-html-calculator/
  3. simple-calculator-cli/

User: The first one

Orchestrator: → Delegating to TaskExecutor (workspace: create-a-calculator/)...
[Delegates with workspace parameter]
```

### Example 3: Manual Workspace Specification

```
User: Update the files in .agent_workspace/my-project/

Orchestrator: → Delegating to TaskExecutor (workspace: .agent_workspace/my-project/)...
[Uses specified workspace]
```

## Technical Details

### Workspace Parameter Flow

1. **Orchestrator receives user request**
   - Analyzes intent (create new vs. update existing)

2. **Decision: Update existing**
   - Calls `list_workspaces` tool
   - Matches user intent to workspace name
   - Selects correct workspace path

3. **Delegation**
   - Calls `delegate_to_executor` with workspace parameter
   - orchestrator_main.py builds command: `python agent.py --workspace <path> "task"`

4. **TaskExecutor**
   - Runs in specified workspace (EDIT mode)
   - Can access all existing files
   - Preserves existing work

### Default Behavior

**Without workspace parameter:**
- TaskExecutor creates NEW isolated workspace
- Workspace name derived from task description
- Path: `.agent_workspace/{slugified-task-description}/`

**With workspace parameter:**
- TaskExecutor uses EXISTING workspace
- All files in workspace are accessible
- Can modify/update existing files

## Benefits

1. **Continuity** - Updates apply to correct project
2. **No confusion** - Files don't get scattered across workspaces
3. **Natural language** - "Update the calculator" just works
4. **Visibility** - User can see all workspaces with `list_workspaces`
5. **Explicit control** - User can specify workspace path directly

## Testing

Run the test suite:
```bash
python test_orchestrator_workspace.py
```

Tests:
- ✅ List workspaces tool
- ✅ Workspace parameter passing
- ✅ Command construction with --workspace flag

## Configuration

### System Prompt Customization

The workspace behavior is controlled by the system prompt in `orchestrator_agent.py:148-193`.

Key sections:
- `WORKSPACE BEHAVIOR (CRITICAL)` - Explains default behavior
- `WORKSPACE DECISION TREE` - Decision logic for workspace selection

### Tool Definitions

**delegate_to_executor:**
```python
{
    "task_description": str,  # Required
    "context": str,           # Optional
    "workspace": str,         # Optional - path to existing workspace
}
```

**list_workspaces:**
```python
{}  # No parameters
```

Returns:
```python
{
    "success": bool,
    "workspaces": [
        {
            "name": str,
            "path": str,
            "files": [str],
            "file_count": int
        }
    ],
    "message": str
}
```

## Troubleshooting

### Orchestrator still creates new workspace

**Problem:** User says "update X" but orchestrator creates new workspace

**Solutions:**
1. Check if workspace exists: Call `list_workspaces` manually
2. Be more explicit: "Update the files in workspace X"
3. Specify full path: "Update .agent_workspace/X/"

### Can't find the workspace

**Problem:** Workspace exists but orchestrator can't find it

**Solutions:**
1. List workspaces to see actual names (may be slugified)
2. Use exact workspace name from list_workspaces output
3. Provide full path explicitly

### Files not accessible in TaskExecutor

**Problem:** TaskExecutor can't see existing files

**Check:**
1. Verify workspace parameter was passed correctly
2. Check orchestrator output for `[orchestrator] Using existing workspace: ...`
3. Ensure files exist in specified workspace
4. Check file permissions

## Implementation Files

- `orchestrator_agent.py` - Tool definitions and system prompt
- `orchestrator_main.py` - Tool execution and workspace listing
- `test_orchestrator_workspace.py` - Test suite
- `agent.py` - TaskExecutor with --workspace support

## Future Enhancements

Potential improvements:

1. **Workspace search** - Fuzzy matching for workspace names
2. **Workspace metadata** - Track project type, creation date, last modified
3. **Workspace suggestions** - Proactively suggest workspace when ambiguous
4. **Workspace cleanup** - Tool to archive/delete old workspaces
5. **Cross-workspace operations** - Copy files between workspaces
6. **Workspace templates** - Start new projects from templates

## Version History

- **v2.2.1** - Initial workspace management implementation
  - Added `workspace` parameter to `delegate_to_executor`
  - Added `list_workspaces` tool
  - Updated system prompt with workspace decision tree
  - Added comprehensive testing

## See Also

- `ORCHESTRATOR_TEST_RESULTS.md` - Test results and benchmarks
- `ORCHESTRATOR_TEST_FINDINGS.md` - Original issue analysis
- `MULTI_AGENT_ARCHITECTURE.md` - Multi-agent system overview
- `README.md` - Main project documentation
